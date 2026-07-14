#!/usr/bin/env python3
"""过滤 / 重命名 TensorBoard events 中的 scalar tags。

用于解决 Megatron-Bridge training_log 默认写入 batch-size / loss-scale /
iteration-time 等冗余指标, 以及指标命名与 ms-swift 不一致的问题。

用法:
    python filter_tb_events.py \
        --input /data/wanghui/deepseekv4_train_mcore_bridge/model_output/phase2_recipe_callback/tb_logs \
        --output /data/wanghui/deepseekv4_train_mcore_bridge/model_output/phase2_recipe_callback/tb_logs_filtered

可选:
    --remove-tags  tag1 tag2          额外删除的 tags
    --rename       old=new old2=new2  自定义重命名规则
    --no-default-remove               不删除默认冗余 tags
    --no-rename                       不重命名
"""
from __future__ import annotations

import argparse
import os
import shutil
from collections.abc import Sequence
from typing import Dict, List, Tuple

from tensorboard.backend.event_processing import event_accumulator
from tensorboard.summary.writer.event_file_writer import EventFileWriter
from tensorboard.compat.proto import event_pb2
from tensorboard.compat.proto import summary_pb2


# 默认会被删除的冗余 tags (Bridge training_log 默认写入, 但通常不需要看)
DEFAULT_REMOVE_TAGS: set = {
    "batch-size",
    "batch-size vs samples",
    "loss-scale",
    "loss-scale vs samples",
    "iteration-time",
    "elapsed time per iteration (ms)",
    # 以 samples 为横轴的重复曲线通常不需要, 保留以 iteration 为横轴的即可
    "grad-norm vs samples",
    "learning-rate vs samples",
    "lm loss vs samples",
}

# 默认重命名: Bridge 命名 -> ms-swift 风格命名
DEFAULT_RENAME: Dict[str, str] = {
    "lm loss": "train/loss",
    "grad-norm": "train/grad_norm",
    "learning-rate": "train/learning_rate",
    "z_loss": "train/z_loss",
    "seq_load_balancing_loss": "train/seq_load_balancing_loss",
    "load_balancing_loss": "train/load_balancing_loss",
    "global_load_balancing_loss": "train/global_load_balancing_loss",
    "memory(GiB)": "train/memory(GiB)",
    "train_speed(s_it)": "train/train_speed(s_it)",
    # MoE 指标: moe/router_entropy -> train/moe_router_entropy
    "moe/router_entropy": "train/moe_router_entropy",
    "moe/router_entropy/max": "train/moe_router_entropy_max",
    "moe/router_entropy/min": "train/moe_router_entropy_min",
    "moe/expert_utilization": "train/moe_expert_utilization",
    "moe/expert_utilization/max": "train/moe_expert_utilization_max",
    "moe/expert_utilization/min": "train/moe_expert_utilization_min",
    "moe/num_active_experts": "train/moe_num_active_experts",
    "moe/num_active_experts/max": "train/moe_num_active_experts_max",
    "moe/num_active_experts/min": "train/moe_num_active_experts_min",
    "moe/top1_prob_mean": "train/moe_top1_prob_mean",
    "moe/top1_prob_mean/max": "train/moe_top1_prob_mean_max",
    "moe/top1_prob_mean/min": "train/moe_top1_prob_mean_min",
    "moe/top1_prob_std": "train/moe_top1_prob_std",
    "moe/top1_prob_std/max": "train/moe_top1_prob_std_max",
    "moe/top1_prob_std/min": "train/moe_top1_prob_std_min",
    "moe/top2_prob_mean": "train/moe_top2_prob_mean",
    "moe/top2_prob_mean/max": "train/moe_top2_prob_mean_max",
    "moe/top2_prob_mean/min": "train/moe_top2_prob_mean_min",
    "moe/per_expert_token_std": "train/moe_per_expert_token_std",
    "moe/per_expert_token_std/max": "train/moe_per_expert_token_std_max",
    "moe/per_expert_token_std/min": "train/moe_per_expert_token_std_min",
    "moe/router_z_loss": "train/moe_router_z_loss",
    "moe/router_z_loss/max": "train/moe_router_z_loss_max",
    "moe/router_z_loss/min": "train/moe_router_z_loss_min",
}


def parse_rename(args: Sequence[str] | None) -> Dict[str, str]:
    """解析命令行 --rename old=new old2=new2。"""
    result: Dict[str, str] = {}
    if not args:
        return result
    for item in args:
        if "=" not in item:
            raise ValueError(f"--rename 参数格式错误, 应为 old=new: {item}")
        old, new = item.split("=", 1)
        result[old.strip()] = new.strip()
    return result


def discover_events(input_dir: str) -> List[Tuple[str, str]]:
    """返回 [(相对路径, 绝对路径)] 的事件文件列表。"""
    events: List[Tuple[str, str]] = []
    for root, _dirs, files in os.walk(input_dir):
        for f in files:
            if f.startswith("events.out.tfevents"):
                abs_path = os.path.join(root, f)
                rel_path = os.path.relpath(abs_path, input_dir)
                events.append((rel_path, abs_path))
    return events


def process_events(
    input_dir: str,
    output_dir: str,
    remove_tags: set,
    rename_map: Dict[str, str],
) -> None:
    """读取 input_dir 下所有 events, 过滤/重命名后写到 output_dir。"""
    os.makedirs(output_dir, exist_ok=True)
    events = discover_events(input_dir)
    if not events:
        print(f"[WARN] 在 {input_dir} 下未找到 events 文件")
        return

    print(f"发现 {len(events)} 个 events 文件")
    for rel_path, abs_path in events:
        print(f"\n处理: {rel_path}")
        out_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        ea = event_accumulator.EventAccumulator(
            abs_path, size_guidance={event_accumulator.SCALARS: 0}
        )
        ea.Reload()

        writer = EventFileWriter(out_path)
        scalar_tags = ea.Tags().get("scalars", [])
        kept = 0
        removed = 0
        renamed = 0

        for tag in scalar_tags:
            if tag in remove_tags:
                removed += 1
                continue
            new_tag = rename_map.get(tag, tag)
            if new_tag != tag:
                renamed += 1

            events_for_tag = ea.Scalars(tag)
            for ev in events_for_tag:
                # 构造新的 summary
                summary = summary_pb2.Summary()
                summary_value = summary.value.add()
                summary_value.tag = new_tag
                summary_value.simple_value = ev.value
                summary_value.metadata.plugin_data.plugin_name = "scalars"

                new_event = event_pb2.Event(
                    wall_time=ev.wall_time,
                    step=ev.step,
                    summary=summary,
                )
                writer.add_event(new_event)
            kept += 1

        writer.close()
        print(f"  kept={kept}, removed={removed}, renamed={renamed}")

    print(f"\n输出目录: {output_dir}")
    print(f"查看命令: tensorboard --logdir {output_dir} --host 0.0.0.0 --port 6006")


def main() -> None:
    parser = argparse.ArgumentParser(description="过滤并重命名 TensorBoard events 的 scalar tags")
    parser.add_argument("--input", "-i", required=True, help="输入 tb_logs 目录")
    parser.add_argument("--output", "-o", required=True, help="输出 tb_logs 目录")
    parser.add_argument(
        "--remove-tags",
        nargs="+",
        default=[],
        help="额外删除的 tags (空格分隔)",
    )
    parser.add_argument(
        "--rename",
        nargs="+",
        default=[],
        help="自定义重命名规则, 格式 old=new (空格分隔)",
    )
    parser.add_argument(
        "--no-default-remove",
        action="store_true",
        help="不删除默认冗余 tags",
    )
    parser.add_argument(
        "--no-rename",
        action="store_true",
        help="不重命名 tags",
    )
    args = parser.parse_args()

    remove_tags: set = set()
    if not args.no_default_remove:
        remove_tags.update(DEFAULT_REMOVE_TAGS)
    remove_tags.update(args.remove_tags)

    rename_map: Dict[str, str] = {}
    if not args.no_rename:
        rename_map.update(DEFAULT_RENAME)
    rename_map.update(parse_rename(args.rename))

    process_events(args.input, args.output, remove_tags, rename_map)


if __name__ == "__main__":
    main()

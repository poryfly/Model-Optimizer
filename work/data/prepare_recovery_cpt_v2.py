#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recovery CPT 数据预处理脚本 v2
==============================
针对 DeepSeek-V4-Flash 4B-A1.5B MoE 裁剪模型的治愈预训练数据准备。

核心特性：
  1. 支持 .jsonl.gz 压缩格式源文件
  2. 字节偏移索引实现 O(1) 随机行读取，峰值内存 ~500MB（非全量加载）
  3. 三阶段配比数据预构建，单次训练自动经历多阶段
  4. 流式写出，避免内存溢出

用法：
  python prepare_recovery_cpt_v2.py \
    --manifest /path/to/manifest.json \
    --output_dir /data/cpt/recovery_cpt \
    --total_steps 100000 \
    --batch_size_per_step 2 \
    --seed 42

输出：
  {output_dir}/recovery_cpt_merged.jsonl       # 合并后的训练数据
  {output_dir}/recovery_cpt_merged.meta.json   # 阶段边界元信息
"""

import argparse
import gzip
import json
import os
import random
import shutil
import struct
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path
from glob import glob
from typing import Dict, List, Optional, Tuple

# ============================================================
# 第一部分：源文件管理与偏移索引
# ============================================================

def decompress_to_single(gz_path: str, tmp_dir: str) -> str:
    """
    将 .jsonl.gz 解压为临时目录下的单个 .jsonl 文件。
    返回解压后的文件路径。
    """
    gz_path = os.path.abspath(gz_path)
    out_path = os.path.join(tmp_dir, os.path.basename(gz_path).replace('.gz', ''))

    if os.path.exists(out_path):
        # 检查大小是否合理（非空）
        if os.path.getsize(out_path) > 0:
            print(f"  [跳过解压] 已存在: {out_path}")
            return out_path
        else:
            os.remove(out_path)

    print(f"  [解压中] {gz_path} -> {out_path}")
    bytes_in = 0
    with gzip.open(gz_path, 'rb') as fin, open(out_path, 'wb') as fout:
        while True:
            chunk = fin.read(64 * 1024 * 1024)  # 64MB chunks
            if not chunk:
                break
            fout.write(chunk)
            bytes_in += len(chunk)
            if bytes_in % (10 * 1024 * 1024 * 1024) == 0:
                print(f"    已解压 {bytes_in / (1024**3):.1f} GB ...")

    size_gb = os.path.getsize(out_path) / (1024**3)
    print(f"  [解压完成] {out_path} ({size_gb:.2f} GB)")
    return out_path


def build_offset_index(jsonl_path: str) -> Tuple[List[int], int]:
    """
    构建字节偏移索引。扫描 jsonl 文件，记录每一行的起始字节位置。
    返回 (offsets_list, num_lines)。
    
    内存占用：约 num_lines * 8 bytes (int64)。
    例如 1 亿行 ≈ 800MB，但实际远少于此（每行很长）。
    """
    offsets = []
    with open(jsonl_path, 'rb') as f:
        # 第一行偏移一定是 0
        offsets.append(f.tell())
        line_count = 1
        while True:
            line = f.readline()
            if not line:
                break
            if line.endswith(b'\n'):
                offsets.append(f.tell())
                line_count += 1
            # 如果最后一行没有换行符，仍然记录（作为最后一行的偏移）

    # 如果最后一行没有换行符，最后一个偏移是文件末尾（已由 tell 记录）
    # 但我们不需要那个哨兵偏移，因为它是文件末尾而非新行起始
    # 实际上 readline 后 tell 指向下一行起始，如果没有下一行则是 EOF
    # 所以 offsets 长度 = line_count（对于有换行结尾的文件）
    # 或 line_count + 1（对于无换行结尾的文件，多了一个 EOF 偏移）
    if offsets[-1] == os.path.getsize(jsonl_path) and line_count > 0:
        # 最后一个偏移是 EOF，去掉
        offsets = offsets[:-1]
        # 但 line_count 已经多算了 1
        line_count -= 1

    return offsets, line_count


def read_line_by_offset(jsonl_path: str, offset: int) -> str:
    """根据字节偏移量读取一行（O(1) seek）"""
    with open(jsonl_path, 'rb') as f:
        f.seek(offset)
        line = f.readline()
    return line.decode('utf-8', errors='replace').strip()


# ============================================================
# 第二部分：数据源注册与训练计划计算
# ============================================================

# 数据源配置 —— 基于实际 manifest.json
# key = 数据源名称（用于阶段配比引用）
# value = (manifest_key, jsonl_gz_file_pattern)
DATA_SOURCE_CONFIG = OrderedDict({
    'zh_multi_style': {
        'manifest_key': 'zh_multi_style',
        'total_tokens': 1.26e9,
        'desc': '中文多风格语料',
    },
    'zh_qa_synthetic': {
        'manifest_key': 'zh_qa_synthetic',
        'total_tokens': 2.72e9,
        'desc': '中文QA合成语料',
    },
    'en_multi_style': {
        'manifest_key': 'en_multi_style',
        'total_tokens': 6.13e9,
        'desc': '英文多风格语料',
    },
    'en_qa_synthetic': {
        'manifest_key': 'en_qa_synthetic',
        'total_tokens': 12.86e9,
        'desc': '英文QA合成语料',
    },
    'open_web_math': {
        'manifest_key': 'open_web_math',
        'total_tokens': 16.19e9,
        'desc': '数学推理语料',
    },
})

# 总 token 数
TOTAL_TOKENS = sum(src['total_tokens'] for src in DATA_SOURCE_CONFIG.values())
# 39.16B tokens


def compute_phase_plan(total_steps: int, samples_per_step: int) -> Dict:
    """
    计算三阶段训练计划。

    三阶段设计：
      Phase-1a (0%  ~ 30%): 路由稳定期 — math 压至 20%，multi_style 放大至 48%
      Phase-1b (30% ~ 70%): 全面恢复期 — math 回升至 33%，全品类均衡
      Phase-1c (70% ~100%): 能力增强期 — math 提升至 39%，QA 增强至 38%

    每阶段内部各数据源按比例采样，阶段间通过数据顺序自然过渡。
    """
    phase_a_end = int(total_steps * 0.30)
    phase_b_end = int(total_steps * 0.70)

    # 各阶段每个数据源的比例（必须加起来 = 1.0）
    phase_ratios = {
        'phase_1a': {
            'zh_multi_style': 0.13,
            'zh_qa_synthetic': 0.13,
            'en_multi_style': 0.35,
            'en_qa_synthetic': 0.19,
            'open_web_math': 0.20,
        },
        'phase_1b': {
            'zh_multi_style': 0.10,
            'zh_qa_synthetic': 0.15,
            'en_multi_style': 0.22,
            'en_qa_synthetic': 0.20,
            'open_web_math': 0.33,
        },
        'phase_1c': {
            'zh_multi_style': 0.06,
            'zh_qa_synthetic': 0.07,
            'en_multi_style': 0.10,
            'en_qa_synthetic': 0.38,
            'open_web_math': 0.39,
        },
    }

    plan = {
        'total_steps': total_steps,
        'samples_per_step': samples_per_step,
        'total_samples': total_steps * samples_per_step,
        'phases': OrderedDict(),
    }

    phases = [
        ('phase_1a', 0, phase_a_end),
        ('phase_1b', phase_a_end, phase_b_end),
        ('phase_1c', phase_b_end, total_steps),
    ]

    for phase_name, step_start, step_end in phases:
        num_steps = step_end - step_start
        if num_steps <= 0:
            continue
        num_samples = num_steps * samples_per_step
        ratios = phase_ratios[phase_name]

        plan['phases'][phase_name] = {
            'step_start': step_start,
            'step_end': step_end,
            'num_steps': num_steps,
            'num_samples': num_samples,
            'ratios': ratios,
        }

        # 验证比例加和
        ratio_sum = sum(ratios.values())
        assert abs(ratio_sum - 1.0) < 1e-6, \
            f"{phase_name} ratios sum to {ratio_sum}, expected 1.0"

    return plan


# ============================================================
# 第三部分：多阶段索引序列生成
# ============================================================

def generate_global_index_sequence(
    source_indices: Dict[str, List[int]],  # src_name -> list of line indices
    plan: Dict,
    rng: random.Random,
) -> Tuple[List[str], List[int]]:
    """
    根据训练计划，预生成全局样本序列。
    
    返回：
      all_src_ids:  每个样本对应的数据源名称列表
      all_line_ids: 每个样本在对应数据源中的行索引列表
    
    关键原理：将"阶段切换"从训练时决策转移到数据构建时决策。
    jsonl 文件的前 N1 个样本对应 Phase-1a 配比，
    接下来 N2 个样本对应 Phase-1b 配比，
    最后 N3 个样本对应 Phase-1c 配比。
    训练时按顺序读取即可自然经历三个阶段。
    """
    all_src_ids = []
    all_line_ids = []

    for phase_name, phase_info in plan['phases'].items():
        num_samples = phase_info['num_samples']
        ratios = phase_info['ratios']
        print(f"  生成 {phase_name}: {num_samples} 样本")

        # 计算每个数据源在此阶段的采样数
        source_samples = {}
        for src_name, ratio in ratios.items():
            n = int(num_samples * ratio)
            source_samples[src_name] = n

        # 修正余数（确保总数精确）
        diff = num_samples - sum(source_samples.values())
        if diff != 0:
            # 将余数分配给比例最大的数据源
            max_src = max(ratios, key=ratios.get)
            source_samples[max_src] += diff

        # 对每个数据源随机采样（不放回 → 放回，因为样本量可能超过源数据行数）
        for src_name, n_samples in source_samples.items():
            if n_samples <= 0:
                continue

            available_indices = source_indices[src_name]
            n_available = len(available_indices)

            if n_samples <= n_available:
                # 不放回采样
                sampled = rng.sample(available_indices, n_samples)
            else:
                # 放回采样：先全部不重复，再用放回补齐
                sampled = list(available_indices)
                rng.shuffle(sampled)
                remaining = n_samples - n_available
                extra = rng.choices(available_indices, k=remaining)
                sampled.extend(extra)

            # 交错写入（避免同一数据源的大块连续）
            all_src_ids.extend([src_name] * n_samples)
            all_line_ids.extend(sampled)

    # === 可选：阶段内 shuffle（保持阶段边界不变） ===
    # 在每个阶段内部进行 shuffle，打破数据源内部顺序
    offset = 0
    for phase_name, phase_info in plan['phases'].items():
        n = phase_info['num_samples']
        phase_pairs = list(zip(
            all_src_ids[offset:offset + n],
            all_line_ids[offset:offset + n],
        ))
        rng.shuffle(phase_pairs)
        all_src_ids[offset:offset + n] = [p[0] for p in phase_pairs]
        all_line_ids[offset:offset + n] = [p[1] for p in phase_pairs]
        offset += n

    assert len(all_src_ids) == len(all_line_ids)
    print(f"  全局索引序列生成完成: {len(all_src_ids)} 样本")

    return all_src_ids, all_line_ids


# ============================================================
# 第四部分：流式写出
# ============================================================

def stream_write_merged(
    output_path: str,
    source_paths: Dict[str, str],      # src_name -> decompressed jsonl path
    source_offsets: Dict[str, List[int]],  # src_name -> offset list
    all_src_ids: List[str],
    all_line_ids: List[int],
    log_interval: int = 100000,
):
    """
    流式写出合并后的 jsonl 文件。
    每次只读取一行，内存占用极低。
    """
    total = len(all_src_ids)
    print(f"  开始流式写出 {total} 样本 -> {output_path}")

    with open(output_path, 'w', encoding='utf-8') as fout:
        for i in range(total):
            src_name = all_src_ids[i]
            line_idx = all_line_ids[i]

            jsonl_path = source_paths[src_name]
            offset = source_offsets[src_name][line_idx]
            line = read_line_by_offset(jsonl_path, offset)

            # 基本校验
            if not line:
                print(f"  [警告] 空行: src={src_name}, line_idx={line_idx}, offset={offset}")
                continue

            # 快速 JSON 合法性检查（跳过完整解析以提升速度）
            if not (line.startswith('{') and line.endswith('}')) and \
               not (line.startswith('[') and line.endswith(']')):
                # 尝试解析，如果失败则跳过
                try:
                    json.loads(line)
                except json.JSONDecodeError:
                    print(f"  [警告] JSON 解析失败: src={src_name}, line_idx={line_idx}")
                    continue

            fout.write(line + '\n')

            if (i + 1) % log_interval == 0:
                pct = (i + 1) / total * 100
                print(f"    进度: {i + 1}/{total} ({pct:.1f}%)")

    size_gb = os.path.getsize(output_path) / (1024**3)
    print(f"  写出完成: {output_path} ({size_gb:.2f} GB)")


# ============================================================
# 第五部分：元信息文件
# ============================================================

def write_meta_json(output_path: str, plan: Dict, source_line_counts: Dict[str, int]):
    """写出 .meta.json 元信息文件，记录阶段边界和数据源统计。"""
    meta = {
        'version': 'v2',
        'description': 'Recovery CPT multi-phase training data',
        'total_sources': len(source_line_counts),
        'sources': {},
        'phases': {},
        'total_samples': plan['total_samples'],
        'total_steps': plan['total_steps'],
        'samples_per_step': plan['samples_per_step'],
    }

    for src_name, count in source_line_counts.items():
        meta['sources'][src_name] = {
            'line_count': count,
            'total_tokens': DATA_SOURCE_CONFIG[src_name]['total_tokens'],
            'desc': DATA_SOURCE_CONFIG[src_name]['desc'],
        }

    cumulative_samples = 0
    for phase_name, phase_info in plan['phases'].items():
        meta['phases'][phase_name] = {
            'sample_start': cumulative_samples,
            'sample_end': cumulative_samples + phase_info['num_samples'],
            'step_start': phase_info['step_start'],
            'step_end': phase_info['step_end'],
            'num_samples': phase_info['num_samples'],
            'ratios': phase_info['ratios'],
        }
        cumulative_samples += phase_info['num_samples']

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"  元信息文件: {output_path}")


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Recovery CPT 数据预处理 v2')
    parser.add_argument('--manifest', type=str, required=True,
                        help='manifest.json 路径（指向各数据源的元数据）')
    parser.add_argument('--data_root', type=str, default=None,
                        help='数据文件根目录（如果 manifest 中的路径是相对路径）')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出目录')
    parser.add_argument('--total_steps', type=int, default=100000,
                        help='总训练步数（含 gradient_accumulation）')
    parser.add_argument('--batch_size_per_step', type=int, default=256,
                        help='每步的样本数 = batch_size * gradient_accumulation_steps')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--tmp_dir', type=str, default=None, help='temp dir')
    parser.add_argument('--keep_temp', action='store_true',
                        help='保留临时解压文件（调试用）')
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # --- Step 0: 准备目录 ---
    os.makedirs(args.output_dir, exist_ok=True)
    output_jsonl = os.path.join(args.output_dir, 'recovery_cpt_merged.jsonl')
    output_meta = os.path.join(args.output_dir, 'recovery_cpt_merged.meta.json')

    # 临时目录
    if args.tmp_dir:
        tmp_dir = args.tmp_dir
        os.makedirs(tmp_dir, exist_ok=True)
        auto_cleanup = False
    else:
        tmp_obj = tempfile.TemporaryDirectory(prefix='recovery_cpt_')
        tmp_dir = tmp_obj.name
        auto_cleanup = True

    try:
        # --- Step 1: 读取 manifest，定位源文件 ---
        print("=" * 60)
        print("Step 1: 读取 manifest 并定位源文件")
        print("=" * 60)

        with open(args.manifest, 'r', encoding='utf-8') as f:
            manifest = json.load(f)

        # manifest 可能是 dict（key -> metadata）或 list
        if isinstance(manifest, list):
            manifest_dict = {item.get('name', item.get('key', str(i))): item
                           for i, item in enumerate(manifest)}
        else:
            manifest_dict = manifest

        # 为每个数据源找到实际的 .jsonl.gz 文件
        source_gz_paths = {}  # src_name -> gz file path
        for src_name, src_config in DATA_SOURCE_CONFIG.items():
            manifest_key = src_config['manifest_key']
            if manifest_key not in manifest_dict:
                print(f"  [错误] manifest 中未找到 key: {manifest_key}")
                print(f"  可用的 key: {list(manifest_dict.keys())}")
                sys.exit(1)

            entry = manifest_dict[manifest_key]
            # 从 manifest entry 中找到 .jsonl.gz 文件路径
            # 常见结构：entry['files'] 是列表，或 entry['file'] 是单个路径
            gz_file = None
            if 'files' in entry and isinstance(entry['files'], list):
                for f in entry['files']:
                    if f.endswith('.jsonl.gz'):
                        gz_file = f
                        break
            elif 'file' in entry:
                gz_file = entry['file']
            elif 'path' in entry:
                gz_file = entry['path']
            elif 'output_dir' in entry:
                gz_file = entry['output_dir']
            else:
                # 尝试从 entry 中找任何 .jsonl.gz 后缀的值
                for v in entry.values():
                    if isinstance(v, str) and v.endswith('.jsonl.gz'):
                        gz_file = v
                        break
            print(f"entry: {entry}",flush=True)
            if gz_file is None:
                print(f"  [错误] 无法从 manifest 中定位 {manifest_key} 的 .jsonl.gz 文件")
                print(f"  entry 内容: {json.dumps(entry, indent=2, ensure_ascii=False)[:500]}")
                sys.exit(1)

            # 处理相对路径
            if args.data_root and not os.path.isabs(gz_file):
                gz_file = os.path.join(args.data_root, gz_file)

            if not os.path.exists(gz_file):
                print(f"  [错误] 文件不存在: {gz_file}")
                sys.exit(1)

            source_gz_paths[src_name] = gz_file
            size_gb = os.path.getsize(gz_file) / (1024**3)
            print(f"  {src_name}: {gz_file} ({size_gb:.2f} GB)")

        # --- Step 2: 解压 .jsonl.gz 并构建偏移索引 ---
        print()
        print("=" * 60)
        print("Step 2: 解压源文件并构建字节偏移索引")
        print("=" * 60)

        source_decompressed = {}  # src_name -> jsonl path
        source_offsets = {}       # src_name -> offset list
        source_line_counts = {}   # src_name -> num lines

        for src_name, gz_path in source_gz_paths.items():
            # 2a: 解压
            jsonl_path = decompress_to_single(gz_path, tmp_dir)
            source_decompressed[src_name] = jsonl_path

            # 2b: 构建偏移索引
            print(f"  [构建索引] {src_name} ...")
            offsets, num_lines = build_offset_index(jsonl_path)
            source_offsets[src_name] = offsets
            source_line_counts[src_name] = num_lines

            index_mb = len(offsets) * 8 / (1024**2)
            print(f"  [索引完成] {src_name}: {num_lines:,} 行, 索引大小 {index_mb:.1f} MB")

        total_lines = sum(source_line_counts.values())
        print(f"\n  总行数: {total_lines:,}")

        # --- Step 3: 计算训练计划 ---
        print()
        print("=" * 60)
        print("Step 3: 计算三阶段训练计划")
        print("=" * 60)

        plan = compute_phase_plan(args.total_steps, args.batch_size_per_step)
        print(f"  总步数: {plan['total_steps']:,}")
        print(f"  每步样本数: {plan['samples_per_step']}")
        print(f"  总样本数: {plan['total_samples']:,}")

        for phase_name, phase_info in plan['phases'].items():
            ratios_str = ', '.join(
                f"{k}={v:.0%}" for k, v in phase_info['ratios'].items()
            )
            print(f"  {phase_name} (step {phase_info['step_start']}-{phase_info['step_end']}, "
                  f"{phase_info['num_samples']:,} 样本):")
            print(f"    配比: {ratios_str}")

        # --- Step 4: 生成多阶段全局索引序列 ---
        print()
        print("=" * 60)
        print("Step 4: 生成多阶段全局索引序列")
        print("=" * 60)

        # 为每个数据源生成 0..N-1 的行索引列表
        source_indices = {}
        for src_name in DATA_SOURCE_CONFIG:
            n = source_line_counts[src_name]
            indices = list(range(n))
            rng.shuffle(indices)  # 预打散
            source_indices[src_name] = indices

        all_src_ids, all_line_ids = generate_global_index_sequence(
            source_indices, plan, rng
        )

        # 打印各阶段数据源分布验证
        print()
        print("  [验证] 各阶段数据源实际分布:")
        offset = 0
        for phase_name, phase_info in plan['phases'].items():
            n = phase_info['num_samples']
            phase_srcs = all_src_ids[offset:offset + n]
            dist = {}
            for s in phase_srcs:
                dist[s] = dist.get(s, 0) + 1
            dist_str = ', '.join(f"{k}={v/n:.1%}" for k, v in sorted(dist.items()))
            print(f"    {phase_name}: {dist_str}")
            offset += n

        # --- Step 5: 流式写出合并文件 ---
        print()
        print("=" * 60)
        print("Step 5: 流式写出合并 jsonl 文件")
        print("=" * 60)

        stream_write_merged(
            output_path=output_jsonl,
            source_paths=source_decompressed,
            source_offsets=source_offsets,
            all_src_ids=all_src_ids,
            all_line_ids=all_line_ids,
            log_interval=100000,
        )

        # --- Step 6: 写出元信息 ---
        print()
        print("=" * 60)
        print("Step 6: 写出元信息文件")
        print("=" * 60)

        write_meta_json(output_meta, plan, source_line_counts)

        # --- 完成 ---
        print()
        print("=" * 60)
        print("全部完成!")
        print("=" * 60)
        output_size_gb = os.path.getsize(output_jsonl) / (1024**3)
        print(f"  输出文件: {output_jsonl} ({output_size_gb:.2f} GB)")
        print(f"  元信息:   {output_meta}")
        print()
        print(f"  启动训练命令示例:")
        print(f"  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \\")
        print(f"  swift sft \\")
        print(f"    --model /path/to/pruned_deepseek_v4_flash_4b_a1.5b \\")
        print(f"    --model_type deepseek-v4 \\")
        print(f"    --train_type full \\")
        print(f"    --dataset {output_jsonl} \\")
        print(f"    --template_type completion \\")
        print(f"    --max_length 8192 \\")
        print(f"    --learning_rate 1.8e-4 \\")
        print(f"    --warmup_ratio 0.04 \\")
        print(f"    --lr_scheduler_type cosine \\")
        print(f"    --min_lr_ratio 0.04 \\")
        print(f"    --optimizer adamw_torch \\")
        print(f"    --weight_decay 0.1 \\")
        print(f"    --adam_beta2 0.95 \\")
        print(f"    --max_grad_norm 1.0 \\")
        print(f"    --batch_size 2 \\")
        print(f"    --gradient_accumulation_steps 128 \\")
        print(f"    --num_train_epochs 1 \\")
        print(f"    --deepspeed zero3 \\")
        print(f"    --bf16 true \\")
        print(f"    --gradient_checkpointing true \\")
        print(f"    --output_dir /path/to/output/recovery_cpt_v1 \\")
        print(f"    --save_steps 200 \\")
        print(f"    --save_total_limit 5 \\")
        print(f"    --logging_steps 10 \\")
        print(f"    --seed 42")

    finally:
        # 清理临时文件
        if not args.keep_temp and auto_cleanup:
            print(f"\n  清理临时目录: {tmp_dir}")
            tmp_obj.cleanup()
        elif args.keep_temp:
            print(f"\n  保留临时目录（--keep_temp）: {tmp_dir}")


if __name__ == '__main__':
    main()

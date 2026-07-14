#!/usr/bin/env python3
"""带 callback 的训练入口 - 基于 Megatron-Bridge run_recipe.py 改造

相比 run_recipe.py 新增：
  - 环境变量 MOE_MONITOR_ENABLED 开启后，自动注册 MoEMonitorCallback

使用方法：
  MOE_MONITOR_ENABLED=true uv run --no-sync python -m torch.distributed.run \\
      --nproc_per_node=8 run_recipe_with_monitor.py \\
      --recipe deepseek_v4_pruned_pretrain_8gpu_bf16_config \\
      --dataset llm-pretrain \\
      --step_func gpt_step \\
      train.train_iters=20000 ...
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

# 复用 Bridge 官方 run_recipe.py 的全部逻辑
# 路径基于 __file__ 相对推导（兼容宿主机 / 容器不同挂载点）
#   run_recipe_with_monitor.py 位于 <root>/examples/models/deepseek_v4/run_recipe_with_monitor.py
#   run_recipe.py             位于 <root>/scripts/training/run_recipe.py
#   (__file__ 在 <root> 下面 4 层，需要 3 次 .parent 回到 <root>)
_THIS_DIR = Path(__file__).resolve().parent
_BRIDGE_RUN_RECIPE = (
    _THIS_DIR.parent.parent.parent / "scripts" / "training" / "run_recipe.py"
)
if not _BRIDGE_RUN_RECIPE.is_file():
    raise FileNotFoundError(
        f"未找到 Bridge 官方 run_recipe.py: {_BRIDGE_RUN_RECIPE}\n"
        f"  期望相对路径: <root>/scripts/training/run_recipe.py\n"
        f"  当前 __file__: {Path(__file__).resolve()}"
    )
spec = importlib.util.spec_from_file_location("run_recipe", str(_BRIDGE_RUN_RECIPE))
run_recipe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_recipe)

from megatron.bridge.recipes.utils.dataset_utils import (
    apply_dataset_override,
    infer_mode_from_dataset,
)
from megatron.bridge.training.utils.omegaconf_utils import process_config_with_overrides

# MoE monitor callback（与本脚本同目录）
_THIS_DIR = Path(__file__).resolve().parent
_callback_spec = importlib.util.spec_from_file_location(
    "moe_monitor_callback", _THIS_DIR / "moe_monitor_callback.py"
)
_moe_monitor_mod = importlib.util.module_from_spec(_callback_spec)
_callback_spec.loader.exec_module(_moe_monitor_mod)
MoEMonitorCallback = _moe_monitor_mod.MoEMonitorCallback


def _build_callbacks():
    """根据环境变量构建 callback 列表"""
    callbacks = []

    if os.environ.get("MOE_MONITOR_ENABLED", "false").lower() in ("1", "true", "yes", "on"):
        top_k = int(os.environ.get("MOE_MONITOR_TOP_K", "6"))
        callbacks.append(MoEMonitorCallback(top_k=top_k))
        if int(os.environ.get("RANK", "0")) == 0:
            print(
                f"[run_recipe_with_monitor] 已注册 MoEMonitorCallback (top_k={top_k})"
            )

    return callbacks if callbacks else None


def main() -> None:
    """Run GPT training (pretrain or finetune) with optional callbacks."""
    args, cli_overrides = run_recipe.parse_args()

    config = run_recipe.load_recipe(args.recipe, args.peft_scheme)

    if args.dataset is not None:
        mode = infer_mode_from_dataset(args.dataset)
        config = apply_dataset_override(
            config,
            dataset_type=args.dataset,
            packed_sequence=args.packed_sequence,
            seq_length=args.seq_length,
            cli_overrides=cli_overrides,
        )
    else:
        mode = run_recipe.infer_train_mode(args.recipe)

    config = process_config_with_overrides(
        config,
        cli_overrides=cli_overrides or None,
    )

    # Ensure dataset.seq_length and model.seq_length stay in sync after CLI overrides
    if (
        hasattr(config, "model")
        and config.model is not None
        and hasattr(config, "dataset")
        and config.dataset is not None
    ):
        if (
            hasattr(config.dataset, "seq_length")
            and config.model.seq_length != config.dataset.seq_length
        ):
            config.model.seq_length = config.dataset.seq_length

    # TensorBoard 输出目录 (优先从 TENSORBOARD_DIR 环境变量读取)
    tb_dir = os.environ.get("TENSORBOARD_DIR")
    if tb_dir:
        os.makedirs(tb_dir, exist_ok=True)
        if getattr(config, "logger", None) is not None:
            config.logger.tensorboard_dir = tb_dir
            if int(os.environ.get("RANK", "0")) == 0:
                print(f"[run_recipe_with_monitor] TensorBoard dir: {tb_dir}")

    forward_step = run_recipe.load_forward_step(args.step_func, mode=mode)
    train_func = run_recipe.TRAIN_FUNCTIONS[mode]

    callbacks = _build_callbacks()

    train_func(
        config=config,
        forward_step_func=forward_step,
        callbacks=callbacks,
    )


if __name__ == "__main__":
    main()

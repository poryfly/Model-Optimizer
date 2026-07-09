#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# DeepSeek-V4 Pruned (4B-A1.5B) 预训练脚本 - 扁平化 CLI 风格
#
# 用法示例:
#   python pretrain_dpskv4_pruned.py \
#       --model /workdir/model_input/dpsk-v4-4B-A1.5B \
#       --dataset_dir /workdir/data/alpaca_train_text_document \
#       --output_dir /workdir/model_output \
#       --max_length 1024 \
#       --tensor_model_parallel_size 1 \
#       --pipeline_model_parallel_size 1 \
#       --expert_model_parallel_size 8 \
#       --micro_batch_size 1 \
#       --global_batch_size 8 \
#       --num_train_epochs 1 \
#       --train_iters 1000 \
#       --lr 3e-4 \
#       --save_steps 200 \
#       --logging_steps 1
#
# 简略形式 (使用 config.json 中的默认值):
#   python pretrain_dpskv4_pruned.py \
#       --model /workdir/model_input/dpsk-v4-4B-A1.5B \
#       --dataset_dir /workdir/data/alpaca_train_text_document \
#       --output_dir /workdir/model_output
#
# Mock 模式:
#   python pretrain_dpskv4_pruned.py \
#       --model /workdir/model_input/dpsk-v4-4B-A1.5B \
#       --output_dir /workdir/model_output \
#       --use_mock_data true

"""DeepSeek-V4 Pruned 预训练 - 扁平化 CLI 风格.

所有训练超参数都通过命令行参数传入, 配置文件 prep_config.py 仅做路径映射。
"""

import argparse
import os
import sys
import torch
from typing import Optional

# Megatron Bridge imports
from megatron.bridge import AutoBridge
from megatron.bridge.models.deepseek.deepseek_v4_bridge import (
    deepseek_v4_supports_blackwell_fused_kernels,
    set_deepseek_v4_pipeline_model_parallel_layout,
)
from megatron.bridge.training.config import (
    ConfigContainer,
    TrainingConfig,
    CheckpointConfig,
    LoggerConfig,
    TokenizerConfig,
    RNGConfig,
    DistributedInitConfig,
    DistributedDataParallelConfig,
    GPTDatasetConfig,
    ValidationConfig,
)
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.mixed_precision import bf16_mixed
from megatron.bridge.recipes.utils.optimizer_utils import (
    distributed_fused_adam_with_cosine_annealing,
)


def parse_args():
    """解析所有扁平化 CLI 参数."""
    p = argparse.ArgumentParser(
        description="DeepSeek-V4 Pruned Pretraining",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- 路径参数 ----
    path = p.add_argument_group("Paths")
    path.add_argument("--model", type=str, required=True,
                      help="HF 模型路径")
    path.add_argument("--dataset_dir", type=str, default=None,
                      help="Megatron binary 数据集前缀 (不含 .bin/.idx)")
    path.add_argument("--dataset_split", type=str, default="9999,8,2",
                      help="train/val/test 分割比例")
    path.add_argument("--output_dir", type=str, required=True,
                      help="checkpoint 和日志输出目录")
    path.add_argument("--use_mock_data", action="store_true", default=False,
                      help="使用 mock 数据验证流程")

    # ---- 并行度参数 ----
    par = p.add_argument_group("Parallelism")
    par.add_argument("--tensor_model_parallel_size", type=int, default=1)
    par.add_argument("--pipeline_model_parallel_size", type=int, default=1)
    par.add_argument("--pipeline_model_parallel_layout", type=str, default=None,
                     help="PP layout 字符串, 如 'Et*4|t*4|t*4|t*4L'")
    par.add_argument("--expert_model_parallel_size", type=int, default=8)
    par.add_argument("--context_parallel_size", type=int, default=1)
    par.add_argument("--num_nodes", type=int, default=1)

    # ---- 序列长度 ----
    p.add_argument("--max_length", type=int, default=1024,
                   help="序列长度 (seq_length)")

    # ---- Batch size / 训练步数 ----
    bs = p.add_argument_group("Batch size & Training")
    bs.add_argument("--micro_batch_size", type=int, default=1)
    bs.add_argument("--global_batch_size", type=int, default=8)
    bs.add_argument("--num_train_epochs", type=int, default=1)
    bs.add_argument("--train_iters", type=int, default=1000,
                   help="总训练迭代数 (优先级高于 num_train_epochs)")
    bs.add_argument("--finetune", action="store_true", default=False,
                   help="微调模式 (从 checkpoint 加载)")
    bs.add_argument("--padding_free", action="store_true", default=False,
                   help="启用 padding-free 训练")

    # ---- 学习率 / 优化器 ----
    opt = p.add_argument_group("Optimizer & LR")
    opt.add_argument("--lr", type=float, default=3e-4)
    opt.add_argument("--min_lr", type=float, default=3e-5)
    opt.add_argument("--lr_warmup_fraction", type=float, default=0.05)
    opt.add_argument("--lr_warmup_iters", type=int, default=None,
                    help="warmup 步数, 默认 lr_warmup_fraction * train_iters")
    opt.add_argument("--weight_decay", type=float, default=0.1)
    opt.add_argument("--clip_grad", type=float, default=1.0)
    opt.add_argument("--adam_beta1", type=float, default=0.9)
    opt.add_argument("--adam_beta2", type=float, default=0.95)

    # ---- Recompute ----
    rec = p.add_argument_group("Recompute")
    rec.add_argument("--recompute_granularity", type=str, default="full",
                     choices=["full", "selective", None])
    rec.add_argument("--recompute_method", type=str, default="uniform",
                     choices=["uniform", "block"])
    rec.add_argument("--recompute_num_layers", type=int, default=1)
    rec.add_argument("--recompute_modules", type=str, nargs="+", default=None,
                     help="selective recompute 的模块列表")

    # ---- MoE ----
    moe = p.add_argument_group("MoE")
    moe.add_argument("--moe_permute_fusion", action="store_true", default=False)
    moe.add_argument("--moe_grouped_gemm", action="store_true", default=False)
    moe.add_argument("--moe_shared_expert_overlap", action="store_true", default=False)
    moe.add_argument("--moe_aux_loss_coeff", type=float, default=0.0)
    moe.add_argument("--moe_z_loss_coeff", type=float, default=0.0)
    moe.add_argument("--moe_token_dispatcher_type", type=str, default="alltoall",
                     choices=["alltoall", "allgather"])

    # ---- Attention / Kernel ----
    attn = p.add_argument_group("Attention & Kernels")
    attn.add_argument("--attention_backend", type=str, default=None,
                      choices=[None, "flash", "fused", "local"])
    attn.add_argument("--apply_rope_fusion", action="store_true", default=False)
    attn.add_argument("--use_fused_mhc", action="store_true", default=False)
    attn.add_argument("--apply_dsa_kernel_fusion", action="store_true", default=False)
    attn.add_argument("--cross_entropy_loss_fusion", action="store_true", default=False)
    attn.add_argument("--mtp_num_layers", type=int, default=0)

    # ---- Checkpoint ----
    ckpt = p.add_argument_group("Checkpoint")
    ckpt.add_argument("--save_steps", type=int, default=200)
    ckpt.add_argument("--save_total_limit", type=int, default=10)
    ckpt.add_argument("--no_save_optim", action="store_true", default=False)
    ckpt.add_argument("--no_save_rng", action="store_true", default=False)
    ckpt.add_argument("--load_from_checkpoint", type=str, default=None,
                     help="从指定 checkpoint 恢复训练")
    ckpt.add_argument("--async_save", action="store_true", default=False)

    # ---- 验证 / 日志 ----
    misc = p.add_argument_group("Misc")
    misc.add_argument("--eval_interval", type=int, default=200)
    misc.add_argument("--eval_iters", type=int, default=5)
    misc.add_argument("--logging_steps", type=int, default=1)
    misc.add_argument("--dataloader_num_workers", type=int, default=4)
    misc.add_argument("--group_by_length", action="store_true", default=False)
    misc.add_argument("--truncation_strategy", type=str, default="split",
                      choices=["split", "discard"])
    misc.add_argument("--seed", type=int, default=1234)

    return p.parse_args()


def build_config(args) -> ConfigContainer:
    """根据扁平化参数构建 ConfigContainer."""

    # ---- Model: 加载 HF 模型并配置 ----
    print(f"[INFO] Loading model from {args.model}")
    model_cfg = AutoBridge.from_hf_pretrained(
        args.model, trust_remote_code=True
    ).to_megatron_provider(load_weights=False)

    # 并行度
    model_cfg.tensor_model_parallel_size = args.tensor_model_parallel_size
    model_cfg.pipeline_model_parallel_size = args.pipeline_model_parallel_size
    model_cfg.expert_model_parallel_size = args.expert_model_parallel_size
    model_cfg.context_parallel_size = args.context_parallel_size
    model_cfg.pipeline_dtype = torch.bfloat16
    model_cfg.virtual_pipeline_model_parallel_size = None
    model_cfg.expert_tensor_parallel_size = 1
    model_cfg.sequence_parallel = False
    model_cfg.seq_length = args.max_length
    model_cfg.params_dtype = torch.bfloat16

    # 必须先设置 mtp_num_layers, 再调用 set_deepseek_v4_pipeline_model_parallel_layout
    # 否则 layout 会用错误的 MTP 数量
    model_cfg.mtp_num_layers = args.mtp_num_layers if args.mtp_num_layers > 0 else None
    model_cfg.mtp_loss_scaling_factor = 0.1 if args.mtp_num_layers > 0 else 0.0

    model_cfg.account_for_embedding_in_pipeline_split = False
    model_cfg.account_for_loss_in_pipeline_split = False
    model_cfg.num_layers_in_first_pipeline_stage = None
    model_cfg.num_layers_in_last_pipeline_stage = None
    set_deepseek_v4_pipeline_model_parallel_layout(model_cfg)

    # MTP
    ratios = getattr(model_cfg, "csa_compress_ratios", None)
    num_layers = getattr(model_cfg, "num_layers", None)
    if ratios is not None and num_layers is not None and len(ratios) > num_layers:
        model_cfg.csa_compress_ratios = list(ratios)[:num_layers]

    # Kernel
    model_cfg.transformer_impl = "transformer_engine"
    model_cfg.attention_backend = args.attention_backend
    model_cfg.apply_dsa_kernel_fusion = args.apply_dsa_kernel_fusion
    model_cfg.apply_rope_fusion = args.apply_rope_fusion
    model_cfg.use_fused_mhc = args.use_fused_mhc and deepseek_v4_supports_blackwell_fused_kernels()
    model_cfg.dsa_indexer_loss_coeff = 0.0
    model_cfg.dsa_indexer_use_sparse_loss = False

    # MoE
    model_cfg.moe_token_dispatcher_type = args.moe_token_dispatcher_type
    model_cfg.moe_aux_loss_coeff = args.moe_aux_loss_coeff
    model_cfg.moe_router_force_load_balancing = False
    model_cfg.cross_entropy_loss_fusion = args.cross_entropy_loss_fusion
    model_cfg.cross_entropy_fusion_impl = "te"

    # Recompute
    model_cfg.recompute_granularity = args.recompute_granularity
    model_cfg.recompute_method = args.recompute_method
    model_cfg.recompute_num_layers = args.recompute_num_layers
    model_cfg.recompute_modules = args.recompute_modules
    model_cfg.fine_grained_activation_offloading = False
    model_cfg.offload_modules = None
    model_cfg.cuda_graph_impl = "none"
    model_cfg.cuda_graph_scope = "full"
    model_cfg.cuda_graph_warmup_steps = 3

    # ---- Tokenizer ----
    tokenizer_cfg = TokenizerConfig(
        tokenizer_type="HuggingFaceTokenizer",
        tokenizer_model=args.model,
        vocab_size=model_cfg.vocab_size,
        make_vocab_size_divisible_by=model_cfg.make_vocab_size_divisible_by,
        tensor_model_parallel_size=model_cfg.tensor_model_parallel_size,
        rank=0,
    )

    # ---- Dataset ----
    if args.use_mock_data:
        blend = None
        blend_per_split = None
    else:
        if args.dataset_dir is None:
            raise ValueError("--dataset_dir required when not using mock data")
        blend = [[args.dataset_dir], None]
        blend_per_split = None

    dataset_cfg = GPTDatasetConfig(
        random_seed=args.seed,
        reset_attention_mask=False,
        reset_position_ids=False,
        eod_mask_loss=False,
        seq_length=args.max_length,
        num_dataset_builder_threads=1,
        blend=blend,
        blend_per_split=blend_per_split,
        split=args.dataset_split,
        data_sharding=True,
        dataloader_type="single",
        num_workers=args.dataloader_num_workers,
        skip_getting_attention_mask_from_dataset=True,
    )

    # ---- Training ----
    train_cfg = TrainingConfig(
        train_iters=args.train_iters,
        global_batch_size=args.global_batch_size,
        micro_batch_size=args.micro_batch_size,
        manual_gc=True,
        manual_gc_interval=5,
        manual_gc_eval=5,
    )

    # ---- Optimizer / Scheduler ----
    warmup_iters = args.lr_warmup_iters or int(args.train_iters * args.lr_warmup_fraction)
    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=warmup_iters,
        lr_decay_iters=args.train_iters,
        max_lr=args.lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        clip_grad=args.clip_grad,
    )
    opt_cfg.adam_beta1 = args.adam_beta1
    opt_cfg.adam_beta2 = args.adam_beta2
    scheduler_cfg.lr_decay_style = "cosine"

    # ---- Validation ----
    val_cfg = ValidationConfig(
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
    )

    # ---- Logger ----
    tb_dir = os.path.join(args.output_dir, "tb_logs")
    log_cfg = LoggerConfig(
        log_interval=args.logging_steps,
        tensorboard_dir=tb_dir,
        log_timers_to_tensorboard=True,
    )

    # ---- Checkpoint ----
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_cfg = CheckpointConfig(
        save_interval=args.save_steps,
        save=ckpt_dir,
        load=args.load_from_checkpoint or ckpt_dir,
        ckpt_format="torch_dist",
        fully_parallel_save=True,
        async_save=args.async_save,
        save_optim=not args.no_save_optim,
        save_rng=not args.no_save_rng,
    )

    # ---- RNG / Distributed / DDP ----
    rng_cfg = RNGConfig(seed=args.seed)
    dist_cfg = DistributedInitConfig()
    ddp_cfg = DistributedDataParallelConfig(
        check_for_nan_in_grad=True,
        grad_reduce_in_fp32=True,
        overlap_grad_reduce=True,
        overlap_param_gather=True,
        average_in_collective=True,
        data_parallel_sharding_strategy="optim_grads_params",
        use_distributed_optimizer=True,
    )

    # ---- Comm overlap ----
    comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
    comm_overlap.delay_wgrad_compute = False
    comm_overlap.overlap_moe_expert_parallel_comm = False

    # ---- 组装 ConfigContainer ----
    cfg = ConfigContainer(
        model=model_cfg,
        tokenizer=tokenizer_cfg,
        dataset=dataset_cfg,
        train=train_cfg,
        optimizer=opt_cfg,
        scheduler=scheduler_cfg,
        validation=val_cfg,
        logger=log_cfg,
        checkpoint=ckpt_cfg,
        rng=rng_cfg,
        dist=dist_cfg,
        ddp=ddp_cfg,
        comm_overlap=comm_overlap,
        mixed_precision=bf16_mixed(),
    )

    return cfg


def main():
    args = parse_args()

    # 打印配置摘要
    print("=" * 60)
    print("DeepSeek-V4 Pruned Pretraining")
    print("=" * 60)
    print(f"  Model:              {args.model}")
    print(f"  Dataset:            {args.dataset_dir or '(mock)'}")
    print(f"  Output dir:         {args.output_dir}")
    print(f"  Max length:         {args.max_length}")
    print(f"  TP/PP/EP/CP:        {args.tensor_model_parallel_size}/{args.pipeline_model_parallel_size}/{args.expert_model_parallel_size}/{args.context_parallel_size}")
    print(f"  Micro batch:        {args.micro_batch_size}")
    print(f"  Global batch:       {args.global_batch_size}")
    print(f"  Train iters:        {args.train_iters}")
    print(f"  LR:                 {args.lr}")
    print(f"  Recompute:          {args.recompute_granularity}/{args.recompute_method}/{args.recompute_num_layers}")
    print("=" * 60)

    cfg = build_config(args)

    # 加载并运行训练
    from megatron.bridge.training.pretrain import pretrain
    from megatron.bridge.training.gpt_step import forward_step as gpt_forward_step

    forward_step = gpt_forward_step
    pretrain(config=cfg, forward_step_func=forward_step)


if __name__ == "__main__":
    main()

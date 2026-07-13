#!/usr/bin/env python3
"""将 Megatron checkpoint 导出为 HuggingFace 格式，供 SGLang/vLLM 部署.

支持两种 checkpoint 来源：
  1. Megatron-Bridge 训练保存（含 run_config.yaml）
  2. ms-swift / 原始 Megatron-LM 训练保存（含 common.pt，走 MLM 兼容路径）

对于第二种，checkpoint 的 args 可能缺少某些属性，
通过 monkey-patch 从 HF 模型 config 补全架构参数，避免 AttributeError。
"""
import argparse

import torch

from megatron.bridge import AutoBridge


# ---- HF config -> Megatron args 字段映射 ----
_HF_TO_MEGATRON = {
    "hidden_size": "hidden_size",
    "num_attention_heads": "num_attention_heads",
    "num_hidden_layers": "num_layers",
    "intermediate_size": "ffn_hidden_size",
    "max_position_embeddings": "max_position_embeddings",
    "vocab_size": "padded_vocab_size",
    "rms_norm_eps": "norm_epsilon",
    "rope_theta": "rotary_base",
    "num_key_value_heads": "num_query_groups",
}

# DeepSeek-V4 等使用 MLA 的 model_type
_MLA_MODEL_TYPES = {"deepseek_v4", "deepseek_v3"}

# 固定默认值（非架构参数）
_DEFAULTS = {
    "heterogeneous_layers_config_path": None,
    "no_persist_layer_norm": False,
    "params_dtype": torch.bfloat16,
    "overlap_p2p_comm": True,
    "rotary_interleaved": False,
    "decoder_first_pipeline_num_layers": None,
    "decoder_last_pipeline_num_layers": None,
    "fp8_param_gather": False,
    "swiglu": True,
    "bias_swiglu_fusion": False,
    "bias_gelu_fusion": False,
    "squared_relu": False,
    "init_method_xavier_uniform": False,
    "group_query_attention": False,
    "config_logger_dir": None,
    "cp_comm_type": [],
    "is_hybrid_model": False,
    "layernorm_zero_centered_gamma": False,
    "apply_layernorm_1p": False,
    "layernorm_epsilon": 1e-5,
    "norm_epsilon": 1e-5,
    "add_bias_linear": False,
    "gated_linear_unit": True,
    "attention_dropout": 0.0,
    "hidden_dropout": 0.0,
    "tensor_model_parallel_size": 1,
    "pipeline_model_parallel_size": 1,
    "expert_model_parallel_size": 1,
    "context_parallel_size": 1,
    "virtual_pipeline_model_parallel_size": None,
    "pipeline_model_parallel_layout": None,
    "num_layers_per_virtual_pipeline_stage": None,
    "pipeline_model_parallel_split_rank": None,
    "distributed_backend": "nccl",
    "ddp_backend": "nccl",
    "use_distributed_optimizer": True,
    "overlap_grad_reduce": False,
    "overlap_param_gather": False,
    "sequence_parallel": False,
    "tp_comm_overlap": False,
    "cross_entropy_loss_fusion": False,
    "apply_rope_fusion": False,
    "use_fused_mhc": False,
}


def _load_hf_config(hf_model_path, trust_remote_code=True):
    """加载 HF 模型 config，返回 dict。"""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(hf_model_path, trust_remote_code=trust_remote_code)
    return config.to_dict()


def _build_arch_defaults(hf_config_dict):
    """从 HF config 构建架构参数默认值。"""
    arch = {}

    # 标准字段映射
    for hf_key, mg_key in _HF_TO_MEGATRON.items():
        if hf_key in hf_config_dict:
            arch[mg_key] = hf_config_dict[hf_key]

    # MLA 检测
    model_type = hf_config_dict.get("model_type", "")
    arch["multi_latent_attention"] = model_type in _MLA_MODEL_TYPES

    # GQA 检测
    num_kv_heads = hf_config_dict.get("num_key_value_heads")
    num_attn_heads = hf_config_dict.get("num_attention_heads")
    if num_kv_heads is not None and num_attn_heads is not None:
        arch["group_query_attention"] = num_kv_heads != num_attn_heads
        arch["num_query_groups"] = num_kv_heads

    # vocab_size 对齐
    vocab_size = hf_config_dict.get("vocab_size")
    if vocab_size is not None:
        arch["padded_vocab_size"] = vocab_size
        arch["vocab_size"] = vocab_size

    # MoE 检测
    num_experts = hf_config_dict.get("n_routed_experts") or hf_config_dict.get("num_experts")
    if num_experts is not None:
        arch["num_experts"] = num_experts

    # norm_epsilon
    eps = hf_config_dict.get("rms_norm_eps")
    if eps is not None:
        arch["norm_epsilon"] = eps
        arch["layernorm_epsilon"] = eps

    # MoE 相关字段
    for hf_key, mg_key in [
        ("moe_intermediate_size", "moe_intermediate_size"),
        ("n_shared_experts", "moe_shared_expert_intermediate_size"),
        ("topk_method", "topk_method"),
        ("scoring_func", "scoring_func"),
        ("n_group", "moe_router_group"),
        ("topk_group", "moe_router_topk_group"),
        ("moe_layer_freq", "moe_layer_freq"),
        ("first_k_dense_replace", "moe_dense_layers"),
    ]:
        if hf_key in hf_config_dict:
            arch[mg_key] = hf_config_dict[hf_key]

    return arch


def _patch_args_defaults(hf_config_dict):
    """Patch _load_args_from_checkpoint to fill missing attributes from HF config."""
    from megatron.bridge.training.mlm_compat import arguments as mlm_args_mod

    arch_defaults = _build_arch_defaults(hf_config_dict)
    _original_load = mlm_args_mod._load_args_from_checkpoint

    def _patched_load(checkpoint_path):
        args = _original_load(checkpoint_path)
        # 先补架构参数（从 HF config 读取）
        for key, default in arch_defaults.items():
            if not hasattr(args, key) or getattr(args, key, None) is None:
                setattr(args, key, default)
        # 再补固定默认值
        for key, default in _DEFAULTS.items():
            if not hasattr(args, key):
                setattr(args, key, default)
        return args

    mlm_args_mod._load_args_from_checkpoint = _patched_load
    # 同时 patch model_load_save 模块中的引用
    try:
        from megatron.bridge.training import model_load_save
        if hasattr(model_load_save, "_load_args_from_checkpoint"):
            model_load_save._load_args_from_checkpoint = _patched_load
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Export Megatron checkpoint to HuggingFace format"
    )
    parser.add_argument(
        "--hf-model",
        required=True,
        help="原始 HuggingFace 模型路径或模型 ID（用于获取 config 和 tokenizer）",
    )
    parser.add_argument(
        "--megatron-path",
        required=True,
        help="Megatron checkpoint 路径，可以是 checkpoints/ 目录或 checkpoints/iter_xxxx 目录",
    )
    parser.add_argument(
        "--hf-path",
        required=True,
        help="导出的 HuggingFace 格式模型保存目录",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="允许加载自定义模型代码（DeepSeek-V4 需要）",
    )
    parser.add_argument(
        "--not-strict",
        action="store_true",
        help="允许源 checkpoint 和目标 HF 模型 key 不完全匹配",
    )
    args = parser.parse_args()

    # 加载 HF config，用于补全 checkpoint args 缺失的架构参数
    print(f"[INFO] Loading HF config from: {args.hf_model}")
    hf_config_dict = _load_hf_config(args.hf_model, trust_remote_code=args.trust_remote_code)

    # 启用 monkey-patch
    _patch_args_defaults(hf_config_dict)

    print(f"[INFO] Loading bridge from: {args.hf_model}")
    bridge = AutoBridge.from_hf_pretrained(
        args.hf_model, trust_remote_code=args.trust_remote_code
    )

    print(f"[INFO] Exporting {args.megatron_path} -> {args.hf_path}")
    bridge.export_ckpt(
        megatron_path=args.megatron_path,
        hf_path=args.hf_path,
        show_progress=True,
        strict=not args.not_strict,
        source_path=args.hf_model,
    )
    print(f"[INFO] Done. HF model saved to: {args.hf_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""全参训练 checkpoint 的 config 补全脚本。

Megatron export 输出的 checkpoint 会丢失若干 config 字段，导致加载时报错或
推理框架无法正确解析模型。本脚本从源模型目录补充这些缺失字段，并保证 config
一致。

用法:
    python fix_full_param_checkpoint.py \
        --src /data/wanghui/megatron_input/v4-pruned-L28-H1024-F512-E32-sglang \
        --dst /data/wanghui/megatron_output/.../checkpoint-XXXX

不传参数时使用默认路径（v4-pruned-L28-H1024-F512-E32-sglang）。
"""
import argparse
import json
import shutil
import struct
import glob
from pathlib import Path
from collections import Counter


# megatron export 会丢、但全参训练 checkpoint 必须保留的字段
FIELDS_TO_COPY = [
    "num_hash_layers",   # hash MoE 层数；不补会跳过 hash_routing 初始化
    "rope_scaling",      # RoPE 缩放参数（YaRN 等）
    "torch_dtype",       # 数据类型标记（bfloat16/float16 等）
    "compress_ratios",   # 各层压缩比，SGLang/vLLM 加载需要
    "compress_rates",    # 各压缩类型对应的压缩率
    "topk_method",       # 路由方法（noaux_tc 等）
    "scoring_func",      # 评分函数（sqrtsoftplus 等）
    "n_group",           # 路由分组数
    "topk_group",        # 每组选几个
]


def detect_weight_dtype(ckpt_dir: Path) -> tuple[str, str]:
    """检测 checkpoint 中 expert 权重的实际精度。

    返回:
        (main_dtype, description) 例如 ("bf16", "BF16 全精度")
    """
    files = sorted(glob.glob(str(ckpt_dir / "model-*.safetensors")))
    if not files:
        raise FileNotFoundError(f"No safetensors found in {ckpt_dir}")

    expert_weights = Counter()

    for fpath in files:
        with open(fpath, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))

        for name, meta in header.items():
            if name == "__metadata__":
                continue
            # 只看 expert weight，不看 scale
            if "expert" in name.lower() and ".weight" in name:
                dt = meta["dtype"]
                expert_weights[dt] += 1

    if not expert_weights:
        raise ValueError(f"No expert weights found in {ckpt_dir}")

    # 找主要类型
    main_dtype = max(expert_weights, key=expert_weights.get)
    main_count = expert_weights[main_dtype]
    total = sum(expert_weights.values())

    # 映射到标准格式
    dtype_map = {
        "BF16": ("bf16", f"BF16 全精度 ({main_count}/{total} 张量)"),
        "F8_E4M3": ("fp8", f"FP8 E4M3 ({main_count}/{total} 张量)"),
        "F8_E5M2": ("fp8", f"FP8 E5M2 ({main_count}/{total} 张量)"),
        "I8": ("fp4", f"FP4 packed I8 ({main_count}/{total} 张量)"),
    }

    expert_dtype, desc = dtype_map.get(main_dtype, (main_dtype.lower(), f"{main_dtype} ({main_count}/{total} 张量)"))

    return expert_dtype, desc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("/data2/.cache/models/deepseek-ai/v4-pruned-nas-final-sglang"),
        help="源模型目录（含完整 config.json）",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path("/data/wanghui/deepseekv4_train_mcore_bridge/model_output/phase0_recipe_callback/checkpoints/iter_0001000/hf"),
        help="checkpoint 目录（需要补全的 config.json 所在路径）",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="只打印概要信息（默认会打印所有字段的详细值）",
    )
    args = parser.parse_args()

    src_dir = args.src
    dst_dir = args.dst
    verbose = not args.quiet

    if not (src_dir / "config.json").exists():
        raise FileNotFoundError(f"src config.json not found: {src_dir / 'config.json'}")
    if not (dst_dir / "config.json").exists():
        raise FileNotFoundError(f"dst config.json not found: {dst_dir / 'config.json'}")

    src_config = json.loads((src_dir / "config.json").read_text())
    dst_config = json.loads((dst_dir / "config.json").read_text())

    # 1) 补全关键字段
    added = []
    for key in FIELDS_TO_COPY:
        if key in src_config and key not in dst_config:
            dst_config[key] = src_config[key]
            added.append(key)
            if verbose:
                print(f"  + Added {key!r}: {src_config[key]!r}")
    if added:
        print(f"Added missing config fields: {added}")
    else:
        print("No missing config fields to add")

    # 2) 检测实际权重精度
    print(f"\n检测权重精度中...")
    expert_dtype, dtype_desc = detect_weight_dtype(dst_dir)
    print(f"检测到: {dtype_desc}")

    # 3) 处理 quantization_config 和 expert_dtype
    if expert_dtype == "bf16":
        # BF16 权重：删除 quantization_config
        if "quantization_config" in dst_config:
            old_qc = dst_config["quantization_config"]
            del dst_config["quantization_config"]
            print("Removed quantization_config (weights are BF16)")
            if verbose:
                print(f"  - Deleted: {old_qc!r}")
        if "quantization_config" in src_config and "quantization_config" not in dst_config:
            print("Skipped copying quantization_config (incompatible with BF16 weights)")
    
        # 修正 expert_dtype 为 bf16
        old_expert_dtype = dst_config.get("expert_dtype")
        if old_expert_dtype != "bf16":
            dst_config["expert_dtype"] = "bf16"
            print(f"Fixed expert_dtype: {old_expert_dtype!r} -> 'bf16' (weights are BF16)")
    
    elif expert_dtype == "fp8":
        # FP8 权重：保留或设置 quantization_config
        if "quantization_config" not in dst_config:
            # 从源模型复制量化配置（如果有）
            if "quantization_config" in src_config:
                dst_config["quantization_config"] = src_config["quantization_config"]
                print("Copied quantization_config from source (weights are FP8)")
                if verbose:
                    print(f"  + Copied: {dst_config['quantization_config']!r}")
            else:
                # 默认 FP8 E4M3 blockwise 配置
                dst_config["quantization_config"] = {
                    "quant_method": "blockwise_fp8",
                    "format": "e4m3",
                }
                print("Added default FP8 quantization_config (weights are FP8 E4M3)")
                if verbose:
                    print(f"  + Added: {dst_config['quantization_config']!r}")
        else:
            print("Kept existing quantization_config (weights are FP8)")
            if verbose:
                print(f"  = Kept: {dst_config['quantization_config']!r}")
    
        # 修正 expert_dtype 为 fp8
        old_expert_dtype = dst_config.get("expert_dtype")
        if old_expert_dtype != "fp8":
            dst_config["expert_dtype"] = "fp8"
            print(f"Fixed expert_dtype: {old_expert_dtype!r} -> 'fp8' (weights are FP8)")
    
    elif expert_dtype == "fp4":
        # FP4 packed expert 权重 + FP8 attention 权重：
        # 必须保留 quantization_config，否则 SGLang 不会构建量化参数（weight_scale_inv 等）
        if "quantization_config" not in dst_config:
            if "quantization_config" in src_config:
                dst_config["quantization_config"] = src_config["quantization_config"]
                print("Copied quantization_config from source (weights are FP4 packed + FP8 attn)")
                if verbose:
                    print(f"  + Copied: {dst_config['quantization_config']!r}")
            else:
                print("Warning: No quantization_config in source, FP4 weights may not load correctly")
        else:
            print("Kept existing quantization_config (weights are FP4 packed + FP8 attn)")
            if verbose:
                print(f"  = Kept: {dst_config['quantization_config']!r}")
    
        old_expert_dtype = dst_config.get("expert_dtype")
        if old_expert_dtype != "fp4":
            dst_config["expert_dtype"] = "fp4"
            print(f"Fixed expert_dtype: {old_expert_dtype!r} -> 'fp4' (weights are FP4 packed)")
    
    else:
        print(f"⚠️ 未知精度: {expert_dtype}，保持 config 不变")

    # 4) 对齐 rope_parameters：强制使用源的 RoPE 参数，防止 export 过程中被改动
    if src_config.get("rope_parameters") != dst_config.get("rope_parameters"):
        old_rope = dst_config.get("rope_parameters")
        dst_config["rope_parameters"] = src_config["rope_parameters"]
        print("Aligned rope_parameters to source")
        if verbose:
            print(f"  - Old: {old_rope!r}")
            print(f"  + New: {src_config['rope_parameters']!r}")
    elif verbose:
        print(f"rope_parameters already aligned (no change)")

    # 5) 写回 config.json
    (dst_dir / "config.json").write_text(json.dumps(dst_config, indent=2))
    print(f"\nUpdated: {dst_dir / 'config.json'}")

    # 6) 复制 generation_config.json（推理参数：BOS/EOS、采样配置等）
    src_gen = src_dir / "generation_config.json"
    dst_gen = dst_dir / "generation_config.json"
    if src_gen.exists() and not dst_gen.exists():
        shutil.copy(src_gen, dst_gen)
        print(f"Copied: {dst_gen}")
    elif src_gen.exists() and dst_gen.exists():
        print(f"Skipped generation_config.json (already exists in dst)")
    else:
        print("No generation_config.json in src, skipped")

    # 7) 复制自定义 config/modeling 文件（如果缺失）
    for filename in ("configuration_deepseek_v4_pruned.py", "modeling_deepseek_v4_pruned.py"):
        src_file = src_dir / filename
        dst_file = dst_dir / filename
        if src_file.exists() and not dst_file.exists():
            shutil.copy(src_file, dst_file)
            print(f"Copied: {dst_file}")
        elif src_file.exists() and dst_file.exists():
            pass  # already exists
        else:
            print(f"Note: {filename} not in src, skipped")

    print("\nPost-processing completed.")


if __name__ == "__main__":
    main()

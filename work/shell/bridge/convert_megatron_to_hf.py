#!/usr/bin/env python3
"""将 Megatron-Bridge checkpoint 导出为 HuggingFace 格式，供 SGLang/vLLM 部署."""
import argparse

from megatron.bridge import AutoBridge


def main():
    parser = argparse.ArgumentParser(
        description="Export Megatron-Bridge checkpoint to HuggingFace format"
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

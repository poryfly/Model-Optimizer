#!/bin/bash
# 将 Megatron-Bridge checkpoint 导出为 HuggingFace 格式
set -e

# 原始 HF 模型路径（需要包含 config.json 和自定义 modeling 文件）
HF_MODEL="/workdir/model_input/dpsk-v4-4B-A1.5B"

# Megatron checkpoint 路径
MEGATRON_CKPT="/workdir/model_output_resume/checkpoints/iter_0000002"

# 导出目标路径
HF_OUTPUT="/workdir/model_output_resume/hf_export/dpsk-v4-4B-A1.5B-hf"

python /workdir/convert_megatron_to_hf.py \
  --hf-model "${HF_MODEL}" \
  --megatron-path "${MEGATRON_CKPT}" \
  --hf-path "${HF_OUTPUT}" \
  --trust-remote-code

echo "导出完成: ${HF_OUTPUT}"

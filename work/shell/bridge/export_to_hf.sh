#!/bin/bash
# 将 Megatron-Bridge checkpoint 导出为 HuggingFace 格式
set -e

# 原始 HF 模型路径（需要包含 config.json 和自定义 modeling 文件）
HF_MODEL="/workdir/model_input/dpsk-v4-4B-A1.5B"
 #dpsk-v4-4B-A1.5B    checkpoint-10000
# Megatron checkpoint 路径
MEGATRON_CKPT="/workdir/model_output/phase2_recipe/checkpoints/iter_0020000"
# 导出目标路径
HF_OUTPUT="/workdir/model_output/hf_export/dpsk-v4-4B-A1.5B-hf-phase1-10000"

cd /workdir/Megatron-Bridge

nohup torchrun --nproc_per_node 8 \
    examples/conversion/convert_checkpoints_multi_gpu.py export \
    --hf-model "${HF_MODEL}" \
    --megatron-path "${MEGATRON_CKPT}" \
    --tp 1 --pp 1 --ep 8 \
    --torch-dtype bfloat16 \
    --hf-path "${HF_OUTPUT}" \
    --distributed-save \
    --trust-remote-code \
    > "${HF_OUTPUT%/}_convert.log" 2>&1 &
echo "PID: $!"

echo "导出完成: ${HF_OUTPUT}"

    # --not-strict \
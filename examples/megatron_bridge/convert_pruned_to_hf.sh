#!/bin/bash
# DeepSeek-V4 NAS Pruning Workflow
# Automatically searches for optimal pruning configuration and applies it

set -e

# Configuration
MODEL_PATH="/data/.cache/models/deepseek-ai/DeepSeek-V4-Flash"
OUTPUT_BASE="/data/output/v4-pruned-nas"

MEGATRON_CKPT_DIR="${OUTPUT_BASE}-final_megatron"
MEGATRON_CKPT="${MEGATRON_CKPT_DIR}"
HF_CONFIG="${OUTPUT_BASE}-final/config.json"
OUTPUT_DIR="${OUTPUT_BASE}-final-sglang"

PYTHONPATH=/opt/Model-Optimizer:$PYTHONPATH \
/opt/venv/bin/python convert_pruned_to_hf.py \
    --megatron_ckpt "$MEGATRON_CKPT" \
    --hf_config "$HF_CONFIG" \
    --output_dir "$OUTPUT_DIR"

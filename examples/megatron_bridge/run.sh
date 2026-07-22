#!/bin/bash
# DeepSeek-V4 importance-based pruning (mcore_minitron, 8x GPU, PP=8)
#
# All pruning parameters are optional env vars. Unset = keep original value.
#
# Usage:
#   bash run.sh                                              # no pruning
#   NUM_LAYERS=39 HIDDEN_SIZE=3584 bash run.sh               # ~20% cut
#   NUM_LAYERS=10 HIDDEN_SIZE=1408 MOE_FFN=512 bash run.sh   # ~4B
#
# Constraints (Marlin MXFP4 MoE kernel):
#   MOE_FFN >= 512  (intermediate_size_per_partition = MOE_FFN/8 >= 64)
#   MOE_FFN % 128 == 0
#   HIDDEN_SIZE % 128 == 0
#
# Environment variables:
#   NUM_LAYERS          - Target number of layers (depth pruning)
#   HIDDEN_SIZE         - Target hidden_size (width pruning)
#   MOE_FFN             - Target moe_ffn_hidden_size (expert FFN width, min 512)
#   NUM_EXPERTS         - Target num_moe_experts (expert count pruning)
#   HEAD_DIM            - Target MLA per-head dimension (post-prune slicing;
#                         DeepSeek-V4-Flash ships with 512; try 128 or 256
#                         for 4B-scale hidden_size). Unset = keep original.
#   CALIB_SAMPLES       - Calibration samples (default: 128)
#   SEQ_LENGTH          - Calibration seq length (default: 2048)
#   OUTPUT_BASE         - Output directory (default: auto-generated)
set -e

export PYTHONPATH=/opt/Model-Optimizer:$PYTHONPATH
export FLASHINFER_DISABLE_VERSION_CHECK=1

export HF_ENDPOINT=https://hf-mirror.com
#hf auth login --token your token

# ── Defaults ───────────────────────────────────────────────────────────
CALIB_SAMPLES="${CALIB_SAMPLES:-128}"
SEQ_LENGTH="${SEQ_LENGTH:-2048}"

# ── Validate constraints ──────────────────────────────────────────────
if [ -n "$MOE_FFN" ]; then
    if [ "$MOE_FFN" -lt 512 ]; then
        echo "ERROR: MOE_FFN=$MOE_FFN is too small. Marlin MXFP4 requires MOE_FFN >= 512"
        echo "  (intermediate_size_per_partition = MOE_FFN / TP=8 must be >= 64)"
        exit 1
    fi
    if [ $((MOE_FFN % 128)) -ne 0 ]; then
        echo "ERROR: MOE_FFN=$MOE_FFN must be divisible by 128"
        exit 1
    fi
fi
if [ -n "$HIDDEN_SIZE" ] && [ $((HIDDEN_SIZE % 128)) -ne 0 ]; then
    echo "ERROR: HIDDEN_SIZE=$HIDDEN_SIZE must be divisible by 128"
    exit 1
fi

# Auto-generate output path from pruning config
if [ -z "$OUTPUT_BASE" ]; then
    _tag="v4-pruned"
    [ -n "$NUM_LAYERS" ] && _tag="${_tag}-L${NUM_LAYERS}"
    [ -n "$HIDDEN_SIZE" ] && _tag="${_tag}-H${HIDDEN_SIZE}"
    [ -n "$MOE_FFN" ] && _tag="${_tag}-F${MOE_FFN}"
    [ -n "$NUM_EXPERTS" ] && _tag="${_tag}-E${NUM_EXPERTS}"
    [ -n "$HEAD_DIM" ] && _tag="${_tag}-hd${HEAD_DIM}"
    OUTPUT_BASE="/data/output/${_tag}"
fi

# ── Build prune args ──────────────────────────────────────────────────
PRUNE_ARGS=(
    --hf_model_name_or_path /data/.cache/models/deepseek-ai/DeepSeek-V4-Flash
    --output_hf_path "$OUTPUT_BASE"
    --pp_size 8
    --trust_remote_code
    --calibration_samples "$CALIB_SAMPLES"
    --seq_length "$SEQ_LENGTH"
)

[ -n "$NUM_LAYERS" ]    && PRUNE_ARGS+=(--num_layers "$NUM_LAYERS")
[ -n "$HIDDEN_SIZE" ]   && PRUNE_ARGS+=(--hidden_size "$HIDDEN_SIZE")
[ -n "$MOE_FFN" ]       && PRUNE_ARGS+=(--moe_ffn_hidden_size "$MOE_FFN")
[ -n "$NUM_EXPERTS" ]   && PRUNE_ARGS+=(--num_moe_experts "$NUM_EXPERTS")
[ -n "$HEAD_DIM" ]      && PRUNE_ARGS+=(--head_dim "$HEAD_DIM")

# ── Run ────────────────────────────────────────────────────────────────
echo "=== Pruning config ==="
echo "  Output: $OUTPUT_BASE"
echo "  Layers: ${NUM_LAYERS:-43 (original)}"
echo "  Hidden: ${HIDDEN_SIZE:-4096 (original)}"
echo "  MoE FFN: ${MOE_FFN:-2048 (original)}"
echo "  Experts: ${NUM_EXPERTS:-256 (original)}"
echo "  Head dim: ${HEAD_DIM:-512 (original)}"
echo "  Calibration: ${CALIB_SAMPLES} samples, seq_len=${SEQ_LENGTH}"
echo ""

/opt/venv/bin/python3 -m torch.distributed.run --nproc_per_node=8 prune_deepseek_v4.py \
    "${PRUNE_ARGS[@]}"

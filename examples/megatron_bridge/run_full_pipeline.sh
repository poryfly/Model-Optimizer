#!/bin/bash
# DeepSeek-V4 pruning pipeline: prune → merge → convert → verify
#
# All pruning parameters are passed as env vars to run.sh.
#
# Usage:
#   bash run_full_pipeline.sh                                           # no pruning
#   NUM_LAYERS=39 HIDDEN_SIZE=3584 bash run_full_pipeline.sh           # ~20% cut
#   NUM_LAYERS=10 HIDDEN_SIZE=1408 MOE_FFN=256 bash run_full_pipeline.sh  # ~4B
#   NUM_LAYERS=10 HIDDEN_SIZE=1408 MOE_FFN=256 NUM_EXPERTS=256 bash run_full_pipeline.sh  # ~4B with expert pruning
set -e

export HF_ENDPOINT=https://hf-mirror.com
#hf auth login --token your token

export PYTHONPATH=/opt/Model-Optimizer:$PYTHONPATH
export FLASHINFER_DISABLE_VERSION_CHECK=1
cd /opt/Model-Optimizer/examples/megatron_bridge

# ── Compute output paths (must match run.sh logic) ────────────────────
if [ -z "$OUTPUT_BASE" ]; then
    _tag="v4-pruned"
    [ -n "$NUM_LAYERS" ] && _tag="${_tag}-L${NUM_LAYERS}"
    [ -n "$HIDDEN_SIZE" ] && _tag="${_tag}-H${HIDDEN_SIZE}"
    [ -n "$MOE_FFN" ] && _tag="${_tag}-F${MOE_FFN}"
    [ -n "$NUM_EXPERTS" ] && _tag="${_tag}-E${NUM_EXPERTS}"
    export OUTPUT_BASE="/data/output/${_tag}"
fi
export NUM_LAYERS HIDDEN_SIZE MOE_FFN NUM_EXPERTS
MEGATRON_DIR="${OUTPUT_BASE}_megatron"
HF_OUTPUT="${OUTPUT_BASE}-sglang"

# ── Clean old outputs ─────────────────────────────────────────────────
if [ -d "$OUTPUT_BASE" ] || [ -d "$MEGATRON_DIR" ] || [ -d "$HF_OUTPUT" ]; then
    echo "=== Cleaning old outputs ==="
    echo "  $OUTPUT_BASE"
    echo "  $MEGATRON_DIR"
    echo "  $HF_OUTPUT"
    rm -rf "$OUTPUT_BASE" "$MEGATRON_DIR" "$HF_OUTPUT"
fi

# ── Step 1: Pruning (mcore_minitron + importance-based selection) ─────
echo "=== Step 1: Pruning ==="
bash run.sh

# ── Step 2: Merge per-rank state_dicts ────────────────────────────────
echo ""
echo "=== Step 2: Merge per-rank state_dicts ==="
/opt/venv/bin/python3 merge_pruned_state_dicts.py \
    --megatron_dir "$MEGATRON_DIR" \
    --num_ranks 8 \
    --cleanup

# Safety check
MERGED_SIZE=$(stat -c%s "$MEGATRON_DIR/pruned_model.pt" 2>/dev/null || echo 0)
if [ "$MERGED_SIZE" -lt 1000000 ]; then
    echo "ERROR: Merged checkpoint is too small (${MERGED_SIZE} bytes)."
    echo "  The prune step may have been skipped."
    exit 1
fi

# ── Step 3: Convert to SGLang-compatible safetensors ──────────────────
echo ""
echo "=== Step 3: Convert to HF safetensors ==="
/opt/venv/bin/python3 convert_pruned_to_hf.py \
    --megatron_ckpt "$MEGATRON_DIR/pruned_model.pt" \
    --hf_config "$OUTPUT_BASE/config.json" \
    --output_dir "$HF_OUTPUT"

# ── Verify ────────────────────────────────────────────────────────────
echo ""
echo "=== Verify output ==="
if [ -f "$HF_OUTPUT/config.json" ]; then
    /opt/venv/bin/python3 -c "
import json
with open('$HF_OUTPUT/config.json') as f:
    cfg = json.load(f)
print(f'  hidden_size={cfg.get(\"hidden_size\")}')
print(f'  num_hidden_layers={cfg.get(\"num_hidden_layers\")}')
print(f'  n_routed_experts={cfg.get(\"n_routed_experts\")}')
print(f'  moe_intermediate_size={cfg.get(\"moe_intermediate_size\")}')
print(f'  compress_ratios={len(cfg.get(\"compress_ratios\", []))} entries')
"
else
    echo "  WARNING: $HF_OUTPUT/config.json not found!"
fi

echo ""
echo "=== Pipeline complete ==="
echo "HF output: $HF_OUTPUT"

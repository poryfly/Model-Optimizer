#!/bin/bash
# DeepSeek-V4 NAS Pruning Workflow
# Automatically searches for optimal pruning configuration and applies it

set -e

# Configuration
MODEL_PATH="/data/.cache/models/deepseek-ai/DeepSeek-V4-Flash"
OUTPUT_BASE="/data/output/v4-pruned-nas"
PP_SIZE=8
TARGET_RATIO=0.014  # Keep 80% of parameters
NUM_GPUS=8

# ─────────────────────────────────────────────────────────────────────
# GPU lifecycle helpers
# ─────────────────────────────────────────────────────────────────────
# Previous interrupted runs (especially of prune_deepseek_v4_nas.py with
# the slow forward-validation path, or the subsequent prune_deepseek_v4.py
# step) can leave 8× python child processes holding 60-80GB per GPU.
# If we don't kill them, the *next* `torch.distributed.run` invocation
# will OOM. These helpers prevent that footgun.

GPU_MEM_TOTAL_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0")
GPU_MEM_FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0")
NUM_GPUS_DETECTED=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)

list_gpu_python_pids() {
    # All python processes that currently hold non-trivial GPU memory.
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null \
        | awk -F',' '$2+0 > 256 { gsub(/ /,"",$1); print $1 }' || true
}

kill_gpu_python() {
    local pids
    pids=$(list_gpu_python_pids)
    if [ -z "$pids" ]; then
        return 0
    fi
    echo "⚠️  Pre-flight: killing residual GPU python processes: $pids"
    # Kill parent torch.distributed.run first, then the workers.
    for pid in $pids; do
        # If this pid is a torch.distributed.run parent, kill its children too.
        local children
        children=$(pgrep -P "$pid" 2>/dev/null | tr '\n' ' ' || true)
        kill -9 $pid $children 2>/dev/null || true
    done
    sleep 2
    # Anything still alive gets another SIGKILL.
    pids=$(list_gpu_python_pids)
    if [ -n "$pids" ]; then
        echo "    still alive after first pass: $pids — SIGKILL again"
        kill -9 $pids 2>/dev/null || true
        sleep 1
    fi
}

preflight_gpu() {
    echo "=== GPU preflight ==="
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "  nvidia-smi not found; skipping GPU check."
        return 0
    fi
    echo "  Detected $NUM_GPUS_DETECTED GPU(s); ${GPU_MEM_FREE_MIB} MiB free on GPU 0 / ${GPU_MEM_TOTAL_MIB} MiB total."

    local occupied
    occupied=$(list_gpu_python_pids)
    if [ -n "$occupied" ]; then
        echo "  Found residual python processes holding GPU memory:"
        nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null
        # Auto-kill: this script is intended to be re-runnable. If the user
        # wants to *preserve* a running process, they should not invoke this.
        kill_gpu_python
    fi

    # Verify GPUs are actually empty (allow 256 MiB tolerance per GPU).
    local still_busy
    still_busy=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null \
        | awk -F',' '$2+0 > 256' || true)
    if [ -n "$still_busy" ]; then
        echo "❌ GPU still has resident processes after cleanup:"
        echo "$still_busy"
        echo "   Aborting to avoid OOM. Please inspect with 'nvidia-smi' and kill manually if needed."
        exit 1
    fi
    echo "  ✓ All GPUs clean."
}

wait_for_gpu_free() {
    local timeout_s="${1:-60}"
    local i=0
    while [ "$i" -lt "$timeout_s" ]; do
        local busy
        busy=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null \
            | awk -F',' '$2+0 > 256' || true)
        if [ -z "$busy" ]; then
            return 0
        fi
        sleep 1
        i=$((i+1))
    done
    echo "  ⚠️  GPU still busy after ${timeout_s}s"
    return 1
}

# Run cleanup_gpu on any abnormal exit so we never leave zombies.
cleanup_on_exit() {
    local exit_code=$?
    echo ""
    echo "=== Cleanup (exit code $exit_code) ==="
    wait_for_gpu_free 30 || kill_gpu_python
    echo "  ✓ Final GPU state:"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv 2>/dev/null | head -"$NUM_GPUS_DETECTED"
}
trap cleanup_on_exit EXIT INT TERM

# ─────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────

preflight_gpu

# ─────────────────────────────────────────────────────────────────────
# Clean intermediate artifacts from previous runs
# ─────────────────────────────────────────────────────────────────────
# `prune_deepseek_v4_nas.py` skips when `${OUTPUT_BASE}-best/config.json`
# already exists, and the wrapper then re-uses the cached
# `best_config.json` even if you changed TARGET_RATIO / NUM_LAYERS etc.
# That produces a pruned model whose config doesn't match what you
# actually asked for. Wipe the per-stage outputs every time so each run
# is reproducible.
cleanup_previous_artifacts() {
    local paths=(
        "${OUTPUT_BASE}-best"
        "${OUTPUT_BASE}-final"
        "${OUTPUT_BASE}-final_megatron"
        "${OUTPUT_BASE}-final-sglang"
        "${OUTPUT_BASE}-best_best_config.json"
        "${OUTPUT_BASE}-best-pruned"
        "${OUTPUT_BASE}-pruned"
    )
    local removed=()
    for p in "${paths[@]}"; do
        if [ -e "$p" ]; then
            echo "  Removing previous artifact: $p"
            rm -rf "$p"
            removed+=("$p")
        fi
    done
    if [ "${#removed[@]}" -eq 0 ]; then
        echo "  (no previous artifacts to remove under $OUTPUT_BASE)"
    fi
}
echo "=== Cleaning previous run artifacts ==="
cleanup_previous_artifacts
echo ""

# Step 1: Run NAS search
echo "========================================================================"
echo "STEP 1: NAS Search (finding optimal configuration)"
echo "========================================================================"
echo ""

FLASHINFER_DISABLE_VERSION_CHECK=1 \
PYTHONPATH=/opt/Model-Optimizer:$PYTHONPATH \
/opt/venv/bin/python3 -m torch.distributed.run --nproc_per_node=$NUM_GPUS prune_deepseek_v4_nas.py \
    --hf_model_name_or_path "$MODEL_PATH" \
    --output_hf_path "${OUTPUT_BASE}-best" \
    --pp_size $PP_SIZE \
    --target_params_ratio $TARGET_RATIO \
    --num_candidates 10 \
    --calibration_samples 1024 \
    --seq_length 2048 \
    --validation_ratio 0.2 \
    --trust_remote_code

# Wait for all child python processes to actually release the GPUs.
echo ""
echo "=== Waiting for Step 1 processes to release GPUs ==="
wait_for_gpu_free 60 || {
    echo "  Step 1 left GPU busy — killing stragglers"
    kill_gpu_python
    wait_for_gpu_free 30
}

# Check if best config was saved
BEST_CONFIG="${OUTPUT_BASE}-best_best_config.json"
if [ ! -f "$BEST_CONFIG" ]; then
    echo "ERROR: Best configuration not found at $BEST_CONFIG"
    exit 1
fi

echo ""
echo "========================================================================"
echo "NAS Search Complete!"
echo "========================================================================"
echo ""
echo "Best configuration:"
cat "$BEST_CONFIG"
echo ""

# Extract best configuration
HIDDEN_SIZE=$(jq -r '.hidden_size' "$BEST_CONFIG")
NUM_LAYERS=$(jq -r '.num_layers' "$BEST_CONFIG")
NUM_EXPERTS=$(jq -r '.num_moe_experts' "$BEST_CONFIG")
FFN_SIZE=$(jq -r '.moe_ffn_hidden_size' "$BEST_CONFIG")
SHARED_FFN=$(jq -r '.moe_shared_expert_intermediate_size' "$BEST_CONFIG")
PARAMS_RATIO=$(jq -r '.params_ratio' "$BEST_CONFIG")

echo "Applying best configuration:"
echo "  hidden_size: $HIDDEN_SIZE"
echo "  num_layers: $NUM_LAYERS"
echo "  num_moe_experts: $NUM_EXPERTS"
echo "  moe_ffn_hidden_size: $FFN_SIZE"
echo "  moe_shared_expert_intermediate_size: $SHARED_FFN"
echo "  params_ratio: $PARAMS_RATIO"
echo ""

# Step 2: Apply best configuration with full importance estimation
echo "========================================================================"
echo "STEP 2: Applying best configuration with activation-based importance"
echo "========================================================================"
echo ""

# Re-run preflight to be safe (an interrupted Step 1 could leak processes).
preflight_gpu

FLASHINFER_DISABLE_VERSION_CHECK=1 \
PYTHONPATH=/opt/Model-Optimizer:$PYTHONPATH \
/opt/venv/bin/python3 -m torch.distributed.run --nproc_per_node=$NUM_GPUS prune_deepseek_v4.py \
    --hf_model_name_or_path "$MODEL_PATH" \
    --output_hf_path "${OUTPUT_BASE}-final" \
    --pp_size $PP_SIZE \
    --hidden_size $HIDDEN_SIZE \
    --num_layers $NUM_LAYERS \
    --num_moe_experts $NUM_EXPERTS \
    --moe_ffn_hidden_size $FFN_SIZE \
    --moe_shared_expert_intermediate_size $SHARED_FFN \
    --calibration_samples 1024 \
    --seq_length 2048 \
    --trust_remote_code

echo ""
echo "=== Waiting for Step 2 processes to release GPUs ==="
wait_for_gpu_free 60 || {
    echo "  Step 2 left GPU busy — killing stragglers"
    kill_gpu_python
    wait_for_gpu_free 30
}

echo ""
echo "========================================================================"
echo "Pruning Complete!"
echo "========================================================================"
echo ""
echo "Pruned model saved to: ${OUTPUT_BASE}-final"
echo ""

# Step 3: Convert to HF format
echo "========================================================================"
echo "STEP 3: Converting to HF format"
echo "========================================================================"
echo ""

# prune_deepseek_v4.py with PP=8 writes one Megatron state_dict per PP rank
# (pruned_model_rank{0..7}.pt). convert_pruned_to_hf.py accepts the parent
# directory and merges them; passing the .pt file directly would FileNotFoundError.
MEGATRON_CKPT_DIR="${OUTPUT_BASE}-final_megatron"
MEGATRON_CKPT="${MEGATRON_CKPT_DIR}"
HF_CONFIG="${OUTPUT_BASE}-final/config.json"
OUTPUT_DIR="${OUTPUT_BASE}-final-sglang"

if [ ! -d "$MEGATRON_CKPT_DIR" ]; then
    echo "ERROR: Megatron checkpoint directory not found: $MEGATRON_CKPT_DIR"
    echo "       Step 2 (prune_deepseek_v4.py) likely failed to save."
    exit 1
fi
ls -la "$MEGATRON_CKPT_DIR"

PYTHONPATH=/opt/Model-Optimizer:$PYTHONPATH \
python3 convert_pruned_to_hf.py \
    --megatron_ckpt "$MEGATRON_CKPT" \
    --hf_config "$HF_CONFIG" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "========================================================================"
echo "Workflow Complete!"
echo "========================================================================"
echo ""
echo "Final pruned model (HF format): $OUTPUT_DIR"
echo ""
echo "To serve with SGLang:"
echo "  bash /data/dpsk-v4-run.sh  # Update model path in script first"
echo ""
echo "Best configuration saved to: $BEST_CONFIG"
echo ""

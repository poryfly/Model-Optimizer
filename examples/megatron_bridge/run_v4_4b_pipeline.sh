#!/bin/bash
# DeepSeek-V4-Flash -> DeepSeek-V4-4B-A1B 端到端裁剪与后训练流水线
#
# 该脚本整合：
#   1. 精确 4B 参数目标配置生成 (prune_deepseek_v4_4b_config.py)
#   2. 基于激活重要性的结构化裁剪 (prune_deepseek_v4.py)
#   3. Megatron -> HF 格式转换 (convert_pruned_to_hf.py)
#   4. 输出可直接用于后续 KD / CPT / SFT 的 4B 小模型
#
# 注意：本脚本只负责生成裁剪后的小模型。后续 KD/CPT/SFT/量化阶段需要额外
# 数据与算力，请结合 distill_deepseek_v4_4b.py / sft_deepseek_v4_4b.py 使用。
#
# Usage:
#   bash run_v4_4b_pipeline.sh

set -e

# ─────────────────────────────────────────────────────────────────────
# 用户可配置参数
# ─────────────────────────────────────────────────────────────────────
MODEL_PATH="/data/.cache/models/deepseek-ai/DeepSeek-V4-Flash"
OUTPUT_BASE="/data/output/v4-4b-a1b"
PP_SIZE=8
NUM_GPUS=8

# 目标总参数量 (默认 4B)。如需 3.5B 可改为 3500000000。
TARGET_TOTAL_PARAMS=4000000000

# 校准样本数 / 序列长度
CALIBRATION_SAMPLES=1024
SEQ_LENGTH=2048

# ─────────────────────────────────────────────────────────────────────
# GPU lifecycle helpers (与 run_nas_pruning.sh 保持一致)
# ─────────────────────────────────────────────────────────────────────
GPU_MEM_TOTAL_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0")
GPU_MEM_FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0")
NUM_GPUS_DETECTED=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)

list_gpu_python_pids() {
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null \
        | awk -F',' '$2+0 > 256 { gsub(/ /,"",$1); print $1 }' || true
}

kill_gpu_python() {
    local pids
    pids=$(list_gpu_python_pids)
    if [ -z "$pids" ]; then
        return 0
    fi
    echo "Pre-flight: killing residual GPU python processes: $pids"
    for pid in $pids; do
        local children
        children=$(pgrep -P "$pid" 2>/dev/null | tr '\n' ' ' || true)
        kill -9 $pid $children 2>/dev/null || true
    done
    sleep 2
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
        kill_gpu_python
    fi

    local still_busy
    still_busy=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null \
        | awk -F',' '$2+0 > 256' || true)
    if [ -n "$still_busy" ]; then
        echo "GPU still has resident processes after cleanup:"
        echo "$still_busy"
        echo "   Aborting to avoid OOM."
        exit 1
    fi
    echo "  All GPUs clean."
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
    echo "  GPU still busy after ${timeout_s}s"
    return 1
}

cleanup_on_exit() {
    local exit_code=$?
    echo ""
    echo "=== Cleanup (exit code $exit_code) ==="
    wait_for_gpu_free 30 || kill_gpu_python
    echo "  Final GPU state:"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv 2>/dev/null | head -"$NUM_GPUS_DETECTED"
}
trap cleanup_on_exit EXIT INT TERM

# ─────────────────────────────────────────────────────────────────────
# 清理历史产物
# ─────────────────────────────────────────────────────────────────────
cleanup_previous_artifacts() {
    local paths=(
        "${OUTPUT_BASE}-best"
        "${OUTPUT_BASE}-final"
        "${OUTPUT_BASE}-final_megatron"
        "${OUTPUT_BASE}-final-sglang"
        "${OUTPUT_BASE}_best_config.json"
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

preflight_gpu

# ─────────────────────────────────────────────────────────────────────
# Step 1: 精确 4B 配置生成
# ─────────────────────────────────────────────────────────────────────
echo "========================================================================"
echo "STEP 1: Generate exact 4B-target pruning configuration"
echo "========================================================================"
echo ""

PYTHONPATH=/opt/Model-Optimizer:$PYTHONPATH \
python3 prune_deepseek_v4_4b_config.py \
    --hf_config "${MODEL_PATH}/config.json" \
    --target_total_params "$TARGET_TOTAL_PARAMS" \
    --output "${OUTPUT_BASE}_best_config.json" \
    --top_k 5

BEST_CONFIG="${OUTPUT_BASE}_best_config.json"
if [ ! -f "$BEST_CONFIG" ]; then
    echo "ERROR: Best configuration not found at $BEST_CONFIG"
    exit 1
fi

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
ESTIMATED_TOTAL=$(jq -r '.estimated_total_params' "$BEST_CONFIG")

echo "Applying best configuration:"
echo "  hidden_size: $HIDDEN_SIZE"
echo "  num_layers: $NUM_LAYERS"
echo "  num_moe_experts: $NUM_EXPERTS"
echo "  moe_ffn_hidden_size: $FFN_SIZE"
echo "  moe_shared_expert_intermediate_size: $SHARED_FFN"
echo "  estimated_total_params: $ESTIMATED_TOTAL"
echo ""

# ─────────────────────────────────────────────────────────────────────
# Step 2: 真裁剪 (激活重要性 + V4 后处理)
# ─────────────────────────────────────────────────────────────────────
echo "========================================================================"
echo "STEP 2: Apply pruning with activation-based importance"
echo "========================================================================"
echo ""

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
    --calibration_samples $CALIBRATION_SAMPLES \
    --seq_length $SEQ_LENGTH \
    --validation_samples 0 \
    --trust_remote_code

echo ""
echo "=== Waiting for Step 2 processes to release GPUs ==="
wait_for_gpu_free 60 || {
    echo "  Step 2 left GPU busy — killing stragglers"
    kill_gpu_python
    wait_for_gpu_free 30
}

# ─────────────────────────────────────────────────────────────────────
# Step 3: 转换为 HF / SGLang 格式
# ─────────────────────────────────────────────────────────────────────
echo "========================================================================"
echo "STEP 3: Convert pruned Megatron shards to HF format"
echo "========================================================================"
echo ""

MEGATRON_CKPT_DIR="${OUTPUT_BASE}-final_megatron"
HF_CONFIG="${OUTPUT_BASE}-final/config.json"
OUTPUT_DIR="${OUTPUT_BASE}-final-sglang"

if [ ! -d "$MEGATRON_CKPT_DIR" ]; then
    echo "ERROR: Megatron checkpoint directory not found: $MEGATRON_CKPT_DIR"
    exit 1
fi

PYTHONPATH=/opt/Model-Optimizer:$PYTHONPATH \
python3 convert_pruned_to_hf.py \
    --megatron_ckpt "$MEGATRON_CKPT_DIR" \
    --hf_config "$HF_CONFIG" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "========================================================================"
echo "Pruning pipeline complete!"
echo "========================================================================"
echo ""
echo "Final 4B pruned model (HF format): $OUTPUT_DIR"
echo "Best config: $BEST_CONFIG"
echo ""
echo "Next steps:"
echo "  1. Knowledge distillation:"
echo "       Local teacher:  examples/llm_distill/distill_deepseek_v4_4b.py --teacher_model <path>"
echo "       Service teacher: examples/llm_distill/serve_teacher_deepseek_v4.py (teacher node)"
echo "                        examples/llm_distill/distill_deepseek_v4_4b.py --teacher_endpoint <url>"
echo "  2. Continual pre-training: see examples/megatron_bridge/cpt_deepseek_v4_4b.py"
echo "  3. Instruction fine-tuning: see examples/megatron_bridge/sft_deepseek_v4_4b.py"
echo "  4. Quantization & deployment: see examples/llm_ptq/hf_ptq.py"
echo ""

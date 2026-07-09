#!/usr/bin/env bash
# ==============================================================================
# DeepSeek-V4 Pruned 预训练启动脚本
# ==============================================================================

set -euo pipefail

# ==============================================================================
# 用户配置区
# ==============================================================================

# ---- 路径 ----
MODEL="/workdir/model_input/dpsk-v4-4B-A1.5B"
DATASET_DIR="/workdir/data/alpaca_train_text_document"
OUTPUT_DIR="/workdir/model_output_resume"
BRIDGE_DIR="/workdir/Megatron-Bridge"
PYTHON_SCRIPT="/workdir/pretrain_dpskv4_pruned.py"
# ---- 恢复训练 ----
# 设置为 checkpoint 路径即可恢复训练，留空则从零/微调启动
LOAD_FROM_CHECKPOINT="/data/wanghui/deepseekv4_train_mcore_bridge/model_output/checkpoints/iter_0000002"
# ---- 数据模式 ----
USE_MOCK_DATA=false

# ---- 并行度 ----
TENSOR_MODEL_PARALLEL_SIZE=1
PIPELINE_MODEL_PARALLEL_SIZE=4
PIPELINE_MODEL_PARALLEL_LAYOUT="Et*4|t*4|t*4|t*4L"
EXPERT_MODEL_PARALLEL_SIZE=2
CONTEXT_PARALLEL_SIZE=1
NUM_NODES=1

# ---- 序列长度 ----
MAX_LENGTH=1024

# ---- Batch size / 训练 ----
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=32
TRAIN_ITERS=5
FINETUNE=true
PADDING_FREE=false

# ---- 学习率 / 优化器 ----
LR=1.8e-4
MIN_LR=1.8e-5
LR_WARMUP_FRACTION=0.04
WEIGHT_DECAY=0.1
CLIP_GRAD=1.0
ADAM_BETA1=0.9
ADAM_BETA2=0.95

# ---- Recompute ----
RECOMPUTE_GRANULARITY="full"
RECOMPUTE_METHOD="uniform"
RECOMPUTE_NUM_LAYERS=1

# ---- MoE ----
MOE_PERMUTE_FUSION=true
MOE_GROUPED_GEMM=true
MOE_SHARED_EXPERT_OVERLAP=true
MOE_AUX_LOSS_COEFF=0.0
MOE_Z_LOSS_COEFF=0.0
MOE_TOKEN_DISPATCHER_TYPE="alltoall"

# ---- Attention / Kernel ----
APPLY_ROPE_FUSION=true
USE_FUSED_MHC=true
APPLY_DSA_KERNEL_FUSION=false
CROSS_ENTROPY_LOSS_FUSION=true
MTP_NUM_LAYERS=0

# ---- Checkpoint ----
SAVE_STEPS=2
SAVE_TOTAL_LIMIT=10
NO_SAVE_OPTIM=false
NO_SAVE_RNG=false
ASYNC_SAVE=false



# ---- 验证 / 日志 ----
EVAL_INTERVAL=200
EVAL_ITERS=5
LOGGING_STEPS=1
DATALOADER_NUM_WORKERS=4
SEED=1234

# ==============================================================================
# 启动训练
# ==============================================================================

NPROC_PER_NODE=$((TENSOR_MODEL_PARALLEL_SIZE * PIPELINE_MODEL_PARALLEL_SIZE * EXPERT_MODEL_PARALLEL_SIZE))
mkdir -p "${OUTPUT_DIR}"

LOG_FILE="${OUTPUT_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="${OUTPUT_DIR}/train.pid"

# 构建 CLI 参数
ARGS=(
    --model "${MODEL}"
    --output_dir "${OUTPUT_DIR}"
    --max_length "${MAX_LENGTH}"
    --tensor_model_parallel_size "${TENSOR_MODEL_PARALLEL_SIZE}"
    --pipeline_model_parallel_size "${PIPELINE_MODEL_PARALLEL_SIZE}"
    --expert_model_parallel_size "${EXPERT_MODEL_PARALLEL_SIZE}"
    --context_parallel_size "${CONTEXT_PARALLEL_SIZE}"
    --micro_batch_size "${MICRO_BATCH_SIZE}"
    --global_batch_size "${GLOBAL_BATCH_SIZE}"
    --train_iters "${TRAIN_ITERS}"
    --seed "${SEED}"
    --lr "${LR}"
    --min_lr "${MIN_LR}"
    --lr_warmup_fraction "${LR_WARMUP_FRACTION}"
    --weight_decay "${WEIGHT_DECAY}"
    --clip_grad "${CLIP_GRAD}"
    --adam_beta1 "${ADAM_BETA1}"
    --adam_beta2 "${ADAM_BETA2}"
    --recompute_granularity "${RECOMPUTE_GRANULARITY}"
    --recompute_method "${RECOMPUTE_METHOD}"
    --recompute_num_layers "${RECOMPUTE_NUM_LAYERS}"
    --moe_aux_loss_coeff "${MOE_AUX_LOSS_COEFF}"
    --moe_token_dispatcher_type "${MOE_TOKEN_DISPATCHER_TYPE}"
    --mtp_num_layers "${MTP_NUM_LAYERS}"
    --save_steps "${SAVE_STEPS}"
    --save_total_limit "${SAVE_TOTAL_LIMIT}"
    --eval_interval "${EVAL_INTERVAL}"
    --eval_iters "${EVAL_ITERS}"
    --logging_steps "${LOGGING_STEPS}"
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
)

[ -n "${PIPELINE_MODEL_PARALLEL_LAYOUT}" ] && ARGS+=(--pipeline_model_parallel_layout "${PIPELINE_MODEL_PARALLEL_LAYOUT}")
[ "${USE_MOCK_DATA}" = "true" ] && ARGS+=(--use_mock_data)
[ "${FINETUNE}" = "true" ] && ARGS+=(--finetune)
[ "${PADDING_FREE}" = "true" ] && ARGS+=(--padding_free)
[ "${MOE_PERMUTE_FUSION}" = "true" ] && ARGS+=(--moe_permute_fusion)
[ "${MOE_GROUPED_GEMM}" = "true" ] && ARGS+=(--moe_grouped_gemm)
[ "${MOE_SHARED_EXPERT_OVERLAP}" = "true" ] && ARGS+=(--moe_shared_expert_overlap)
[ "${APPLY_ROPE_FUSION}" = "true" ] && ARGS+=(--apply_rope_fusion)
[ "${USE_FUSED_MHC}" = "true" ] && ARGS+=(--use_fused_mhc)
[ "${APPLY_DSA_KERNEL_FUSION}" = "true" ] && ARGS+=(--apply_dsa_kernel_fusion)
[ "${CROSS_ENTROPY_LOSS_FUSION}" = "true" ] && ARGS+=(--cross_entropy_loss_fusion)
[ "${NO_SAVE_OPTIM}" = "true" ] && ARGS+=(--no_save_optim)
[ "${NO_SAVE_RNG}" = "true" ] && ARGS+=(--no_save_rng)
[ "${ASYNC_SAVE}" = "true" ] && ARGS+=(--async_save)
[ -n "${LOAD_FROM_CHECKPOINT}" ] && ARGS+=(--load_from_checkpoint "${LOAD_FROM_CHECKPOINT}")

if [ "${USE_MOCK_DATA}" = "false" ]; then
    ARGS+=(--dataset_dir "${DATASET_DIR}")
fi

# 打印配置
echo "======================================"
echo "DeepSeek-V4 Pruned Pretraining"
echo "======================================"
echo "Model:       ${MODEL}"
echo "Output:      ${OUTPUT_DIR}"
echo "Log:         ${LOG_FILE}"
echo "Parallelism: TP=${TENSOR_MODEL_PARALLEL_SIZE} PP=${PIPELINE_MODEL_PARALLEL_SIZE} EP=${EXPERT_MODEL_PARALLEL_SIZE} CP=${CONTEXT_PARALLEL_SIZE}"
echo "Layout:      ${PIPELINE_MODEL_PARALLEL_LAYOUT}"
echo "Batch:       micro=${MICRO_BATCH_SIZE} global=${GLOBAL_BATCH_SIZE}"
echo "Finetune:    ${FINETUNE}"
echo "LR:          ${LR} -> ${MIN_LR}"
echo "======================================"

# 后台启动
cd "${BRIDGE_DIR}"
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
nohup setsid uv run --no-sync torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --nnodes="${NUM_NODES}" \
    "${PYTHON_SCRIPT}" \
    "${ARGS[@]}" \
    > "${LOG_FILE}" 2>&1 < /dev/null &

TRAIN_PID=$!
echo "${TRAIN_PID}" > "${PID_FILE}"

echo "Training PID: ${TRAIN_PID}"
echo "  tail -f ${LOG_FILE}"
echo "  kill -TERM \$(cat ${PID_FILE})"

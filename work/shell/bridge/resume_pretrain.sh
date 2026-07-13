#!/usr/bin/env bash
# ==============================================================================
# DeepSeek-V4 Pruned 恢复训练脚本 (Resume Training)
# ==============================================================================
#
# 从 run_pretrain.sh 保存的 checkpoint 恢复训练, 加载 optimizer + rng state,
# 从上次保存的 iteration 继续训练.
#
# 与 run_pretrain.sh 的关键区别:
#   1. LOAD_FROM_CHECKPOINT 指向上一次训练保存的 checkpoints 目录
#   2. NO_LOAD_OPTIM=false  — 加载 optimizer state (初始训练时跳过因为 distcp 无 optim)
#   3. NO_LOAD_RNG=false    — 加载 rng state
#   4. FINETUNE=false       — resume 模式 (非 finetune), 保留 iteration 计数
#   5. OUTPUT_DIR 使用新目录, 避免覆盖之前的 checkpoint
#   6. TRAIN_ITERS 是总目标迭代数 (不是剩余), Megatron 从 checkpoint 的 iteration 继续到该值
#
# 用法:
#   1. 修改 PREV_OUTPUT_DIR 指向上一次训练的输出目录
#   2. 修改 TRAIN_ITERS 为总目标迭代数 (已完成的 + 要继续的)
#   3. bash resume_pretrain.sh
# ==============================================================================

set -euo pipefail

# ==============================================================================
# 用户配置区
# ==============================================================================

# ---- 路径 ----
MODEL="/workdir/model_input/dpsk-v4-4B-A1.5B"
DATASET_DIR="/workdir/data/alpaca_train_text_document"

# 上一次训练的输出目录 (从中加载 checkpoint)
PREV_OUTPUT_DIR="/workdir/model_output/phase1"
# 本次 resume 训练的输出目录 (保存新 checkpoint, 避免覆盖)
OUTPUT_DIR="/workdir/model_output/phase2"

BRIDGE_DIR="/workdir/Megatron-Bridge"
PYTHON_SCRIPT="/workdir/pretrain_dpskv4_pruned.py"

# ---- Checkpoint 加载 (从上一次训练保存的 checkpoint 恢复) ----
# 指向 PREV_OUTPUT_DIR/checkpoints, Megatron 会自动读取 latest_checkpointed_iteration.txt
# 找到最新的 iter_xxxx 并加载 (含 optimizer + rng state)
LOAD_FROM_CHECKPOINT="${PREV_OUTPUT_DIR}/checkpoints"

# ---- 数据模式 ----
USE_MOCK_DATA=false

# ---- 并行度 (必须与初始训练一致, 否则无法加载 checkpoint) ----
TENSOR_MODEL_PARALLEL_SIZE=1
PIPELINE_MODEL_PARALLEL_SIZE=4
PIPELINE_MODEL_PARALLEL_LAYOUT="Et*4|t*4|t*4|t*4L"
EXPERT_MODEL_PARALLEL_SIZE=2
CONTEXT_PARALLEL_SIZE=1
NUM_NODES=1

# ---- 序列长度 (必须与初始训练一致) ----
MAX_LENGTH=1024

# ---- Batch size / 训练 ----
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=32
# 总目标迭代数 (已完成的 + 要继续的)
# 例: 上次训练到 iter 2, 想再训练 998 步 → TRAIN_ITERS=1000
TRAIN_ITERS=1000
FINETUNE=false
PADDING_FREE=false

# ---- Checkpoint 加载控制 ----
# Resume 模式: 加载 optimizer + rng (与初始训练相反)
NO_LOAD_OPTIM=false
NO_LOAD_RNG=false

# ---- Checkpoint 保存 ----
SAVE_STEPS=100
SAVE_TOTAL_LIMIT=10
NO_SAVE_OPTIM=false
NO_SAVE_RNG=false
ASYNC_SAVE=false

# ---- 学习率 / 优化器 (与初始训练一致) ----
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
MOE_PERMUTE_FUSION=false
MOE_GROUPED_GEMM=true
MOE_SHARED_EXPERT_OVERLAP=false
MOE_AUX_LOSS_COEFF=0.0
MOE_Z_LOSS_COEFF=0.0
MOE_TOKEN_DISPATCHER_TYPE="alltoall"

# ---- Attention / Kernel ----
APPLY_ROPE_FUSION=true
USE_FUSED_MHC=true
APPLY_DSA_KERNEL_FUSION=false
CROSS_ENTROPY_LOSS_FUSION=true
MTP_NUM_LAYERS=0

# ---- 验证 / 日志 ----
EVAL_INTERVAL=0    # 0 = 禁用 eval
EVAL_ITERS=0
LOGGING_STEPS=1
DATALOADER_NUM_WORKERS=4
SEED=1234

# ==============================================================================
# 派生变量
# ==============================================================================
NPROC_PER_NODE=$((TENSOR_MODEL_PARALLEL_SIZE * PIPELINE_MODEL_PARALLEL_SIZE * EXPERT_MODEL_PARALLEL_SIZE))
mkdir -p "${OUTPUT_DIR}"

# ==============================================================================
# 启动训练
# ==============================================================================

LOG_FILE="${OUTPUT_DIR}/resume_$(date +%Y%m%d_%H%M%S).log"
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
    --moe_z_loss_coeff "${MOE_Z_LOSS_COEFF}"
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
[ "${NO_LOAD_OPTIM}" = "true" ] && ARGS+=(--no_load_optim)
[ "${NO_LOAD_RNG}" = "true" ] && ARGS+=(--no_load_rng)
[ "${ASYNC_SAVE}" = "true" ] && ARGS+=(--async_save)
[ -n "${LOAD_FROM_CHECKPOINT}" ] && ARGS+=(--load_from_checkpoint "${LOAD_FROM_CHECKPOINT}")

if [ "${USE_MOCK_DATA}" = "false" ]; then
    ARGS+=(--dataset_dir "${DATASET_DIR}")
fi

# 打印配置
echo "======================================"
echo "DeepSeek-V4 Pruned Resume Training"
echo "======================================"
echo "  Load checkpoint:  ${LOAD_FROM_CHECKPOINT}"
echo "  Model (config):   ${MODEL}"
echo "  Output:           ${OUTPUT_DIR}"
echo "  Log:              ${LOG_FILE}"
echo "  Parallelism:      TP=${TENSOR_MODEL_PARALLEL_SIZE} PP=${PIPELINE_MODEL_PARALLEL_SIZE} EP=${EXPERT_MODEL_PARALLEL_SIZE} CP=${CONTEXT_PARALLEL_SIZE}"
echo "  Batch:            micro=${MICRO_BATCH_SIZE} global=${GLOBAL_BATCH_SIZE}"
echo "  Train iters:      ${TRAIN_ITERS} (total, resumed from checkpoint)"
echo "  Finetune:         ${FINETUNE}"
echo "  load_optim:       $([ "${NO_LOAD_OPTIM}" = "false" ] && echo "true" || echo "false")"
echo "  load_rng:         $([ "${NO_LOAD_RNG}" = "false" ] && echo "true" || echo "false")"
echo "  LR:               ${LR} -> ${MIN_LR}"
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

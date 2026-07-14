#!/bin/bash
# ==============================================================================
# DeepSeek-V4 Pruned (dpsk-v4-4B-A1.5B) Pretraining — Single-Node 8-GPU
#
# This script follows the official slurm_pretrain.sh pattern but is adapted for
# a local single-node setup (no Slurm, no container).
#
# Usage:
#   bash run_pretrain_pruned.sh
#
# The recipe (deepseek_v4_pruned_pretrain_8gpu_bf16_config) hardcodes the model
# path. All other parameters are overridable via CLI_OVERRIDES (key=value format),
# exactly like the official flow.
# ==============================================================================

set -euo pipefail

# ==============================================================================
# Configuration (按类别集中管理，修改只需改这里)
# ==============================================================================

# ---- 1. 环境 / 路径 ----
WORKSPACE=${WORKSPACE:-/workdir}
BRIDGE_DIR=${BRIDGE_DIR:-/workdir/Megatron-Bridge}

# ---- 2. Recipe / 数据 ----
RECIPE_NAME=deepseek_v4_pruned_pretrain_8gpu_bf16_config
DATASET_NAME=local                                        # "mock" 或 "local"
DATASET_BLEND_PATH="/workdir/data/recovery_cpt_seq4k_text_document"
SEQ_LENGTH=1024

# ---- 3. 训练参数 ----
TRAIN_ITERS=20000                                         # 1 epoch
GLOBAL_BATCH_SIZE=64
MICRO_BATCH_SIZE=1
EVAL_INTERVAL=0                                           # 0 = disabled
EVAL_ITERS=0
LR_WARMUP_ITERS=40
SAVE_INTERVAL=1000
LOG_INTERVAL=5
CHECKPOINT_KEEP_LIMIT=5                                   # 保留最近 N 个 checkpoint
SEED=1234

# ---- 4. 并行配置 (TP,PP,EP,CP; 8 GPUs: 1*1*8*1=8) ----
PARALLELISM_CONFIG=${PARALLELISM_CONFIG:-1,1,8,1}

# ---- 5. Checkpoint 加载 / 保存 ----
PRETRAINED_CHECKPOINT="/workdir/model_input/checkpoint-10000"  # 空字符串 = random init
FINETUNE=true                                             # true=从HF加载后重置训练状态
NO_LOAD_OPTIM=true                                        # true=不加载优化器状态
NO_LOAD_RNG=true                                          # true=不加载 RNG 状态
NO_SAVE_OPTIM=false                                       # false=保存优化器状态
NO_SAVE_RNG=false                                         # false=保存 RNG 状态

# ---- 6. 输出 / 日志 ----
OUTPUT_DIR="/workdir/model_output/phase2_recipe_callback"
# TensorBoard 每次新训练自动创建带时间戳的子目录，避免多个 run 的 events 文件混在一起
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${OUTPUT_DIR}/tb_logs/run_$(date +%Y%m%d_%H%M%S)}"

# ---- 7. MoE 监控 Callback ----
# true  → 使用 run_recipe_with_monitor.py (注册 MoEMonitorCallback)
# false → 使用官方 run_recipe.py (无 callback)
MOE_MONITOR_ENABLED=true
MOE_MONITOR_TOP_K=6                                       # DeepSeek-V4 的 top_k
# true  → 官方 training_log 行 + callback MoE 路由指标行 (两行输出)
# false → callback 完全替代官方 iteration log (单行输出)
MOE_MONITOR_COMPARE_MODE=true

# ==============================================================================
# Environment Setup
# ==============================================================================

export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0
export NCCL_PXN_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Export MoE monitor + TB env vars (consumed by run_recipe_with_monitor.py)
export MOE_MONITOR_ENABLED
export MOE_MONITOR_TOP_K
export MOE_MONITOR_COMPARE_MODE
export OUTPUT_DIR
export TENSORBOARD_DIR

# ==============================================================================
# Job Execution
# ==============================================================================

mkdir -p "${OUTPUT_DIR}"

if [ "$DATASET_NAME" = "mock" ]; then
    DATASET_TYPE=llm-pretrain-mock
    DATASET_OVERRIDES=""
else
    DATASET_TYPE=llm-pretrain
    DATASET_OVERRIDES="dataset.blend=[[\"${DATASET_BLEND_PATH}\"],null] dataset.split='\"9999,8,2\"'"
fi

# Parse parallelism
OLD_IFS=$IFS
IFS=',' read -r TP PP EP CP <<< "$PARALLELISM_CONFIG"
IFS=$OLD_IFS

NNODES=1
NPROC_PER_NODE=$((TP * PP * EP))
MASTER_PORT=${MASTER_PORT:-29571}
MASTER_ADDR=${MASTER_ADDR:-localhost}

RUN_NAME=dpsk_v4_pruned_${DATASET_NAME}_tp${TP}_pp${PP}_ep${EP}
CHECKPOINT_DIR=${OUTPUT_DIR}/checkpoints

# Build CLI overrides (pretrained_checkpoint is optional)
PRETRAINED_OVERRIDES=""
if [ -n "${PRETRAINED_CHECKPOINT}" ]; then
    PRETRAINED_OVERRIDES="checkpoint.pretrained_checkpoint=${PRETRAINED_CHECKPOINT}"
fi

# 启用 MoE 监控时, 默认保留 Megatron 官方 iteration log (对比模式),
# callback 只补充 MoE 路由指标作为第二行。
# 如需 callback 完全替代官方 log (单行输出), 把 MOE_MONITOR_COMPARE_MODE 改为 false
# 并把下面的 MOE_MONITOR_LOG_OVERRIDE 取消注释。
if [ "${MOE_MONITOR_ENABLED}" = "true" ] && [ "${MOE_MONITOR_COMPARE_MODE}" != "true" ]; then
    MOE_MONITOR_LOG_OVERRIDE="logger.skip_train_metrics_log=true"
else
    MOE_MONITOR_LOG_OVERRIDE=""
fi

CLI_OVERRIDES=" \
    model.seq_length=$SEQ_LENGTH \
    dataset.sequence_length=$SEQ_LENGTH \
    train.train_iters=$TRAIN_ITERS \
    train.global_batch_size=$GLOBAL_BATCH_SIZE \
    train.micro_batch_size=$MICRO_BATCH_SIZE \
    validation.eval_interval=$EVAL_INTERVAL \
    validation.eval_iters=$EVAL_ITERS \
    scheduler.lr_warmup_iters=$LR_WARMUP_ITERS \
    scheduler.lr_decay_iters=$TRAIN_ITERS \
    checkpoint.save=${CHECKPOINT_DIR} \
    checkpoint.load=${CHECKPOINT_DIR} \
    checkpoint.save_interval=$SAVE_INTERVAL \
    checkpoint.most_recent_k=$CHECKPOINT_KEEP_LIMIT \
    logger.log_interval=$LOG_INTERVAL \
    $MOE_MONITOR_LOG_OVERRIDE \
    model.tensor_model_parallel_size=$TP \
    model.pipeline_model_parallel_size=$PP \
    model.expert_model_parallel_size=$EP \
    model.context_parallel_size=$CP \
    $PRETRAINED_OVERRIDES \
    checkpoint.finetune=$FINETUNE \
    checkpoint.load_optim=$([ "$NO_LOAD_OPTIM" = "true" ] && echo "false" || echo "true") \
    checkpoint.load_rng=$([ "$NO_LOAD_RNG" = "true" ] && echo "false" || echo "true") \
    checkpoint.save_optim=$([ "$NO_SAVE_OPTIM" = "true" ] && echo "false" || echo "true") \
    checkpoint.save_rng=$([ "$NO_SAVE_RNG" = "true" ] && echo "false" || echo "true") \
    rng.seed=$SEED \
    $DATASET_OVERRIDES"

# Select entry script based on MOE_MONITOR_ENABLED
# true  → run_recipe_with_monitor.py  (auto-registers MoEMonitorCallback)
# false → run_recipe.py               (official, no callback)
if [ "${MOE_MONITOR_ENABLED}" = "true" ]; then
    ENTRY_SCRIPT="${BRIDGE_DIR}/examples/models/deepseek_v4/run_recipe_with_monitor.py"
else
    ENTRY_SCRIPT="${BRIDGE_DIR}/scripts/training/run_recipe.py"
fi

CMD="uv run --no-sync python -m torch.distributed.run \
    --nproc_per_node=$NPROC_PER_NODE \
    --nnodes=$NNODES \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    ${ENTRY_SCRIPT} \
    --recipe $RECIPE_NAME \
    --dataset $DATASET_TYPE \
    --step_func gpt_step \
    $CLI_OVERRIDES"

echo "======================================"
echo "DeepSeek-V4 Pruned Pretraining (Recipe Flow)"
echo "======================================"
echo "Recipe: $RECIPE_NAME"
echo "Dataset: $DATASET_TYPE/$DATASET_NAME"
echo "Parallelism: TP=$TP PP=$PP EP=$EP CP=$CP"
if [ -n "${PRETRAINED_CHECKPOINT}" ]; then
    echo "Checkpoint load: $PRETRAINED_CHECKPOINT (HF pretrained_checkpoint)"
else
    echo "Checkpoint load: none (random init, like official pretraining)"
fi
echo "Checkpoint save: $CHECKPOINT_DIR"
echo "Output dir: $OUTPUT_DIR"
if [ "${MOE_MONITOR_ENABLED}" = "true" ]; then
    echo "MoE monitor:    ENABLED  (top_k=$MOE_MONITOR_TOP_K, frequency=LOG_INTERVAL=$LOG_INTERVAL)"
    if [ "${MOE_MONITOR_COMPARE_MODE}" = "true" ]; then
        echo "                 对比模式: 官方 training_log 行 + callback MoE 路由指标行 (两行输出)"
    else
        echo "                 callback 完全替代官方 iteration log, 输出包含全部基础 + 新增指标, 用 | 分隔。"
    fi
else
    echo "MoE monitor:    disabled (set MOE_MONITOR_ENABLED=true to enable)"
fi
echo "======================================"
echo "$CMD"
echo "======================================"

cd "${BRIDGE_DIR}"
export PYTHONPATH=${BRIDGE_DIR}/src:${BRIDGE_DIR}/3rdparty/Megatron-LM:${PYTHONPATH:-}

# Launch in background (like run_pretrain.sh)
LOG_FILE="${OUTPUT_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="${OUTPUT_DIR}/train.pid"

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
nohup setsid $CMD \
    > "${LOG_FILE}" 2>&1 < /dev/null &

TRAIN_PID=$!
echo "${TRAIN_PID}" > "${PID_FILE}"

echo "Training PID: ${TRAIN_PID}"
echo "  tail -f ${LOG_FILE}"
echo "  kill -TERM \$(cat ${PID_FILE})"

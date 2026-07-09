#!/usr/bin/env bash
# ==============================================================================
# DeepSeek-V4 Pruned (4B-A1.5B) 预训练 - 单节点版本
#
# 基于官方 slurm_pretrain.sh 改写，去除 Slurm 依赖，适配单节点 8x RTX 5090。
#
# 前置条件:
#   1. fast_hadamard_transform 已安装
#   2. 数据已预处理 (prepare_data.sh)
#   3. 在 docker 容器内执行
#
# 用法:
#   bash pretrain_dpskv4_pruned.sh              # 使用真实数据
#   bash pretrain_dpskv4_pruned.sh mock         # mock 数据验证流程
# ==============================================================================

set -euo pipefail

# ==============================================================================
# 配置区 - 所有超参数集中在这里
# ==============================================================================

# ---- 路径 ----
WORKSPACE=${WORKSPACE:-/workdir}
HF_CONFIG=${HF_CONFIG:-/workdir/model_input/dpsk-v4-4B-A1.5B}
DATA_PREFIX=${DATA_PREFIX:-/workdir/data/alpaca_train_text_document}
OUTPUT_DIR=${OUTPUT_DIR:-/workdir/model_output}
RUN_NAME=${RUN_NAME:-dsv4_pruned_pretrain}
RECIPE_NAME=deepseek_v4_pruned_pretrain_8gpu_rtb5090_bf16_config

# ---- 数据模式: real 或 mock ----
DATA_MODE=${1:-real}

# ---- 并行度 (TP,PP,EP,CP) ----
PARALLELISM_CONFIG=${PARALLELISM_CONFIG:-1,1,8,1}

# ---- 序列长度 ----
SEQ_LENGTH=${SEQ_LENGTH:-1024}

# ---- 训练超参数 ----
TRAIN_ITERS=${TRAIN_ITERS:-1000}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-8}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}

# ---- 优化器/学习率 ----
LR=${LR:-3e-4}
MIN_LR=${MIN_LR:-3e-5}
LR_WARMUP_ITERS=${LR_WARMUP_ITERS:-50}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}

# ---- 验证 ----
EVAL_INTERVAL=${EVAL_INTERVAL:-200}
EVAL_ITERS=${EVAL_ITERS:-5}

# ---- Checkpoint ----
SAVE_INTERVAL=${SAVE_INTERVAL:-200}
SAVE_CKPT=${SAVE_CKPT:-true}  # true=保存, false=不保存
LOAD_CKPT=${LOAD_CKPT:-false} # true=从 checkpoint 恢复

# ---- 日志 ----
LOG_INTERVAL=${LOG_INTERVAL:-1}
WANDB_PROJECT=${WANDB_PROJECT:-megatron-bridge-dsv4-pruned}
USE_WANDB=${USE_WANDB:-false}  # true=启用 wandb

# ==============================================================================
# 环境变量 (NCCL / CUDA)
# ==============================================================================
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0
export NCCL_PXN_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_MAX_CONNECTIONS=1

# ==============================================================================
# 路径与并行度解析
# ==============================================================================
BRIDGE_DIR=/workdir/Megatron-Bridge

OLD_IFS=$IFS
IFS=',' read -r TP PP EP CP <<< "$PARALLELISM_CONFIG"
IFS=$OLD_IFS

NPROC_PER_NODE=$((TP * PP * EP))
NNODES=1

CHECKPOINT_DIR="${OUTPUT_DIR}/${RUN_NAME}/checkpoints"
TENSORBOARD_DIR="${OUTPUT_DIR}/${RUN_NAME}/tb_logs"
mkdir -p "${CHECKPOINT_DIR}" "${TENSORBOARD_DIR}"

# ==============================================================================
# 数据集配置
# ==============================================================================
if [ "${DATA_MODE}" = "mock" ]; then
    DATASET_TYPE=llm-pretrain-mock
else
    DATASET_TYPE=llm-pretrain
fi

# ==============================================================================
# CLI Overrides - 拼装所有训练超参数
# ==============================================================================
CLI_OVERRIDES=" \
    model.seq_length=${SEQ_LENGTH} \
    dataset.sequence_length=${SEQ_LENGTH} \
    train.train_iters=${TRAIN_ITERS} \
    train.global_batch_size=${GLOBAL_BATCH_SIZE} \
    train.micro_batch_size=${MICRO_BATCH_SIZE} \
    validation.eval_interval=${EVAL_INTERVAL} \
    validation.eval_iters=${EVAL_ITERS} \
    scheduler.lr_warmup_iters=${LR_WARMUP_ITERS} \
    scheduler.lr_decay_iters=${TRAIN_ITERS} \
    scheduler.weight_decay=${WEIGHT_DECAY} \
    optimizer.lr=${LR} \
    optimizer.min_lr=${MIN_LR} \
    checkpoint.save=${CHECKPOINT_DIR} \
    checkpoint.save_interval=${SAVE_INTERVAL} \
    checkpoint.async_save=true \
    logger.log_interval=${LOG_INTERVAL} \
    logger.tensorboard_dir=${TENSORBOARD_DIR} \
    model.tensor_model_parallel_size=${TP} \
    model.pipeline_model_parallel_size=${PP} \
    model.expert_model_parallel_size=${EP} \
    model.context_parallel_size=${CP}"

# 真实数据时指定数据路径
if [ "${DATA_MODE}" != "mock" ]; then
    CLI_OVERRIDES="$CLI_OVERRIDES dataset.blend=[[\\\"$DATA_PREFIX\\\"],null] dataset.split='\\\"9999,8,2\\\"'"
fi

# 可选：checkpoint 恢复
if [ "${LOAD_CKPT}" = "true" ]; then
    CLI_OVERRIDES="$CLI_OVERRIDES checkpoint.load=${CHECKPOINT_DIR}"
fi

# 可选：禁用 checkpoint 保存
if [ "${SAVE_CKPT}" = "false" ]; then
    CLI_OVERRIDES="$CLI_OVERRIDES checkpoint.save_interval=999999999"
fi

# 可选：启用 wandb
if [ "${USE_WANDB}" = "true" ]; then
    CLI_OVERRIDES="$CLI_OVERRIDES logger.wandb_project=${WANDB_PROJECT} logger.wandb_exp_name=${RUN_NAME}"
fi

# ==============================================================================
# 构建启动命令
# ==============================================================================
CMD="cd ${BRIDGE_DIR} && uv run --no-sync python -m torch.distributed.run \
    --nproc_per_node=$NPROC_PER_NODE \
    --nnodes=$NNODES \
    scripts/training/run_recipe.py \
    --recipe $RECIPE_NAME \
    --dataset $DATASET_TYPE \
    --step_func gpt_step \
    $CLI_OVERRIDES"

# ==============================================================================
# 打印配置
# ==============================================================================
echo "======================================"
echo "DeepSeek-V4 Pruned (4B-A1.5B) Pretraining"
echo "======================================"
echo "Recipe:        $RECIPE_NAME"
echo "HF Config:     $HF_CONFIG"
echo "Hardware:      ${NPROC_PER_NODE}x RTX 5090 (Blackwell)"
echo "Parallelism:   TP=$TP PP=$PP EP=$EP CP=$CP"
echo "Data mode:     $DATA_MODE"
echo "Seq length:    $SEQ_LENGTH"
echo "Batch size:    global=$GLOBAL_BATCH_SIZE micro=$MICRO_BATCH_SIZE"
echo "Train iters:   $TRAIN_ITERS"
echo "LR:            $LR (min: $MIN_LR, warmup: $LR_WARMUP_ITERS)"
echo "Weight decay:  $WEIGHT_DECAY"
echo "Eval:          interval=$EVAL_INTERVAL iters=$EVAL_ITERS"
echo "Save:          interval=$SAVE_INTERVAL enabled=$SAVE_CKPT"
echo "Load ckpt:     $LOAD_CKPT"
echo "Wandb:         $USE_WANDB (project: $WANDB_PROJECT)"
echo "Run name:      $RUN_NAME"
echo "Checkpoint:    $CHECKPOINT_DIR"
echo "TensorBoard:   $TENSORBOARD_DIR"
echo "======================================"
echo ""
echo "Command:"
echo "$CMD"
echo "======================================"

# ==============================================================================
# 执行
# ==============================================================================
eval "$CMD"

echo ""
echo "======================================"
echo "Training complete!"
echo "Checkpoints:  $CHECKPOINT_DIR"
echo "TensorBoard:  $TENSORBOARD_DIR"
echo "======================================"

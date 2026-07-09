#!/bin/bash
# DeepSeek-V4 裁剪模型 CPT 训练脚本 - 带完整 MoE 监控
# 
# 功能：
#   ✅ MoE 路由监控（Router Entropy、Expert Utilization、Load Balance Loss）
#   ✅ 训练指标监控（loss、grad_norm、learning_rate）
#   ✅ 阶段进度监控（如果使用 moe_phase callback）
#   ✅ TensorBoard 日志记录
#
# 使用方法:
#   bash train_cpt_full_deepseek_v4_with_monitor.sh 2>&1 | tee train_cpt_full_deepseek_v4_with_monitor.log
export LD_LIBRARY_PATH=/mnt/nvme4n1/miniconda3/envs/ms-swift-train/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
# ModelScope 缓存重定向到大盘
export MODELSCOPE_CACHE=/mnt/nvme4n1/modelscope_cache
# HuggingFace datasets 缓存也一起重定向（保险）
export HF_DATASETS_CACHE=/mnt/nvme4n1/hf_cache/datasets
export HF_HOME=/mnt/nvme4n1/hf_cache
# 临时目录也指到大盘
export TMPDIR=/mnt/nvme4n1/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# ===== 后台运行逻辑 =====
if [ "$1" != "--daemon" ]; then
    LOG=/mnt/nvme4n1/dpsk-v4-train/train/deepseekv4_train/train_cpt_full_deepseek_v4_with_monitor.log
    nohup setsid bash "$0" --daemon > "$LOG" 2>&1 &
    echo "CPT 训练（带监控）已在后台启动，PID: $!"
    echo "查看日志: tail -f $LOG"
    echo "查看 TensorBoard: tensorboard --logdir /data/wanghui/deepseekv4_train/megatron_output/dpsk-v4-4B-A1.5B_pt_monitor/v4-20260701-142211/runs --host 0.0.0.0 --port 6006"
    exit 0
fi

# ===== 环境准备 =====
source /mnt/nvme4n1/miniconda3/etc/profile.d/conda.sh
conda activate ms-swift-train

# ===== 训练配置 =====
# 模型路径
MODEL_PATH="/mnt/nvme4n1/dpsk-v4-train/train/deepseekv4_train/megatron_output/dpsk-v4-4B-A1.5B_pt_monitor/v8-20260707-115421/checkpoint-10000"

# 数据集路径（#1000 表示限制 1000 个样本，用于测试）
DATASET_PATH="/mnt/nvme4n1/dpsk-v4-train/data/recovery_cpt_v4/seq4k/recovery_cpt_seq4k.jsonl"

# 输出目录
OUTPUT_DIR="/mnt/nvme4n1/dpsk-v4-train/train/deepseekv4_train/megatron_output/dpsk-v4-4B-A1.5B_pt_monitor"

# ===== 创建输出目录 =====
mkdir -p "${OUTPUT_DIR}"

# ===== 启动训练 =====
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
megatron pt \
    --model "${MODEL_PATH}" \
    --save_safetensors true \
    --dataset "${DATASET_PATH}" \
    --dataset_num_proc 128 \
    --load_from_cache_file true \
    --max_length 4096 \
    \
    --tensor_model_parallel_size 1 \
    --expert_model_parallel_size 8 \
    \
    --micro_batch_size 2 \
    --global_batch_size 512 \
    --padding_free false \
    \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 0.002 \
    --moe_z_loss_coeff 0.001 \
    \
    --num_train_epochs 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    \
    --lr 1.8e-4 \
    --lr_warmup_fraction 0.04 \
    --min_lr 1.8e-5 \
    \
    --output_dir "${OUTPUT_DIR}" \
    --save_steps 1000 \
    --save_total_limit 10 \
    --dataloader_num_workers 8 \
    \
    --no_save_optim false \
    --no_save_rng false \
    --mtp_num_layers 0 \
    --attention_backend flash \
    --group_by_length true \
    --truncation_strategy split \
    \
    \
    --logging_steps 10 \
    --callbacks moe_monitor \
    --report_to tensorboard \

# ===== 训练完成 =====
echo ""
echo "=========================================="
echo "  训练完成！"
echo "=========================================="
echo "模型保存路径: ${OUTPUT_DIR}"
echo ""
echo "查看 TensorBoard:"
echo "  tensorboard --logdir ${OUTPUT_DIR}"
echo ""
echo "关键监控指标："
echo "  - loss: 训练 loss 曲线"
echo "  - moe/router_entropy: 路由熵（目标: > 0.7）"
echo "  - moe/expert_utilization: 专家利用率（目标: > 80%）"
echo "  - moe/load_balance_loss: 负载均衡损失（应逐步下降）"
echo "  - grad_norm: 梯度范数（突然飙升需排查）"
echo ""

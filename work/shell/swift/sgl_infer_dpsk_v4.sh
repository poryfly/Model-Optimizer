#!/bin/bash
#
source /mnt/nvme4n1/miniconda3/etc/profile.d/conda.sh
conda activate sglang_0.5.14

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_DISABLE=1
export NCCL_DEBUG=INFO

export CUDA_LAUNCH_BLOCKING=1 #把 async error 变成 sync，能看到真正崩溃的 kernel
# 全参训练后 expert 权重是 BF16（非 MXFP4），必须：
# 1. 关闭 FP4 expert 路径，避免 SGLang 走错误的内存布局
# 2. 显式指定 triton MoE runner（marlin/flashinfer_mxfp4 仅用于量化权重）
export SGLANG_DSV4_FP4_EXPERTS=0  #关闭 FP4 expert 路径，告诉 SGLang "expert 权重不是 FP4"，按非 FP4（如 BF16/FP8）方式处理
export SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=0
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m sglang.launch_server \
  --host 0.0.0.0 \
  --port 9003 \
  --trust-remote-code \
  --model-path /mnt/nvme4n1/dpsk-v4-train/train/deepseekv4_train/megatron_output/dpsk-v4-4B-A1.5B_pt_monitor/v8-20260707-115421/checkpoint-10000 \
  --mem-fraction-static 0.85 \
  --disable-cuda-graph \
  --served-model-name dpsk_v4 \
  --tp 4 \
  --max-running-requests 8 \
  > /mnt/nvme4n1/dpsk-v4-train/train/deepseekv4_train/sgl_infer_dpsk_v4.log 2>&1 &

echo "SGLang 服务已在后台启动，进程号: $!"
echo "日志文件: /data/wanghui/deepseekv4_train/sgl_infer_dpsk_v4.log"
echo "查看日志: tail -f /data/wanghui/deepseekv4_train/sgl_infer_dpsk_v4.log"
#   删除 --moe-runner-backend marlin：让 SGLang 自动选择 BF16 对应的 MoE 后端（正常路径）   --moe-runner-backend triton \

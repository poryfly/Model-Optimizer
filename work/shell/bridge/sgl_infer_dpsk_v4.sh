#!/bin/bash
#
source /data/miniconda3/etc/profile.d/conda.sh
conda activate sgl-dpsk-v4-dev

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_DISABLE=1
export NCCL_DEBUG=INFO

# export CUDA_LAUNCH_BLOCKING=1 #把 async error 变成 sync，能看到真正崩溃的 kernel
# export SGLANG_OPT_FUSE_WQA_WKV=0 #模型权重里只有 wq_a.weight_scale_inv，没有 wqkv_a.weight_scale_inv，SGLang 融合后去 params_dict 里找，找不到，就报 KeyError。

# 全参训练后 expert 权重是 BF16（非 MXFP4），必须：
# 1. 关闭 FP4 expert 路径，避免 SGLang 走错误的内存布局
# 2. 显式指定 triton MoE runner（marlin/flashinfer_mxfp4 仅用于量化权重）
# export SGLANG_DSV4_FP4_EXPERTS=0  # 关闭 FP4 expert 路径
export SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=0
export SGLANG_SM120_TRITON_FLASHMLA=0  # 使用 PyTorch fallback，避免 Triton 内核的硬编码 head_dim=512 布局
#--model-path /data2/.cache/models/deepseek-ai/dpsk-v4-4B-A1.5-headdim256 \

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m sglang.launch_server \
  --host 0.0.0.0 \
  --port 9003 \
  --model-path /data/wanghui/deepseekv4_train_mcore_bridge/model_output/phase0_recipe_callback/checkpoints/iter_0001000/hf \
  --trust-remote-code \
  --mem-fraction-static 0.85 \
  --disable-cuda-graph \
  --served-model-name dpsk_v4 \
  --moe-runner-backend marlin \
  --tp 4 \
  --max-running-requests 8 \
  > ./sgl_infer_dpsk_v4.log 2>&1 &

echo "SGLang 服务已在后台启动，进程号: $!"
echo "查看日志: tail -f /data/work/train/sgl_infer/sgl_infer_dpsk_v4.log"
#   删除 --moe-runner-backend marlin：让 SGLang 自动选择 BF16 对应的 MoE 后端（正常路径）   --moe-runner-backend triton \

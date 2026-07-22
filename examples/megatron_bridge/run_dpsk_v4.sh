#!/bin/bash
source /data/dpsk-v4/bin/activate
export CUDA_LAUNCH_BLOCKING=1
sglang serve \
  --trust-remote-code \
  --model-path /data/output/v4-pruned-nas-final-sglang \
  --tp 4 \
  --moe-runner-backend marlin \
  --disable-cuda-graph \
  --mem-fraction-static 0.75 \
  --served-model-name dpsk_v4_width \
  --host 0.0.0.0 \
  --port 30001

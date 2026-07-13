#!/usr/bin/env bash
# ==============================================================================
# DeepSeek-V4 Pruned: HF safetensors -> Megatron .distcp 格式转换 (Phase 1)
# ==============================================================================
#
# 一次性脚本: 把 HF 原始权重 (ms-swift 训的 checkpoint 或 dpsk-v4-4B-A1.5B
# 原始权重) 转成 Megatron .distcp 格式, 转换后即可被 run_pretrain.sh 加载.
#
# 与训练解耦: 转换与训练分离后, 可以独立调试转换流程, 训练阶段不再触发
# HF safetensors 加载 (避开 Bridge HF 加载路径上的 tid2eid 缺失 bug).
# ==============================================================================

set -euo pipefail

# ==============================================================================
# 路径配置 (与 run_pretrain.sh 保持一致)
# ==============================================================================

# HF safetensors 源 (ms-swift 训的 checkpoint, 也可用 dpsk-v4-4B-A1.5B 原始权重)
HF_SOURCE_FOR_IMPORT="/workdir/model_input/dpsk-v4-4B-A1.5B"
# Megatron .distcp 转换目标
DISTCP_DIR="/workdir/model_output/dpsk-v4-4B-A1.5B_distcp"
# Bridge 仓库根目录
BRIDGE_DIR="/workdir/Megatron-Bridge"
# 转换日志输出目录
LOG_DIR="/workdir/model_output"

# 并行度 (与 run_pretrain.sh 保持一致: TP=1 PP=4 EP=2, 8 GPU)
# 改用 PP=4 EP=2 是为了跟之前能跑通的 iter_0000002 一致, 避免 PP=1 + EP=8 触发的
# MoE alltoall split size mismatch 问题.
TENSOR_MODEL_PARALLEL_SIZE=1
PIPELINE_MODEL_PARALLEL_SIZE=1
EXPERT_MODEL_PARALLEL_SIZE=8
NUM_NODES=1

# 已存在 .distcp 时跳过 import (默认 true, 避免每次重转)
SKIP_IMPORT_IF_EXISTS=true

# ==============================================================================
# 派生变量
# ==============================================================================
NPROC_PER_NODE=$((TENSOR_MODEL_PARALLEL_SIZE * PIPELINE_MODEL_PARALLEL_SIZE * EXPERT_MODEL_PARALLEL_SIZE))
mkdir -p "${LOG_DIR}"

# Bridge distcp 的 marker: iter_0000000/run_config.yaml 存在则视为已转换
DISTCP_MARKER="${DISTCP_DIR}/iter_0000000/run_config.yaml"

# ==============================================================================
# Phase 0: 临时 patch HF config (num_nextn_predict_layers -> 0)
# pruned 模型 safetensors 里没有 mtp.* keys, 但 HF config 声明了
# num_nextn_predict_layers=1, 导致 conversion import 时 KeyError.
# 临时改 + 备份, 跑完 import 立即还原 (不污染原始 HF 目录).
# ==============================================================================
HF_CONFIG_FILE="${HF_SOURCE_FOR_IMPORT}/config.json"
HF_CONFIG_BAK_FILE="${HF_CONFIG_FILE}.bak_convert_to_distcp"
HF_CONFIG_PATCHED=false
if [ -f "${HF_CONFIG_FILE}" ]; then
    MTP_LAYERS=$(python3 -c "import json,sys; c=json.load(open('${HF_CONFIG_FILE}')); print(int(c.get('num_nextn_predict_layers', 0) or 0))" 2>/dev/null || echo "0")
    if [ "${MTP_LAYERS}" -gt 0 ] 2>/dev/null; then
        if [ ! -f "${HF_CONFIG_BAK_FILE}" ]; then
            cp "${HF_CONFIG_FILE}" "${HF_CONFIG_BAK_FILE}"
        fi
        python3 -c "
import json
p = '${HF_CONFIG_FILE}'
c = json.load(open(p))
c['num_nextn_predict_layers'] = 0
json.dump(c, open(p, 'w'), indent=2)
print(f'  patched num_nextn_predict_layers ${MTP_LAYERS} -> 0')
"
        HF_CONFIG_PATCHED=true
        echo "  backup: ${HF_CONFIG_BAK_FILE}"
    fi
fi

# ==============================================================================
# 主体: 跳过 / 转换
# ==============================================================================
if [ "${SKIP_IMPORT_IF_EXISTS}" = "true" ] && [ -f "${DISTCP_MARKER}" ]; then
    echo "======================================"
    echo "Phase 1: SKIP (distcp already exists at ${DISTCP_DIR})"
    echo "  marker: ${DISTCP_MARKER}"
    echo "======================================"
    # 还原 HF config (即使跳过也确保 config 是原样)
    if [ "${HF_CONFIG_PATCHED}" = "true" ] && [ -f "${HF_CONFIG_BAK_FILE}" ]; then
        mv "${HF_CONFIG_BAK_FILE}" "${HF_CONFIG_FILE}"
        echo "  restored HF config: ${HF_CONFIG_FILE}"
    fi
    # 仍然跑一次 patch, 保证已存在的 distcp 也被修复
    RUN_CONFIG="${DISTCP_DIR}/iter_0000000/run_config.yaml"
    if [ -f "${RUN_CONFIG}" ]; then
        python3 - <<PYEOF
import re
p = "${RUN_CONFIG}"
with open(p, 'r') as f:
    txt = f.read()
patterns = [
    (r'(  moe_permute_fusion: )true', r'\1false'),
    (r'(  moe_shared_expert_overlap: )true', r'\1false'),
]
changed = 0
for pat, rep in patterns:
    new_txt, n = re.subn(pat, rep, txt)
    if n > 0:
        txt = new_txt
        changed += n
if changed > 0:
    with open(p, 'w') as f:
        f.write(txt)
    print(f"  patched run_config.yaml: {changed} field(s) set to false")
else:
    print("  run_config.yaml: already patched (no true->false change needed)")
PYEOF
    fi
    exit 0
fi

echo "======================================"
echo "Phase 1: HF -> Megatron .distcp"
echo "  HF source:    ${HF_SOURCE_FOR_IMPORT}"
echo "  distcp dest:  ${DISTCP_DIR}"
echo "  parallelism:  TP=${TENSOR_MODEL_PARALLEL_SIZE} PP=${PIPELINE_MODEL_PARALLEL_SIZE} EP=${EXPERT_MODEL_PARALLEL_SIZE}"
echo "======================================"

IMPORT_LOG="${LOG_DIR}/import_$(date +%Y%m%d_%H%M%S).log"

cd "${BRIDGE_DIR}"
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
uv run --no-sync torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --nnodes="${NUM_NODES}" \
    examples/conversion/convert_checkpoints_multi_gpu.py import \
        --hf-model "${HF_SOURCE_FOR_IMPORT}" \
        --megatron-path "${DISTCP_DIR}" \
        --tp "${TENSOR_MODEL_PARALLEL_SIZE}" \
        --pp "${PIPELINE_MODEL_PARALLEL_SIZE}" \
        --ep "${EXPERT_MODEL_PARALLEL_SIZE}" \
        --etp 1 \
        --torch-dtype bfloat16 \
        --trust-remote-code \
    > "${IMPORT_LOG}" 2>&1

# 失败也还原 HF config
if [ ! -f "${DISTCP_MARKER}" ]; then
    echo "ERROR: HF -> .distcp import failed (no ${DISTCP_MARKER})"
    echo "  see log: ${IMPORT_LOG}"
    if [ "${HF_CONFIG_PATCHED}" = "true" ] && [ -f "${HF_CONFIG_BAK_FILE}" ]; then
        mv "${HF_CONFIG_BAK_FILE}" "${HF_CONFIG_FILE}"
        echo "  restored HF config (after failed import)"
    fi
    exit 1
fi

echo "Phase 1 done: ${DISTCP_MARKER}"
echo "  log: ${IMPORT_LOG}"

# ==============================================================================
# Phase 1.5: patch run_config.yaml — 禁用 MoE permute_fusion / shared_expert_overlap
# --------------------------------------------------------------------------------
# convert_checkpoints_multi_gpu.py 不接受 --moe_permute_fusion / --moe_shared_expert_overlap
# 参数, 转换产出的 distcp run_config.yaml 里这两个会是 true, 加载训练时会覆盖
# run_pretrain.sh 的 CLI 设置, 配合 EP=8 + round-robin tid2eid 触发
# "Split sizes doesn't match total dim 0 size" alltoall 错误.
# 这里直接在生成的 run_config.yaml 上把这两个强制改回 false.
# ==============================================================================
RUN_CONFIG="${DISTCP_DIR}/iter_0000000/run_config.yaml"
if [ -f "${RUN_CONFIG}" ]; then
    python3 - <<PYEOF
import re
p = "${RUN_CONFIG}"
with open(p, 'r') as f:
    txt = f.read()
# 只在 model section 下 (两空格缩进) 改, 避免误改其他子配置
patterns = [
    (r'(  moe_permute_fusion: )true', r'\1false'),
    (r'(  moe_shared_expert_overlap: )true', r'\1false'),
]
changed = 0
for pat, rep in patterns:
    new_txt, n = re.subn(pat, rep, txt)
    if n > 0:
        txt = new_txt
        changed += n
if changed > 0:
    with open(p, 'w') as f:
        f.write(txt)
    print(f"  patched run_config.yaml: {changed} field(s) set to false (moe_permute_fusion / moe_shared_expert_overlap)")
else:
    print("  run_config.yaml: no true->false patch needed (already disabled)")
PYEOF
else
    echo "WARNING: run_config.yaml not found at ${RUN_CONFIG}, skip patch"
fi

# ==============================================================================
# 还原 HF config
# ==============================================================================
if [ "${HF_CONFIG_PATCHED}" = "true" ] && [ -f "${HF_CONFIG_BAK_FILE}" ]; then
    mv "${HF_CONFIG_BAK_FILE}" "${HF_CONFIG_FILE}"
    echo "Phase 0: restored HF config -> ${HF_CONFIG_FILE}"
fi

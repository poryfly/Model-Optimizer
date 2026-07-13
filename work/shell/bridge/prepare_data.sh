#!/usr/bin/env bash
# ==============================================================================
# DeepSeek-V4 Pruned 数据预处理
#
# 将 JSONL 转换为 Megatron 预训练所需的二进制格式:
#   JSONL → Megatron indexed dataset (.bin + .idx)
#
# 用法: bash prepare_data.sh
# ==============================================================================

set -euo pipefail

# ---- 路径配置 ----
BASE_DIR="${BASE_DIR:-/workdir}"
# 输入 JSONL 文件 (每行需有 "text" 字段)
JSONL_FILE="${JSONL_FILE:-/workdir/data_org/seq4k/recovery_cpt_seq4k.jsonl}"
# 输出前缀
OUTPUT_PREFIX="${OUTPUT_PREFIX:-${BASE_DIR}/data/recovery_cpt_seq4k}"
TOKENIZER_PATH="${BASE_DIR}/model_input/dpsk-v4-4B-A1.5B"
MEGATRON_TOOLS="/workdir/Megatron-Bridge/3rdparty/Megatron-LM/tools"
# 并发数 (建议根据 CPU 核心数调整)
WORKERS="${WORKERS:-128}"

echo "=========================================="
echo "Data preprocessing"
echo "=========================================="
echo "Input JSONL: ${JSONL_FILE}"
echo "Line count: $(wc -l < "${JSONL_FILE}")"
echo "Output prefix: ${OUTPUT_PREFIX}"
echo "Workers: ${WORKERS}"

echo ""
echo "=========================================="
echo "Tokenizing JSONL → Megatron binary"
echo "=========================================="

cd "${MEGATRON_TOOLS}/.."

uv run --no-sync python tools/preprocess_data.py \
    --input "${JSONL_FILE}" \
    --json-keys text \
    --output-prefix "${OUTPUT_PREFIX}" \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model "${TOKENIZER_PATH}" \
    --append-eod \
    --workers "${WORKERS}" \
    --log-interval 1000

echo ""
echo "=========================================="
echo "Data preprocessing complete!"
echo "=========================================="
echo "Output files:"
ls -lh "${OUTPUT_PREFIX}"*
echo ""
echo "Use this prefix for training:"
echo "  dataset.blend=[[\"${OUTPUT_PREFIX}\"],null]"

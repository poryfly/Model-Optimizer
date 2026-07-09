#!/usr/bin/env bash
# ==============================================================================
# DeepSeek-V4 Pruned 数据预处理
#
# 将 alpaca CSV 转换为 Megatron 预训练所需的二进制格式:
#   1. CSV → JSONL (拼接 instruction/input/output 为 text 字段)
#   2. JSONL → Megatron indexed dataset (.bin + .idx)
#
# 用法: bash prepare_data.sh
# ==============================================================================

set -euo pipefail

# ---- 路径配置 ----
BASE_DIR="${BASE_DIR:-/workdir}"
CSV_FILE="${BASE_DIR}/alpaca-gpt4-data-zh/train.csv"
JSONL_FILE="${BASE_DIR}/data/alpaca_train.jsonl"
OUTPUT_PREFIX="${BASE_DIR}/data/alpaca_train"
TOKENIZER_PATH="${BASE_DIR}/model_input/dpsk-v4-4B-A1.5B"
MEGATRON_TOOLS="/workdir/Megatron-Bridge/3rdparty/Megatron-LM/tools"
WORKERS="${WORKERS:-8}"

mkdir -p "$(dirname "${JSONL_FILE}")"

# ==============================================================================
# Step 1: CSV → JSONL
# ==============================================================================
echo "=========================================="
echo "Step 1: Converting CSV → JSONL"
echo "=========================================="

python3 - "${CSV_FILE}" "${JSONL_FILE}" <<'PYEOF'
import csv
import json
import sys

csv_file = sys.argv[1]
jsonl_file = sys.argv[2]

count = 0
with open(csv_file, "r", encoding="utf-8") as fin, \
     open(jsonl_file, "w", encoding="utf-8") as fout:
    reader = csv.DictReader(fin)
    for row in reader:
        instruction = row.get("instruction", "").strip()
        inp = row.get("input", "").strip()
        output = row.get("output", "").strip()

        # 拼接为预训练文本格式
        if inp:
            text = f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n{output}"
        else:
            text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

        if text.strip():
            fout.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            count += 1

print(f"Converted {count} samples → {jsonl_file}")
PYEOF

echo "JSONL file: ${JSONL_FILE}"
echo "Line count: $(wc -l < "${JSONL_FILE}")"

# ==============================================================================
# Step 2: JSONL → Megatron binary (.bin + .idx)
# ==============================================================================
echo ""
echo "=========================================="
echo "Step 2: Tokenizing JSONL → Megatron binary"
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

#!/bin/bash
set -e

for HF_NAME in Nemotron-CC-Diverse-QA Nemotron-CC-High-Quality Nemotron-CC-High-Quality-Synthetic Nemotron-CC-MATH Nemotron-CC-Translated-Diverse-QA Nemotron-SFT-Code Nemotron-SFT-General Nemotron-SFT-MATH Nemotron-Synthetic-Code; do
    python -m modelopt.torch.utils.plugins.megatron_preprocess_data \
        --hf_dataset nvidia/Nemotron-Pretraining-Dataset-sample \
        --hf_name ${HF_NAME} \
        --hf_split train \
        --json_keys text \
        --tokenizer /data/.cache/models/Qwen/Qwen3-8B \
        --output_dir /data/distill_megatron_data/tokenized_qwen3 \
        --workers 180 \
        --max_sequence_length 256_000 \
        --append_eod \
        --strip_newlines
done

python -m modelopt.torch.utils.plugins.megatron_preprocess_data \
    --hf_dataset iohadrubin/wikitext-103-raw-v1 \
    --hf_split train \
    --json_keys text \
    --tokenizer /data/.cache/models/Qwen/Qwen3-8B \
    --output_dir /data/distill_megatron_data/tokenized_qwen3 \
    --workers 180 \
    --max_sequence_length 256_000 \
    --append_eod \
    --strip_newlines

# # 数据量太大
# python -m modelopt.torch.utils.plugins.megatron_preprocess_data \
#     --hf_dataset nvidia/Nemotron-Post-Training-Dataset-v1 \
#     --hf_name default \
#     --hf_split stem \
#     --hf_streaming \
#     --hf_max_samples_per_split 5_000_000 \
#     --json_keys messages \
#     --tokenizer /data/.cache/models/Qwen/Qwen3-8B \
#     --output_dir /data/distill_megatron_data/tokenized_qwen3 \
#     --workers 180 \
#     --max_sequence_length 256_000

for SPLIT in high_part00 high_part01; do
    python -m modelopt.torch.utils.plugins.megatron_preprocess_data \
        --hf_dataset nvidia/Nemotron-Math-v2 \
        --hf_split ${SPLIT} \
        --json_keys messages \
        --tokenizer /data/.cache/models/Qwen/Qwen3-8B \
        --output_dir /data/distill_megatron_data/tokenized_qwen3 \
        --workers 180 \
        --max_sequence_length 256_000 \
        --reasoning_content inline
done

hf download nvidia/Nemotron-SFT-Competitive-Programming-v2 \
    --repo-type dataset \
    --local-dir /data/distill_org_datasets/Nemotron-SFT-Competitive-Programming-v2/

for FILE in competitive_programming_python_00 competitive_programming_cpp_00; do
    python -m modelopt.torch.utils.plugins.megatron_preprocess_data \
        --jsonl_paths /data/distill_org_datasets/Nemotron-SFT-Competitive-Programming-v2/data/${FILE}.jsonl \
        --json_keys messages \
        --tokenizer /data/.cache/models/Qwen/Qwen3-8B \
        --output_dir /data/distill_megatron_data/tokenized_qwen3 \
        --workers 180 \
        --max_sequence_length 256_000 \
        --reasoning_content inline
done

hf download nvidia/Nemotron-Science-v1 \
    --repo-type dataset \
    --local-dir /data/distill_org_datasets/Nemotron-Science-v1/

python -m modelopt.torch.utils.plugins.megatron_preprocess_data \
    --input_dir /data/distill_org_datasets/Nemotron-Science-v1/data/ \
    --json_keys messages \
    --tokenizer /data/.cache/models/Qwen/Qwen3-8B \
    --output_dir /data/distill_megatron_data/tokenized_qwen3 \
    --workers 180 \
    --max_sequence_length 256_000 \
    --reasoning_content inline

hf download nvidia/Nemotron-SFT-Instruction-Following-Chat-v2 \
    --repo-type dataset \
    --local-dir /data/distill_org_datasets/Nemotron-SFT-Instruction-Following-Chat-v2/

python -m modelopt.torch.utils.plugins.megatron_preprocess_data \
    --input_dir /data/distill_org_datasets/Nemotron-SFT-Instruction-Following-Chat-v2/data/ \
    --json_keys messages \
    --tokenizer /data/.cache/models/Qwen/Qwen3-8B \
    --output_dir /data/distill_megatron_data/tokenized_qwen3 \
    --workers 180 \
    --max_sequence_length 256_000 \
    --reasoning_content inline
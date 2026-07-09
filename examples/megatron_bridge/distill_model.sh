#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun --nnodes 1 --nproc_per_node 4 distill.py \
    --tp_size 8 \
    --teacher_hf_path /data/.cache/models/Qwen/Qwen3-8B \
    --student_hf_path /data/.cache/models/Qwen/Qwen3-8B-Pruned-6B \
    --data_paths 1.0 tokenized_qwen3/data1_text_document 1.0 tokenized_qwen3/data2_text_document \
    --data_path_to_cache /data/dataset_indices_qwen3 \
    --seq_length 8192 \
    --mbs 1 \
    --gbs 768 \
    --train_iters 15000 \
    --lr 1e-4 \
    --min_lr 1e-5 \
    --lr_warmup_iters 50 \
    --eval_interval 100 \
    --eval_iters 32 \
    --log_interval 10 \
    --output_dir /output/qwen3_8b_to_4b_distill
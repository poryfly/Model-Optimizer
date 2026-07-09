#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Instruction fine-tuning (SFT) for a pruned DeepSeek-V4-4B-A1B model.

Expects instruction-following datasets in ``messages`` format and applies the
model's chat template.  Supports multi-dataset mixing and packing.

Example:
    torchrun --nproc_per_node=8 sft_deepseek_v4_4b.py \
        --model_name_or_path /data/output/v4-4b-a1b-cpt \
        --dataset_name OpenHermes-2.5 \
        --output_dir /data/output/v4-4b-a1b-sft \
        --num_train_epochs 3 \
        --per_device_train_batch_size 4 \
        --learning_rate 2e-5 \
        --max_seq_length 4096
"""

import argparse
import json
import os
from dataclasses import dataclass, field

import datasets
import torch
import transformers
from accelerate.logging import get_logger
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

logger = get_logger(__name__, log_level="INFO")


@dataclass
class SFTArguments:
    model_name_or_path: str = field(metadata={"help": "Path to CPT/Distilled HF model"})
    dataset_name: str | None = field(default=None)
    dataset_config: str | None = field(default=None)
    data_mix_config: str | None = field(
        default=None,
        metadata={"help": "JSON file listing datasets and sampling weights"},
    )
    max_seq_length: int = field(default=4096)
    aux_loss_coef: float = field(default=0.001)


def parse_args() -> tuple[TrainingArguments, SFTArguments]:
    parser = transformers.HfArgumentParser((TrainingArguments, SFTArguments))
    return parser.parse_args_into_dataclasses()


def normalize_messages_dataset(dset: datasets.Dataset) -> datasets.Dataset:
    """Convert common SFT formats into {'messages': [...]} format."""
    if "messages" in dset.column_names:
        return dset

    def _to_messages(example):
        messages = []
        if "instruction" in example and "response" in example:
            messages = [
                {"role": "user", "content": example["instruction"]},
                {"role": "assistant", "content": example["response"]},
            ]
            if "input" in example and example["input"]:
                messages[0]["content"] += "\n" + example["input"]
        elif "query" in example and "answer" in example:
            messages = [
                {"role": "user", "content": example["query"]},
                {"role": "assistant", "content": example["answer"]},
            ]
        elif "conversations" in example:
            messages = example["conversations"]
        return {"messages": messages}

    dset = dset.map(_to_messages, remove_columns=dset.column_names)
    return dset


def load_dataset_single(sft_args: SFTArguments) -> datasets.Dataset:
    logger.info(f"Loading dataset {sft_args.dataset_name}...")
    kwargs = {}
    if sft_args.dataset_config:
        kwargs["name"] = sft_args.dataset_config
    try:
        dset = datasets.load_dataset(sft_args.dataset_name, split="train", **kwargs)
    except Exception as e:
        logger.warning(f"Could not load from HF hub: {e}. Trying local path...")
        dset = datasets.load_from_disk(sft_args.dataset_name)
    return normalize_messages_dataset(dset)


def load_dataset_mix(sft_args: SFTArguments) -> datasets.Dataset:
    with open(sft_args.data_mix_config) as f:
        mix = json.load(f)

    datasets_list = []
    weights = []
    for item in mix["datasets"]:
        name = item["name"]
        weight = item["weight"]
        config = item.get("config")
        logger.info(f"Loading mixed dataset: {name} (weight={weight})")
        kwargs = {}
        if config:
            kwargs["name"] = config
        try:
            dset = datasets.load_dataset(name, split="train", **kwargs)
        except Exception:
            dset = datasets.load_from_disk(name)
        dset = normalize_messages_dataset(dset)
        datasets_list.append(dset)
        weights.append(weight)

    return datasets.interleave_datasets(datasets_list, probabilities=weights, seed=42)


def apply_chat_template_and_tokenize(
    dset: datasets.Dataset, tokenizer: AutoTokenizer, max_seq_length: int
) -> datasets.Dataset:
    def _process(examples):
        texts = []
        for messages in examples["messages"]:
            if isinstance(messages, list):
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            else:
                text = str(messages)
            texts.append(text)
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    dset = dset.map(_process, batched=True, remove_columns=dset.column_names, num_proc=8)
    return dset


class MoESFTTrainer(Trainer):
    """Trainer that adds MoE auxiliary loss if present in model outputs."""

    def __init__(self, aux_loss_coef: float = 0.001, **kwargs):
        super().__init__(**kwargs)
        self.aux_loss_coef = aux_loss_coef

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        if hasattr(outputs, "aux_loss") and outputs.aux_loss is not None:
            loss = loss + self.aux_loss_coef * outputs.aux_loss
        return (loss, outputs) if return_outputs else loss


def main() -> None:
    training_args, sft_args = parse_args()

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        sft_args.model_name_or_path, use_fast=True, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        sft_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()

    logger.info("Loading dataset...")
    if sft_args.data_mix_config:
        train_dset = load_dataset_mix(sft_args)
    elif sft_args.dataset_name:
        train_dset = load_dataset_single(sft_args)
    else:
        raise ValueError("Either --dataset_name or --data_mix_config must be provided.")

    train_dset = apply_chat_template_and_tokenize(
        train_dset, tokenizer, sft_args.max_seq_length
    )

    trainer = MoESFTTrainer(
        aux_loss_coef=sft_args.aux_loss_coef,
        model=model,
        args=training_args,
        train_dataset=train_dset,
        tokenizer=tokenizer,
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    logger.info("SFT complete.")


if __name__ == "__main__":
    main()

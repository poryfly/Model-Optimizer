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

"""Continual pre-training for a pruned DeepSeek-V4-4B-A1B model.

Loads a HuggingFace-format MoE model (e.g. output of run_v4_4b_pipeline.sh) and
continues pre-training on a mixed corpus.  The trainer preserves the MoE
auxiliary load-balancing loss if the model exposes it.

Supports:
  - Single dataset or multi-dataset mixing via ``--data_mix_config``
  - Curriculum learning (optional, via dataset ordering)
  - Gradient checkpointing and FSDP2 / DeepSpeed

Example:
    torchrun --nproc_per_node=8 cpt_deepseek_v4_4b.py \
        --model_name_or_path /data/output/v4-4b-a1b-final-sglang \
        --dataset_name nvidia/Nemotron-Post-Training-Dataset-v2 \
        --output_dir /data/output/v4-4b-a1b-cpt \
        --num_train_tokens 300_000_000_000 \
        --per_device_train_batch_size 4 \
        --learning_rate 2e-4 \
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
class CPTArguments:
    model_name_or_path: str = field(metadata={"help": "Path to pruned HF model"})
    dataset_name: str | None = field(default=None)
    dataset_config: str | None = field(default=None)
    text_column: str = field(default="text")
    data_mix_config: str | None = field(
        default=None,
        metadata={"help": "JSON file listing datasets and sampling weights"},
    )
    num_train_tokens: int = field(default=200_000_000_000)
    max_seq_length: int = field(default=4096)
    aux_loss_coef: float = field(
        default=0.001,
        metadata={"help": "Coefficient for MoE load-balancing auxiliary loss."},
    )


def parse_args() -> tuple[TrainingArguments, CPTArguments]:
    parser = transformers.HfArgumentParser((TrainingArguments, CPTArguments))
    return parser.parse_args_into_dataclasses()


def load_single_dataset(cpt_args: CPTArguments, tokenizer: AutoTokenizer) -> datasets.Dataset:
    logger.info(f"Loading dataset {cpt_args.dataset_name}...")
    kwargs = {}
    if cpt_args.dataset_config:
        kwargs["name"] = cpt_args.dataset_config
    try:
        dset = datasets.load_dataset(cpt_args.dataset_name, split="train", **kwargs)
    except Exception as e:
        logger.warning(f"Could not load from HF hub: {e}. Trying local path...")
        dset = datasets.load_from_disk(cpt_args.dataset_name)

    return normalize_text_dataset(dset, cpt_args.text_column)


def load_mixed_datasets(cpt_args: CPTArguments, tokenizer: AutoTokenizer) -> datasets.Dataset:
    with open(cpt_args.data_mix_config) as f:
        mix = json.load(f)

    datasets_list = []
    weights = []
    for item in mix["datasets"]:
        name = item["name"]
        weight = item["weight"]
        config = item.get("config")
        text_col = item.get("text_column", "text")
        logger.info(f"Loading mixed dataset: {name} (weight={weight})")
        kwargs = {}
        if config:
            kwargs["name"] = config
        try:
            dset = datasets.load_dataset(name, split="train", **kwargs)
        except Exception:
            dset = datasets.load_from_disk(name)
        dset = normalize_text_dataset(dset, text_col)
        datasets_list.append(dset)
        weights.append(weight)

    # Interleave by weight
    mixed = datasets.interleave_datasets(datasets_list, probabilities=weights, seed=42)
    return mixed


def normalize_text_dataset(dset: datasets.Dataset, text_column: str) -> datasets.Dataset:
    if text_column not in dset.column_names:
        if "messages" in dset.column_names:
            def _messages_to_text(example):
                text = "\n".join(
                    str(m.get("content", "")) for m in example["messages"] if isinstance(m, dict)
                )
                return {"text": text}
            dset = dset.map(_messages_to_text, remove_columns=dset.column_names)
        else:
            raise ValueError(
                f"Column {text_column} not found. Available: {dset.column_names}"
            )
    return dset


def tokenize_and_group(dset: datasets.Dataset, tokenizer: AutoTokenizer, max_seq_length: int) -> datasets.Dataset:
    def _tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
        )

    dset = dset.map(_tokenize, batched=True, remove_columns=dset.column_names, num_proc=8)

    def _group(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = (len(concatenated["input_ids"]) // max_seq_length) * max_seq_length
        result = {
            k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    dset = dset.map(_group, batched=True, num_proc=8)
    return dset


class MoECPTTrainer(Trainer):
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
    training_args, cpt_args = parse_args()

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        cpt_args.model_name_or_path, use_fast=True, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        cpt_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()

    logger.info("Loading dataset...")
    if cpt_args.data_mix_config:
        train_dset = load_mixed_datasets(cpt_args, tokenizer)
    elif cpt_args.dataset_name:
        train_dset = load_single_dataset(cpt_args, tokenizer)
    else:
        raise ValueError("Either --dataset_name or --data_mix_config must be provided.")

    train_dset = tokenize_and_group(train_dset, tokenizer, cpt_args.max_seq_length)

    num_examples = len(train_dset)
    tokens_per_example = cpt_args.max_seq_length
    num_epochs = max(1, int(cpt_args.num_train_tokens / (num_examples * tokens_per_example)))
    logger.info(
        f"Dataset: {num_examples} examples, ~{num_examples * tokens_per_example / 1e9:.1f}B tokens. "
        f"Training for {num_epochs} epochs to reach ~{cpt_args.num_train_tokens / 1e9:.1f}B tokens."
    )

    trainer = MoECPTTrainer(
        aux_loss_coef=cpt_args.aux_loss_coef,
        model=model,
        args=training_args,
        train_dataset=train_dset,
        tokenizer=tokenizer,
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    logger.info("CPT complete.")


if __name__ == "__main__":
    main()

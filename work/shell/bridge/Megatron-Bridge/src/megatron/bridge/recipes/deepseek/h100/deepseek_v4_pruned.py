# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DeepSeek-V4 Pruned (dpsk-v4-4B-A1.5B) pre-training recipes.

These recipes mirror the official DeepSeek-V4-Flash recipes but target a
locally-pruned 4B / 64-expert variant on a single 8-GPU node.

Key differences from the official Flash recipes:
  - Model path points to a local HF directory (dpsk-v4-4B-A1.5B)
  - Parallelism defaults: TP=1, PP=1, EP=8 (fits 8 GPUs)
  - seq_length defaults to 1024 (overridable via CLI)
  - MTP disabled (mtp_num_layers=None)
  - Recompute: full / uniform (memory-constrained single node)
  - Optimizer: distributed fused Adam, LR=1.8e-4

All dynamic training parameters (train_iters, batch_size, lr, checkpoint
paths, dataset, etc.) are overridable via CLI key=value overrides through
run_recipe.py, exactly like the official flow.
"""

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.models.deepseek.deepseek_v4_bridge import (
    deepseek_v4_supports_blackwell_fused_kernels,
    set_deepseek_v4_pipeline_model_parallel_layout,
)
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.recipes.utils.optimizer_utils import (
    distributed_fused_adam_with_cosine_annealing,
)
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer

# Local pruned model path (HF format: config.json + safetensors)
DPSK_V4_PRUNED_HF_PATH = "/workdir/model_input/checkpoint-10000"


def deepseek_v4_pruned_pretrain_8gpu_bf16_config() -> ConfigContainer:
    """Return the DeepSeek-V4 Pruned pre-training config for a single 8-GPU node.

    Recommended baseline: TP=1, PP=1, EP=8, CP=1.
    Model weights are loaded from the local HF directory via
    ``checkpoint.pretrained_checkpoint`` (set via CLI).
    """
    use_fused_mhc = deepseek_v4_supports_blackwell_fused_kernels()
    cfg = _pretrain_common()

    # ---- Model (local HF config, no weights loaded here) ----
    cfg.model = AutoBridge.from_hf_pretrained(
        DPSK_V4_PRUNED_HF_PATH, trust_remote_code=True
    ).to_megatron_provider(load_weights=False)

    # ---- Parallelism (8 GPUs: TP=1 * PP=1 * EP=8) ----
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 8
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.seq_length = 1024
    cfg.model.params_dtype = torch.bfloat16

    cfg.model.account_for_embedding_in_pipeline_split = False
    cfg.model.account_for_loss_in_pipeline_split = False
    cfg.model.num_layers_in_first_pipeline_stage = None
    cfg.model.num_layers_in_last_pipeline_stage = None
    set_deepseek_v4_pipeline_model_parallel_layout(cfg.model)

    # ---- MTP disabled ----
    cfg.model.mtp_num_layers = None
    cfg.model.mtp_loss_scaling_factor = 0.0
    ratios = getattr(cfg.model, "csa_compress_ratios", None)
    num_layers = getattr(cfg.model, "num_layers", None)
    if ratios is not None and num_layers is not None and len(ratios) > num_layers:
        cfg.model.csa_compress_ratios = list(ratios)[:num_layers]

    # ---- Kernel / Attention ----
    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.attention_backend = None
    cfg.model.apply_dsa_kernel_fusion = False
    cfg.model.apply_rope_fusion = True
    cfg.model.use_fused_mhc = use_fused_mhc
    cfg.model.dsa_indexer_loss_coeff = 0.0
    cfg.model.dsa_indexer_use_sparse_loss = False

    # ---- MoE ----
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_aux_loss_coeff = 0.0
    cfg.model.moe_z_loss_coeff = 0.0
    cfg.model.moe_router_force_load_balancing = False
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"

    # ---- Recompute (memory-constrained: full recompute) ----
    cfg.model.recompute_granularity = "full"
    cfg.model.recompute_method = "uniform"
    cfg.model.recompute_num_layers = 1
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # ---- Tokenizer (use model's own HF tokenizer) ----
    cfg.tokenizer.tokenizer_type = "HuggingFaceTokenizer"
    cfg.tokenizer.tokenizer_model = DPSK_V4_PRUNED_HF_PATH
    cfg.tokenizer.vocab_size = cfg.model.vocab_size
    cfg.tokenizer.make_vocab_size_divisible_by = cfg.model.make_vocab_size_divisible_by
    cfg.tokenizer.tensor_model_parallel_size = cfg.model.tensor_model_parallel_size
    cfg.tokenizer.rank = 0

    # ---- Dataset (default: mock; override via CLI dataset.blend=...) ----
    cfg.dataset.blend = None
    cfg.dataset.blend_per_split = None
    cfg.dataset.seq_length = 1024
    cfg.dataset.num_workers = 4
    cfg.dataset.skip_getting_attention_mask_from_dataset = True
    cfg.dataset.dataloader_type = "single"

    # ---- Training defaults (overridable via CLI) ----
    cfg.train.train_iters = 1000
    cfg.train.global_batch_size = 32
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 5
    cfg.train.manual_gc_eval = 5
    cfg.validation.eval_interval = 0
    cfg.validation.eval_iters = 0

    # ---- Optimizer / Scheduler ----
    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=40,
        lr_decay_iters=cfg.train.train_iters,
        max_lr=1.8e-4,
        min_lr=1.8e-5,
        weight_decay=0.1,
        clip_grad=1.0,
    )
    opt_cfg.adam_beta1 = 0.9
    opt_cfg.adam_beta2 = 0.95
    scheduler_cfg.lr_decay_style = "cosine"
    cfg.optimizer = opt_cfg
    cfg.scheduler = scheduler_cfg

    # ---- Logger ----
    cfg.logger.log_interval = 1

    # ---- Checkpoint defaults (overridable via CLI) ----
    cfg.checkpoint.save_interval = 5
    cfg.checkpoint.async_save = False
    cfg.checkpoint.most_recent_k = 2
    cfg.checkpoint.save_optim = True
    cfg.checkpoint.save_rng = True
    cfg.checkpoint.load_optim = False
    cfg.checkpoint.load_rng = False
    cfg.checkpoint.finetune = True

    # ---- Distributed / DDP ----
    cfg.dist.enable_megatron_core_experimental = True

    # ---- Comm overlap ----
    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
    cfg.comm_overlap.delay_wgrad_compute = False
    cfg.comm_overlap.overlap_moe_expert_parallel_comm = False

    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_megatron_fsdp = False
    return cfg


__all__ = [
    "deepseek_v4_pruned_pretrain_8gpu_bf16_config",
]

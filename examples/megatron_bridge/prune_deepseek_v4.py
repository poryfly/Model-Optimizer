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

"""DeepSeek-V4 structured pruning via ModelOpt mcore_minitron.

This script prunes a DeepSeek-V4-Flash model using the mcore_minitron algorithm
with pipeline-parallelism (PP) support for multi-GPU model loading.

V4-specific pruning constraints:

  - **hidden_size**: Width pruning of MLA input projections and all MoE layers.
  - **num_layers**: Depth pruning maintains CSA/HCA alternation pattern.
    The first 2 sliding-window layers are protected (never dropped).
    Remaining CSA (ratio=4) and HCA (ratio=128) layers are pruned in pairs
    to preserve the ``compress_ratios`` alternation invariant.
  - **num_moe_experts**: Expert pruning. Hash routing layers (first 3 MoE layers)
    have their ``tid2eid`` buffers remapped to the retained expert IDs.
  - **moe_ffn_hidden_size**: Width pruning of routed expert FFN layers.
  - **moe_shared_expert_intermediate_size**: Width pruning of shared expert FFN.
  - **head_dim**: Post-prune per-head slicing target for MLA (applied by
    ``convert_pruned_to_hf.py``). The Megatron-side prune keeps the
    original head_dim; only the exported HF checkpoint has reduced head
    dims. Useful because head_dim=512 is disproportionately wide for a
    4B-scale model with hidden_size=2048.

NOT pruned (by design):
  - ``q_lora_rank``, ``kv_lora_rank``, ``o_lora_rank`` (MLA LoRA dimensions)
  - ``compress_ratios`` (CSA/HCA compression ratios)
  - ``num_residual_streams`` (Hyper-Connection stream count)

Usage (8× GPU with PP=8):
    torchrun --nproc_per_node=8 prune_deepseek_v4.py \\
        --hf_model_name_or_path /data/models/deepseek-ai/DeepSeek-V4-Flash \\
        --output_hf_path /output/pruned-v4 \\
        --pp_size 8 \\
        --hidden_size 2048 \\
        --num_layers 35 \\
        --num_moe_experts 128 \\
        --moe_ffn_hidden_size 1024 \\
        --head_dim 128 \\
        --trust_remote_code
"""

import argparse
import json
import os

import torch
from megatron.bridge import AutoBridge
from safetensors.torch import save_file
from transformers import AutoConfig

import modelopt.torch.opt as mto
import modelopt.torch.prune as mtp
import modelopt.torch.utils.distributed as dist
try:
    from modelopt.torch.export import copy_hf_ckpt_remote_code
except ImportError:
    def copy_hf_ckpt_remote_code(src, dst):
        pass
try:
    from modelopt.torch.utils import print_args, print_rank_0
except ImportError:
    import torch.distributed as _td

    def print_rank_0(*a, **kw):
        if not _td.is_initialized() or _td.get_rank() == 0:
            print(*a, **kw)

    def print_args(args):
        for k, v in sorted(vars(args).items()):
            print_rank_0(f"  {k}: {v}")
from modelopt.torch.utils.plugins.megatron_calibration import get_megatron_calibration_forward_loop


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prune DeepSeek-V4 model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--hf_model_name_or_path",
        type=str,
        required=True,
        help="Path to DeepSeek-V4-Flash HF checkpoint",
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument(
        "--output_hf_path",
        type=str,
        required=True,
        help="Output path for pruned HF model",
    )

    parser.add_argument("--pp_size", type=int, default=1, help="Pipeline parallelism size")

    parser.add_argument("--hidden_size", type=int, default=None, help="Target hidden_size")
    parser.add_argument("--num_layers", type=int, default=None, help="Target num_layers")
    parser.add_argument("--num_moe_experts", type=int, default=None, help="Target num_moe_experts")
    parser.add_argument(
        "--moe_ffn_hidden_size", type=int, default=None, help="Target moe_ffn_hidden_size"
    )
    parser.add_argument(
        "--moe_shared_expert_intermediate_size",
        type=int,
        default=None,
        help="Target shared expert width",
    )
    parser.add_argument(
        "--head_dim",
        type=int,
        default=None,
        help=(
            "Target MLA per-head dimension. DeepSeek-V4-Flash ships with "
            "head_dim=512; for aggressive (4B-scale) pruning this is "
            "disproportionately wide relative to hidden_size=2048. Setting "
            "e.g. 128 or 256 reshapes wq_b / wkv / wo_a / compressor weights "
            "per-head during convert_pruned_to_hf.py. The Megatron-side "
            "prune keeps the original head_dim; only the exported HF model "
            "sees the reduced value. Must be a power of 2 and >= qk_rope_head_dim."
        ),
    )

    parser.add_argument(
        "--calib_dataset_name",
        type=str,
        default="nvidia/Nemotron-Post-Training-Dataset-v2",
        help="Calibration dataset name",
    )
    parser.add_argument(
        "--calibration_samples", type=int, default=128, help="Number of calibration samples"
    )
    parser.add_argument("--seq_length", type=int, default=2048, help="Calibration seq length")
    parser.add_argument(
        "--validation_samples",
        type=int,
        default=0,
        help=(
            "Number of held-out samples for pruned-model validation loss. "
            "Used as the ranking signal for NAS. Default 0 — the forward "
            "pass over the pruned model under PP=8 / TP=1 + mHC + MTP "
            "is currently unstable (rank-7 forward can hang on lm_head / "
            "MTP), so we skip the validation forward and rely on the "
            "heuristic quality score instead. Set to e.g. 32 to "
            "re-enable the slow path."
        ),
    )

    args = parser.parse_args()

    targets = [
        args.hidden_size,
        args.num_layers,
        args.num_moe_experts,
        args.moe_ffn_hidden_size,
        args.moe_shared_expert_intermediate_size,
    ]
    if not any(t is not None for t in targets):
        parser.error("At least one pruning target must be specified.")

    print_args(args)
    return args


def build_export_config(args):
    export_config = {}
    if args.hidden_size is not None:
        export_config["hidden_size"] = args.hidden_size
    if args.num_layers is not None:
        export_config["num_layers"] = args.num_layers
    if args.num_moe_experts is not None:
        export_config["num_moe_experts"] = args.num_moe_experts
    if args.moe_ffn_hidden_size is not None:
        export_config["moe_ffn_hidden_size"] = args.moe_ffn_hidden_size
    if args.moe_shared_expert_intermediate_size is not None:
        export_config["moe_shared_expert_intermediate_size"] = (
            args.moe_shared_expert_intermediate_size
        )
    return export_config


def _build_pp_layout(n_layers, pp, n_mtp=0):
    """Build a balanced pipeline-model-parallel layout string for V4.

    Returns the layout string of length ``pp``, separated by ``|``, where each
    stage has ``E`` (embed) on stage 0, ``L`` (LM head) on stage pp-1, ``m`` for
    each MTP layer on stage pp-1, and ``t`` for each transformer layer.

    With ``n_layers`` total transformer layers split across ``pp`` stages,
    the first ``n_layers % pp`` stages get one extra layer (standard Megatron
    balance). Use this for both the initial build (with native num_layers) and
    after pruning (with the new, smaller num_layers).
    """
    base = n_layers // pp
    extra = n_layers % pp
    stages = []
    for i in range(pp):
        n = base + (1 if i < extra else 0)
        if i == 0:
            stages.append("E" + "t" * n)
        elif i == pp - 1:
            stages.append("t" * n + "m" * n_mtp + "L")
        else:
            stages.append("t" * n)
    return "|".join(stages)


def _rebuild_pp_layout_after_prune(unwrapped_model, pp_size, n_mtp=0, n_layers=None):
    """Recompute and patch the PP layout on the model after mcore_minitron prune.

    mcore_minitron drops transformer layers in-place but does NOT rewrite the
    pipeline-model-parallel layout string. As a result, after pruning:

    * The model's per-rank ``decoder.layers`` list may have a different length
      than the original layout said it should.
    * The model-level ``pipeline_model_parallel_layout`` string still reflects
      the pre-prune layer count, so downstream consumers (and any second-pass
      forward like validation) hit ``IndexError: list index out of range``
      inside the pipeline schedule when a tail rank ends up with 0 layers.

    This function updates the *layout metadata* on the unwrapped model so that
    the new, post-prune layer counts per rank match the new layout. It does
    NOT rebuild the model — the model structure is already correct after
    mcore_minitron (it correctly redistributes layers across ranks). We just
    need to keep the layout metadata in sync.

    Args:
        unwrapped_model: The model chunk for this PP rank (the value returned
            by ``unwrap_model(model[0])``).
        pp_size: Pipeline-parallel world size.
        n_mtp: Number of MTP layers (added to the last PP stage).
        n_layers: **Total** post-prune num_layers across all PP ranks. Must
            be passed explicitly — ``len(unwrapped_model.decoder.layers)`` is
            the *per-rank* count (e.g. 5 when total is 35 / PP=8), and using
            the per-rank number here produces a layout with empty middle
            stages and the pipeline schedule returns an empty loss list. If
            ``None``, falls back to the per-rank count (legacy behaviour; do
            not rely on it for PP>1 + a post-prune forward pass).

    Returns the new layout string for logging.
    """
    if pp_size <= 1:
        return ""
    if n_layers is None:
        # Legacy fallback: per-rank count. Only correct if there is exactly
        # one PP rank (i.e. pp_size == 1, which we already returned above) or
        # if no second-pass forward (e.g. direct convert) will use the layout.
        n_layers = len(unwrapped_model.decoder.layers)
    new_layout = _build_pp_layout(n_layers, pp_size, n_mtp=n_mtp)

    # Megatron's transformer config turns the layout string into a
    # `PipelineParallelLayerLayout` object during config post-init. After
    # that, the forward pass calls methods on that object (e.g.
    # `pipeline_model_parallel_layout.get_layer_offset(...)`). Writing the
    # raw string back would crash with
    # `'str' object has no attribute 'get_layer_offset'`. So if the existing
    # value is a `PipelineParallelLayerLayout`, build a fresh one from the
    # new string; otherwise just set the string.
    def _wrap(layout_str):
        try:
            from megatron.core.transformer.pipeline_parallel_layer_layout import (
                PipelineParallelLayerLayout,
            )
            return PipelineParallelLayerLayout.from_str(layout_str, pp_size)
        except Exception:
            return layout_str

    cfg = getattr(unwrapped_model, "config", None)
    if cfg is not None and hasattr(cfg, "pipeline_model_parallel_layout"):
        try:
            cfg.pipeline_model_parallel_layout = _wrap(new_layout)
        except (AttributeError, TypeError):
            pass

    # Megatron pipeline schedule also reads these on the model itself.
    for attr in ("pipeline_model_parallel_layout", "_pp_layout"):
        if hasattr(unwrapped_model, attr):
            try:
                setattr(unwrapped_model, attr, _wrap(new_layout))
            except (AttributeError, TypeError):
                pass

    return new_layout


def _compute_pruned_validation_loss(
    unwrapped_model,
    tokenizer,
    calib_texts,
    seq_length,
    num_samples=32,
    model_chunks=None,
):
    """Compute mean cross-entropy validation loss on the (already-pruned) model.

    Uses a small held-out subset of ``calib_texts`` (skipping the first ~80%
    used for importance estimation). The forward pass runs under the same PP
    schedule as importance estimation; only the loss function changes.

    Returns ``(mean_loss, num_samples_evaluated)``. On failure returns
    ``(nan, 0)`` so callers can fall back to heuristic.
    """
    import torch.distributed as _td
    from megatron.core import parallel_state
    from megatron.core.pipeline_parallel import get_forward_backward_func

    # Use the latter half of calib_texts as held-out validation set; the first
    # half is what forward_loop consumed during importance estimation.
    if calib_texts is None or len(calib_texts) == 0:
        return float("nan"), 0
    n_total = len(calib_texts)
    held_out_start = max(0, int(n_total * 0.8))
    held_out = calib_texts[held_out_start:]

    # Pick `model_chunks` if provided (PP wants a list of chunks); otherwise
    # wrap the unwrapped model in a list so the pipeline schedule is happy.
    model = model_chunks if model_chunks is not None else [unwrapped_model]

    losses = []
    n_used = 0
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # Cache broadcast target once: the last PP stage is fixed across all
    # calls in this validation pass.
    pp_group = parallel_state.get_pipeline_model_parallel_group()
    pp_last_rank_global = parallel_state.get_pipeline_model_parallel_last_rank()

    def _forward_step(data, model):
        tokens = data["tokens"]

        def _loss_func(output_tensor, non_loss_data=False):
            logits = output_tensor[0] if isinstance(output_tensor, tuple) else output_tensor
            shift_logits = logits[..., :-1, :].contiguous().float()
            shift_labels = tokens[..., 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="mean",
                ignore_index=pad_id,
            )
            if dist.is_initialized():
                loss = loss / dist.size()
            # Schedule contract: return `(output_tensor, loss_reduced)`. Only
            # the last PP stage calls this; on other stages
            # `forward_data_store` stays empty.
            return output_tensor, loss

        b, s = tokens.shape
        device = tokens.device
        mask = torch.triu(torch.ones(b, s, s, device=device), 1).bool().view(b, 1, s, s)
        out = model(tokens, position_ids=None, attention_mask=mask)
        return out, _loss_func

    unwrapped_model.eval()
    with torch.no_grad():
        for i, text in enumerate(held_out):
            if n_used >= num_samples:
                break
            tokens = tokenizer.encode(text)
            if len(tokens) < 10:
                continue
            tokens = tokens[:seq_length]
            tokens_t = torch.tensor([tokens], dtype=torch.long, device="cuda")
            try:
                # Use the loss-returning path of the pipeline schedule.
                # On the last PP stage, the schedule returns
                # `[loss_tensor]`; on other stages it returns `[]` (the
                # loss function is only called on the last stage). For the
                # other stages we broadcast the loss from the last stage so
                # every rank sees the same scalar.
                schedule_output = get_forward_backward_func()(
                    forward_step_func=_forward_step,
                    data_iterator=[{"tokens": tokens_t}],
                    model=model,
                    num_microbatches=1,
                    seq_length=tokens_t.shape[-1],
                    micro_batch_size=1,
                    decoder_seq_length=tokens_t.shape[-1],
                    forward_only=True,
                    collect_non_loss_data=False,
                )

                # Some configs wrap the list in a tuple; unwrap one level.
                if isinstance(schedule_output, tuple) and len(schedule_output) == 1:
                    schedule_output = schedule_output[0]
                if not isinstance(schedule_output, list):
                    continue

                # Cross-rank loss gathering. The schedule only invokes the
                # loss function on the last PP stage, so on every other rank
                # `schedule_output` is `[]`. We need every rank to see the
                # same scalar.
                #
                # CRITICAL: `torch.distributed.broadcast` is a collective
                # op — every rank in the PP group must call it, otherwise
                # the participating ranks block forever on the NCCL
                # watchdog timeout. So we must NOT guard the broadcast on
                # `len(schedule_output) == 0`. Instead:
                #   1. every rank allocates a fixed-size buffer
                #   2. the last stage copies its local loss into the buffer
                #   3. every rank joins the broadcast (the buffer on the
                #      non-last ranks is just overwritten)
                loss_buf = torch.zeros((), dtype=torch.float32, device="cuda")
                if len(schedule_output) > 0:
                    src = schedule_output[0]
                    if hasattr(src, "detach"):
                        loss_buf = (
                            src.detach()
                            .to(device="cuda", dtype=torch.float32)
                            .reshape(())
                            .contiguous()
                        )
                    else:
                        loss_buf = torch.tensor(
                            float(src), dtype=torch.float32, device="cuda"
                        )
                # All ranks (last stage + others) call broadcast together.
                torch.distributed.broadcast(
                    loss_buf, src=pp_last_rank_global, group=pp_group
                )
                schedule_output = [loss_buf]

                for l in schedule_output:
                    if hasattr(l, "item") and l.numel() > 0:
                        losses.append(l.item())
                n_used += 1
            except Exception as _e:
                # Don't *silently* skip — that hides rank-7 forward hangs
                # and makes the run look like a hang. Log the first failure
                # on every rank (not just rank 0) so we can see which PP
                # stage actually crashed.
                import sys as _sys
                import traceback as _tb
                if not getattr(_compute_pruned_validation_loss, "_traced", False):
                    setattr(_compute_pruned_validation_loss, "_traced", True)
                    _sys.stdout.write(
                        f"[validation] rank={_td.get_rank()} pp_rank="
                        f"{parallel_state.get_pipeline_model_parallel_rank()} "
                        f"SAMPLE FAILED: {type(_e).__name__}: {_e}\n"
                        f"--- traceback (first failure) ---\n"
                        f"{_tb.format_exc()}"
                        f"--- end traceback ---\n"
                    )
                    _sys.stdout.flush()
                continue

    if not losses:
        return float("nan"), 0
    return float(sum(losses) / len(losses)), len(losses)


def _apply_v4_post_prune_slicing(model, export_config, hidden_size_order=None):
    """Slice V4-specific weights not tracked by mcore_minitron's DynamicModule.

    After mcore_minitron pruning + export, standard weights are pruned but V4's
    custom attention submodules retain original dimensions. This function slices
    them in-place using importance-based channel selection when available,
    falling back to contiguous slicing (first N channels) otherwise.

    Args:
        model: The pruned Megatron model (already exported).
        export_config: Dict with target hidden_size etc.
        hidden_size_order: Tensor of channel indices sorted by importance
            (from mcore_minitron's sort_parameters). When provided, the top
            target_hidden indices are used for slicing V4 weights.
    """
    target_hidden = export_config.get("hidden_size")
    if target_hidden is None:
        return

    cfg = model.config
    hc_mult = getattr(cfg, "hc_mult", 4)
    target_hc_dim = hc_mult * target_hidden

    # Build the index tensor for hidden_size dimension slicing
    if hidden_size_order is not None:
        # Importance-based: take the top target_hidden channels
        hidden_indices = hidden_size_order[:target_hidden].to("cuda")
        slicing_mode = "importance-based"
    else:
        # Fallback: contiguous first N channels
        hidden_indices = None
        slicing_mode = "contiguous (first N)"

    # For HC weights: dim = hc_mult * hidden_size, structured as
    # [hc_mult * hidden_size] = [hidden, hidden, hidden, hidden] (repeated).
    # When using importance-based slicing, repeat the hidden_indices hc_mult times
    # with offsets: [i, i+H, i+2H, i+3H] for each selected channel i.
    if hidden_indices is not None:
        orig_hidden = hidden_size_order.shape[0]
        hc_offsets = torch.arange(hc_mult, device=hidden_indices.device) * orig_hidden
        hc_indices = (hidden_indices.unsqueeze(0) + hc_offsets.unsqueeze(1)).flatten()
    else:
        hc_indices = None

    sliced_count = 0

    def _slice_param(param, dim, target, indices=None):
        """Slice param along given dim. Use indices for importance-based, else contiguous."""
        if param.data.shape[dim] <= target:
            return False
        if indices is not None:
            param.data = torch.index_select(param.data, dim, indices).contiguous()
        elif dim == 0:
            param.data = param.data[:target, ...].contiguous()
        elif dim == 1:
            param.data = param.data[:, :target, ...].contiguous()
        return True

    for layer in model.decoder.layers:
        sa = getattr(layer, "self_attention", None)
        if sa is None:
            continue

        core_attn = getattr(sa, "core_attention", None)
        if core_attn is None:
            continue

        # Compressor: wkv, wgate — shape [compress_dim, hidden_size]
        compressor = getattr(core_attn, "compressor", None)
        if compressor is not None:
            for name in ("linear_wkv", "linear_wgate"):
                mod = getattr(compressor, name, None)
                if mod and hasattr(mod, "weight"):
                    sliced_count += _slice_param(mod.weight, 1, target_hidden, hidden_indices)

        # Indexer: weights_proj [index_n_heads, hidden], compressor wkv/wgate
        indexer = getattr(core_attn, "indexer", None)
        if indexer is not None:
            wp = getattr(indexer, "linear_weights_proj", None)
            if wp and hasattr(wp, "weight"):
                sliced_count += _slice_param(wp.weight, 1, target_hidden, hidden_indices)

            idx_comp = getattr(indexer, "compressor", None)
            if idx_comp is not None:
                for name in ("linear_wkv", "linear_wgate"):
                    mod = getattr(idx_comp, name, None)
                    if mod and hasattr(mod, "weight"):
                        sliced_count += _slice_param(mod.weight, 1, target_hidden, hidden_indices)

        # HC mapping_proj: shape [mix_hc, hc_mult * hidden_size]
        for hc_name in ("self_attention_hyper_connection", "mlp_hyper_connection"):
            hc = getattr(layer, hc_name, None)
            if hc is None:
                continue
            mp = getattr(hc, "mapping_proj", None)
            if mp and hasattr(mp, "weight"):
                sliced_count += _slice_param(mp.weight, 1, target_hc_dim, hc_indices)

    # hc_head_fn: shape [hc_mult, hc_mult * hidden_size]
    for hc_head_name in ("hc_head_fn",):
        hc_param = getattr(model, hc_head_name, None)
        if hc_param is None:
            hc_param = getattr(model.decoder, hc_head_name, None)
        if hc_param is not None and isinstance(hc_param, torch.nn.Parameter):
            if hc_param.dim() == 2:
                sliced_count += _slice_param(hc_param, 1, target_hc_dim, hc_indices)

    print_rank_0(
        f"V4 post-prune slicing: {sliced_count} parameters ({slicing_mode})"
    )


def main(args):
    assert dist.size() == args.pp_size, "Only Pipeline parallelism is supported for pruning."

    if os.path.exists(f"{args.output_hf_path}/config.json"):
        print_rank_0(f"\nPruned model already exists at {args.output_hf_path}. Exiting...")
        return

    export_config = build_export_config(args)

    provider_overrides = {
        "tensor_model_parallel_size": 1,
        "expert_tensor_parallel_size": 1,
        "pipeline_model_parallel_size": args.pp_size,
        "pipeline_dtype": torch.bfloat16,
        "seq_length": args.seq_length,
        "use_fused_mhc": False,  # tileiras doesn't support sm_90 (Hopper)
    }

    # V4's custom layer spec is not supported by get_gpt_mtp_block_spec.
    # Monkey-patch to return None (skip MTP) on ValueError instead of crashing.
    import megatron.core.models.gpt.gpt_layer_specs as _gpt_specs

    _orig_mtp_spec = _gpt_specs.get_gpt_mtp_block_spec_for_backend

    def _safe_mtp_spec(*a, **kw):
        try:
            return _orig_mtp_spec(*a, **kw)
        except ValueError:
            return None

    _gpt_specs.get_gpt_mtp_block_spec_for_backend = _safe_mtp_spec

    # Monkey-patch GPTModel._postprocess to skip MTP when input is None.
    # During calibration with pipeline parallelism, the last rank's MTP
    # module may receive None input, causing a crash. Skip MTP in that case.
    from megatron.core.models.gpt.gpt_model import GPTModel

    _orig_postprocess = GPTModel._postprocess

    def _safe_postprocess(self, hidden_states, *args, **kwargs):
        if hasattr(self, "mtp") and self.mtp is not None:
            # Temporarily disable MTP if hidden_states is None
            if hidden_states is None:
                return hidden_states
        return _orig_postprocess(self, hidden_states, *args, **kwargs)

    GPTModel._postprocess = _safe_postprocess

    # Monkey-patch MTP roll_tensor to handle None tensors.
    # During calibration with PP, the last rank's MTP may call roll_tensor
    # with None input (e.g., when embedding is not on this rank).
    try:
        from megatron.core.transformer import multi_token_prediction as _mtp_mod

        _orig_roll_tensor = _mtp_mod.roll_tensor

        def _safe_roll_tensor(tensor, *args, **kwargs):
            if tensor is None:
                return None, torch.tensor(0.0)
            return _orig_roll_tensor(tensor, *args, **kwargs)

        _mtp_mod.roll_tensor = _safe_roll_tensor
    except (ImportError, AttributeError):
        pass

    # Patch weight loading for V4: the installed modelopt's mapping registry
    # doesn't have entries for V4's MLA projections and sequential expert weights.
    # Suppress the thousands of "No mapping found" warnings and filter None tasks.
    from megatron.bridge.models.conversion import model_bridge as _mb
    import logging as _logging

    _orig_load = _mb.MegatronModelBridge.load_weights_hf_to_megatron

    def _safe_load(self, hf_pretrained, megatron_model, allowed_mismatched_params=None):
        _orig_build = self.build_conversion_tasks

        def _filtered_build(*a, **kw):
            _mb_logger = _logging.getLogger("megatron.bridge.models.conversion.model_bridge")
            _old_level = _mb_logger.level
            _mb_logger.setLevel(_logging.ERROR)
            try:
                tasks = _orig_build(*a, **kw)
            finally:
                _mb_logger.setLevel(_old_level)
            valid = [t for t in tasks if t is not None and t.megatron_module is not None]
            print_rank_0(
                f"Weight mapping: {len(valid)}/{len(tasks)} params mapped"
            )
            return valid

        self.build_conversion_tasks = _filtered_build
        try:
            return _orig_load(self, hf_pretrained, megatron_model, allowed_mismatched_params)
        finally:
            self.build_conversion_tasks = _orig_build

    _mb.MegatronModelBridge.load_weights_hf_to_megatron = _safe_load

    # V4 uses hash MoE layers which require explicit pipeline layout when PP > 1.
    if args.pp_size > 1:
        with open(os.path.join(args.hf_model_name_or_path, "config.json")) as f:
            hf_raw = json.load(f)
        n_layers = hf_raw["num_hidden_layers"]
        n_mtp = hf_raw.get("num_nextn_predict_layers", 0)
        provider_overrides["pipeline_model_parallel_layout"] = _build_pp_layout(
            n_layers, args.pp_size, n_mtp=n_mtp
        )

    # Use Bridge API directly for model loading. The installed modelopt's
    # load_mbridge_model_from_hf uses a weight conversion path that doesn't
    # support V4's MLA projections and expert weights. The Bridge's own
    # to_megatron_provider uses stream_weights which handles V4 correctly.
    from megatron.core.utils import unwrap_model
    from transformers import AutoTokenizer

    # Patch fast_hadamard_transform with PyTorch fallback.
    # The V4 DSA indexer uses hadamard_transform for rotate_activation,
    # but the library is not installed. Provide a pure PyTorch implementation.
    try:
        import fast_hadamard_transform  # noqa: F401
    except ImportError:
        import sys
        import types as _types

        def _hadamard_transform(x, scale=1.0):
            """PyTorch implementation of Hadamard transform."""
            n = x.shape[-1]
            assert n & (n - 1) == 0, f"Hadamard size must be power of 2, got {n}"
            orig_dtype = x.dtype
            h = x.float()
            # In-place fast Walsh-Hadamard transform
            step = 1
            while step < n:
                h_view = h.view(*h.shape[:-1], n // (2 * step), 2, step)
                a = h_view[..., 0, :]
                b = h_view[..., 1, :]
                h_view_new = torch.stack([a + b, a - b], dim=-2)
                h = h_view_new.reshape(*h.shape[:-1], n)
                step *= 2
            return (h * scale).to(orig_dtype)

        _fht_module = _types.ModuleType("fast_hadamard_transform")
        _fht_module.hadamard_transform = _hadamard_transform
        sys.modules["fast_hadamard_transform"] = _fht_module
        # Also patch the already-imported reference in dsa.py
        try:
            from megatron.core.transformer.experimental_attention_variant import dsa as _dsa
            _dsa.hadamard_transform = _hadamard_transform
        except (ImportError, AttributeError):
            pass
        print_rank_0("Patched fast_hadamard_transform with PyTorch fallback")

    print_rank_0(f"Loading model from {args.hf_model_name_or_path}...")
    bridge = AutoBridge.from_hf_pretrained(
        args.hf_model_name_or_path, trust_remote_code=args.trust_remote_code
    )
    provider = bridge.to_megatron_provider(load_weights=True)
    for key, value in provider_overrides.items():
        assert hasattr(provider, key), f"{type(provider)} does not have attribute {key}"
        setattr(provider, key, value)
    provider.finalize()
    provider.initialize_model_parallel(seed=0)
    model = provider.provide_distributed_model(wrap_with_ddp=False)
    assert len(model) == 1
    unwrapped_model = unwrap_model(model[0])
    tokenizer = AutoTokenizer.from_pretrained(
        args.hf_model_name_or_path, trust_remote_code=args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    mcore_cfg = unwrapped_model.config
    print_rank_0(
        f"Original config: hidden_size={mcore_cfg.hidden_size}, "
        f"num_layers={mcore_cfg.num_layers}, "
        f"num_moe_experts={mcore_cfg.num_moe_experts}"
    )
    if hasattr(mcore_cfg, "compress_ratios"):
        print_rank_0(f"compress_ratios: {mcore_cfg.compress_ratios}")

    # Build calibration data from local Nemotron Post-Training Dataset v2.
    # Matches prune_minitron.py's approach: 1024 real samples for reliable importance estimation.
    _calib_texts = None
    _NEMOTRON_PATH = "/root/.cache/huggingface/hub/datasets--nvidia--Nemotron-Post-Training-Dataset-v2"

    print_rank_0(f"Loading calibration dataset from {_NEMOTRON_PATH}...")
    try:
        import pyarrow.parquet as pq
        import glob as _glob

        # Find all parquet files (chat, stem, code, math, multilingual)
        _parquet_files = _glob.glob(f"{_NEMOTRON_PATH}/snapshots/*/data/*.parquet")
        if not _parquet_files:
            raise FileNotFoundError(f"No parquet files found in {_NEMOTRON_PATH}")

        _calib_texts = []
        _target_samples = max(args.calibration_samples, 1024)

        for _pf in sorted(_parquet_files):
            if len(_calib_texts) >= _target_samples:
                break
            try:
                _table = pq.read_table(_pf)
                _rows = _table.to_pydict()
                _msg_col = 'messages' if 'messages' in _rows else None
                _text_col = 'text' if 'text' in _rows else None

                for _i in range(len(list(_rows.values())[0])):
                    if len(_calib_texts) >= _target_samples:
                        break
                    _text = ""
                    if _msg_col:
                        _msgs = _rows[_msg_col][_i]
                        if isinstance(_msgs, list):
                            for _msg in _msgs:
                                if isinstance(_msg, dict) and 'content' in _msg:
                                    _text += str(_msg['content']) + "\n"
                    elif _text_col:
                        _text = str(_rows[_text_col][_i])

                    if _text and len(_text) > 50:
                        _calib_texts.append(_text)
            except Exception as _e:
                print_rank_0(f"  Skipping {_pf}: {_e}")
                continue

        print_rank_0(f"Loaded {len(_calib_texts)} calibration texts from Nemotron dataset")
    except Exception as e:
        print_rank_0(f"Nemotron dataset load failed ({e}), using fallback texts")
        _calib_texts = [
            "Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data.",
            "The transformer architecture revolutionized natural language processing by introducing the self-attention mechanism.",
            "Deep learning models have achieved superhuman performance in tasks like image recognition and game playing.",
            "Neural networks learn by adjusting millions of parameters through a process called backpropagation.",
            "The internet has fundamentally changed how people communicate, access information, and conduct business.",
            "Space exploration continues to push boundaries, with plans for human missions to Mars in the coming decades.",
            "The theory of general relativity describes gravity as the curvature of spacetime caused by mass and energy.",
            "Autonomous vehicles use a combination of sensors, cameras, and AI to navigate without human intervention.",
        ] * 16  # repeat to get ~128 samples as fallback

    def forward_loop(model):
        """Activation-based importance estimation via real forward passes.

        Hooks into input_layernorm and pre_mlp_layernorm to collect activation
        magnitudes across calibration samples (same approach as prune_minitron.py
        for Qwen3). Falls back to weight-magnitude if forward pass fails.
        """
        from modelopt.torch.nas.plugins.megatron import (
            _DynamicMCoreLanguageModel,
            _DynamicTransformerLayer,
            _DynamicHyperConnectionTransformerLayer,
        )
        import torch.distributed as td

        # Find the DynamicModule language model
        lm = None
        for m in model.modules():
            if isinstance(m, _DynamicMCoreLanguageModel):
                lm = m
                break
        if lm is None:
            print_rank_0("Warning: Could not find DynamicMCoreLanguageModel, skipping importance")
            return

        hp = lm.get_hparam("hidden_size")
        orig_hidden = hp.max
        _rank = td.get_rank() if td.is_initialized() else -1

        # ── Register activation hooks (same as prune_minitron.py) ────
        _activations = {}

        def _layernorm_hook(mod, input, output):
            """Collect activation magnitudes from layernorm output."""
            act = output.detach().float()
            # Mean over seq_len, L2 over batch, accumulate
            if act.dim() == 3:  # [seq, batch, hidden]
                act = act.abs().mean(dim=0)  # [batch, hidden]
            elif act.dim() == 2:
                act = act.abs()
            scores = act.pow(2).sum(dim=0)  # [hidden]
            key = id(mod)
            if key not in _activations:
                _activations[key] = scores.cpu()
            else:
                _activations[key] += scores.cpu()

        _hooks = []
        for layer in lm.decoder.layers:
            if isinstance(layer, (_DynamicTransformerLayer, _DynamicHyperConnectionTransformerLayer)):
                if hasattr(layer, "input_layernorm"):
                    h = layer.input_layernorm.register_forward_hook(_layernorm_hook)
                    _hooks.append(h)
                if hasattr(layer, "pre_mlp_layernorm"):
                    h = layer.pre_mlp_layernorm.register_forward_hook(_layernorm_hook)
                    _hooks.append(h)

        # ── Run forward passes ────────────────────────────────────────
        from megatron.core.pipeline_parallel import get_forward_backward_func

        model.eval()
        _forward_ok = True
        _sample_count = 0

        def _dummy_loss(output_tensor, non_loss_data=True):
            return output_tensor

        def _forward_step(data, model):
            tokens = data["tokens"]
            b, s = tokens.shape
            device = tokens.device
            mask = torch.triu(torch.ones(b, s, s, device=device), 1).bool().view(b, 1, s, s)
            out = model(tokens, position_ids=None, attention_mask=mask)
            return out, _dummy_loss

        with torch.no_grad():
            if _calib_texts is not None:
                data_iter = iter(_calib_texts)
            else:
                data_iter = _calib_ds_iter

            for i, text in enumerate(data_iter):
                if i >= args.calibration_samples:
                    break
                tokens = tokenizer.encode(text)
                if len(tokens) < 10:
                    continue
                tokens = tokens[: args.seq_length]
                tokens_t = torch.tensor([tokens], dtype=torch.long, device="cuda")
                try:
                    get_forward_backward_func()(
                        forward_step_func=_forward_step,
                        data_iterator=[{"tokens": tokens_t}],
                        model=model,
                        num_microbatches=1,
                        seq_length=tokens_t.shape[-1],
                        micro_batch_size=1,
                        decoder_seq_length=tokens_t.shape[-1],
                        forward_only=True,
                        collect_non_loss_data=True,
                    )
                    _sample_count += 1
                    if _sample_count % 20 == 0:
                        print(f"[rank{_rank}] Forward {_sample_count}/{args.calibration_samples}", flush=True)
                except Exception as e:
                    import traceback
                    if _sample_count == 0:
                        print(f"[rank{_rank}] Forward pass failed on first sample: {e}", flush=True)
                        traceback.print_exc()
                        _forward_ok = False
                        break
                    print(f"[rank{_rank}] Forward sample {i} failed: {e}", flush=True)

        # Clean up hooks
        for h in _hooks:
            h.remove()

        # ── Compute importance ────────────────────────────────────────
        if _forward_ok and _activations:
            # Activation-based importance (same as prune_minitron.py)
            agg = [act.pow(0.5) for act in _activations.values()]
            importance = torch.stack(agg).sum(dim=0)  # [hidden]
            # All-reduce across PP ranks
            importance = importance.clone()
            torch.distributed.all_reduce(importance, op=torch.distributed.ReduceOp.SUM)
            order = importance.argsort(descending=True)
            print_rank_0(
                f"Activation-based importance: {_sample_count} samples, "
                f"{len(_activations)} hooks (top 16: {order[:16].tolist()})"
            )
        else:
            # Fallback: weight-magnitude importance
            print_rank_0(f"Forward pass failed ({_sample_count} samples), falling back to weight-magnitude")
            importance = torch.zeros(orig_hidden, device="cpu")
            count = 0
            for name, param in model.named_parameters():
                if param is None or not hasattr(param, 'shape'):
                    continue
                p = param.data.float().cpu()
                for dim in range(p.dim()):
                    if p.shape[dim] == orig_hidden:
                        other_dims = [d for d in range(p.dim()) if d != dim]
                        scores = p.norm(dim=other_dims) if other_dims else p.abs()
                        importance += scores
                        count += 1
                        break
            order = importance.argsort(descending=True)
            print_rank_0(
                f"Weight-magnitude fallback: {count} tensors (top 16: {order[:16].tolist()})"
            )

        # Register importance function
        hp._importance_estimators = [lambda: order]
        hp._importance_is_order = True

        # Pre-populate layer scores for depth pruning
        for layer in lm.decoder.layers:
            if isinstance(layer, (_DynamicTransformerLayer, _DynamicHyperConnectionTransformerLayer)):
                layer_norm = 0.0
                for p in layer.parameters():
                    if p is not None and p.numel() > 0:
                        layer_norm += p.data.float().norm().item()
                layer._scores = layer_norm + 1e-6  # ensure > 0

    # Monkey-patch get_sliced_tensor_by_slices to handle size-0 tensors safely.
    # Without this, the DynamicModule export crashes with IndexError when trying
    # to apply importance ordering to empty tensors (e.g. shared expert bias).
    from modelopt.torch.nas.modules import utils as _nas_utils

    _orig_get_sliced = _nas_utils.get_sliced_tensor_by_slices

    def _safe_get_sliced(tensor, slices):
        if tensor is None or not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
            return tensor
        # Validate that index slices don't exceed tensor dimensions
        for i, s in enumerate(slices):
            if isinstance(s, torch.Tensor) and tensor.shape[i] > 0:
                if s.max().item() >= tensor.shape[i]:
                    return tensor  # can't apply ordering, return as-is
        return _orig_get_sliced(tensor, slices)

    _nas_utils.get_sliced_tensor_by_slices = _safe_get_sliced

    # Monkey-patch _DynamicMCoreLanguageModel.export() to capture hidden_size
    # channel ordering BEFORE export replaces DynamicModule weights.
    # After sort_parameters(), each hidden_size hparam has _slice_order set to
    # the importance-based ranking. We capture it here so it can be used by
    # _apply_v4_post_prune_slicing for V4-specific weights.
    from modelopt.torch.nas.plugins.megatron import _DynamicMCoreLanguageModel

    _orig_lm_export = _DynamicMCoreLanguageModel.export
    _captured_v4_order = {}

    def _patched_lm_export(self):
        try:
            hp = self.get_hparam("hidden_size")
            if getattr(hp, "_slice_order", None) is not None:
                _captured_v4_order["hidden_size"] = hp._slice_order.clone()
                print_rank_0(
                    f"Captured hidden_size importance ordering "
                    f"(first 16: {hp._slice_order[:16].tolist()})"
                )
        except Exception as e:
            print_rank_0(f"Warning: Could not capture hidden_size ordering: {e}")
        return _orig_lm_export(self)

    _DynamicMCoreLanguageModel.export = _patched_lm_export

    print_rank_0(f"Pruning with export_config={export_config}...")
    print_rank_0(f"Calibration: {args.calibration_samples} samples, seq_len={args.seq_length}")
    print_rank_0(f"Importance-based channel selection: ENABLED")
    ss_config = mtp.mcore_minitron.get_mcore_minitron_config(
        channel_divisor=128,
        num_moe_experts_divisor=8,
    )
    unwrapped_model, pruning_scores = mtp.prune(
        unwrapped_model,
        mode=[("mcore_minitron", ss_config)],
        constraints={"export_config": export_config},
        dummy_input=None,
        config={"forward_loop": forward_loop, "skip_sorting": False},
    )

    # Restore original export and slicing function
    _DynamicMCoreLanguageModel.export = _orig_lm_export
    _nas_utils.get_sliced_tensor_by_slices = _orig_get_sliced

    hidden_size_order = _captured_v4_order.get("hidden_size")
    if hidden_size_order is not None:
        print_rank_0(
            f"Using importance ordering for V4 weight slicing "
            f"(first 16: {hidden_size_order[:16].tolist()})"
        )

    if mto.ModeloptStateManager.has_state_for_mode_type("prune", model=unwrapped_model):
        mto.ModeloptStateManager.remove_state(unwrapped_model)

    # ── PP layout rebuild after prune ────────────────────────────────
    # mcore_minitron drops layers in-place but leaves the original
    # pipeline_model_parallel_layout string stale (still pointing at the
    # pre-prune layer count). Recompute and patch the layout metadata on the
    # model so any post-prune forward pass (validation, etc.) sees a layout
    # whose per-stage transformer count matches the model's actual
    # `decoder.layers` length. Without this, validation loss crashes inside
    # the pipeline schedule with `IndexError: list index out of range` when a
    # tail rank ends up with 0 layers.
    n_mtp_after = 0
    try:
        with open(os.path.join(args.hf_model_name_or_path, "config.json")) as f:
            _hf_raw_for_layout = json.load(f)
        n_mtp_after = _hf_raw_for_layout.get("num_nextn_predict_layers", 0)
    except Exception:
        pass
    new_layout = _rebuild_pp_layout_after_prune(
        unwrapped_model,
        args.pp_size,
        n_mtp=n_mtp_after,
        # Use the *target* post-prune num_layers (the CLI value or the
        # export_config override), NOT len(unwrapped_model.decoder.layers)
        # which is only the per-rank chunk count.
        n_layers=export_config.get("num_layers") or args.num_layers,
    )
    print_rank_0(
        f"Rebuilt PP layout after prune: {new_layout} "
        f"(model.decoder.layers={len(unwrapped_model.decoder.layers)} on this rank)"
    )

    # ── V4-specific post-pruning weight slicing ──────────────────────
    # mcore_minitron's DynamicModule system correctly prunes standard
    # weights (embedding, layernorms, MLA projections, expert FFN, router)
    # but does NOT track V4's custom attention submodules:
    #   - Compressor (wkv, wgate): input dim = hidden_size
    #   - Indexer (weights_proj, compressor.wkv/wgate): input dim = hidden_size
    #   - Hyper-Connection fn weights: last dim = hc_mult * hidden_size
    # These must be sliced in-place to match the pruned hidden_size.
    _apply_v4_post_prune_slicing(unwrapped_model, export_config, hidden_size_order)

    # ── Validation loss on the pruned model ─────────────────────────
    # Use a small held-out subset of the calibration data to estimate the
    # validation loss of the pruned configuration. This becomes the ranking
    # signal in NAS (rather than the heuristic quality_score) and a
    # check on whether pruning degraded the model too much.
    #
    # Disabled by default (`--validation_samples` defaults to 0): the
    # forward pass over the pruned model under PP=8 / TP=1 + mHC + MTP
    # is currently unstable on the 286B V4 model (rank-7 forward can hang
    # on lm_head / MTP). Re-enable with `--validation_samples N>0` once
    # that's fixed.
    val_loss = float("nan")
    val_num_samples = 0
    val_skipped = getattr(args, "validation_samples", 0) <= 0
    if val_skipped:
        print_rank_0(
            "Pruned validation loss: SKIPPED (--validation_samples <= 0). "
            "Using heuristic quality_score."
        )
    try:
        val_loss, val_num_samples = _compute_pruned_validation_loss(
            unwrapped_model=unwrapped_model,
            tokenizer=tokenizer,
            calib_texts=_calib_texts,
            seq_length=args.seq_length,
            num_samples=getattr(args, "validation_samples", 32),
            model_chunks=model,
        )
        print_rank_0(
            f"Pruned validation loss: {val_loss:.4f} "
            f"(on {val_num_samples} samples)"
        )
    except Exception as _e:
        print_rank_0(
            f"Pruned validation loss failed: {type(_e).__name__}: {_e}. "
            f"Continuing without validation signal."
        )

    mcore_cfg = unwrapped_model.config
    print_rank_0(
        f"Pruned config: hidden_size={mcore_cfg.hidden_size}, "
        f"num_layers={mcore_cfg.num_layers}, "
        f"num_moe_experts={mcore_cfg.num_moe_experts}"
    )
    if hasattr(mcore_cfg, "compress_ratios"):
        print_rank_0(f"Pruned compress_ratios: {mcore_cfg.compress_ratios}")

    print_rank_0(f"Saving pruned model to {args.output_hf_path}")

    megatron_path = args.output_hf_path + "_megatron"
    os.makedirs(megatron_path, exist_ok=True)
    state_dict = unwrapped_model.state_dict()
    # Skip routed expert BF16 weights: convert_pruned_to_hf.py reloads them
    # from the native MXFP4 checkpoint. Expert BF16 is ~98% of state_dict.
    # NOTE: must NOT filter shared_experts — they are needed in the output.
    state_dict = {
        k: v for k, v in state_dict.items()
        if "shared_experts" in k or "experts.linear_fc" not in k
    }

    # PP>1: each rank only sees a subset of layers (its pipeline stage).
    # convert_pruned_to_hf.py expects one .pt per PP rank in this directory
    # (e.g. ``pruned_model_rank{0..7}.pt``) and merges them by taking the
    # largest shard as the base + filling missing keys from the others.
    # So each rank writes its own shard here — we deliberately do NOT
    # gather+merge on rank 0, because the gather path produces a single
    # ``pruned_model.pt`` that convert_pruned_to_hf.py does not accept
    # under PP>1.
    if dist.is_initialized() and dist.size() > 1:
        rank = dist.rank()
        megatron_file = os.path.join(
            megatron_path, f"pruned_model_rank{rank}.pt"
        )
        torch.save(state_dict, megatron_file)
        if rank == 0:
            print_rank_0(
                f"Saved per-rank shards under {megatron_path} "
                f"(rank 0 has {len(state_dict)} params, experts excluded)"
            )
    else:
        megatron_file = os.path.join(megatron_path, "pruned_model.pt")
        torch.save(state_dict, megatron_file)
        print_rank_0(
            f"Saved {len(state_dict)} params (experts excluded) to {megatron_file}"
        )
    torch.distributed.barrier()

    if dist.rank() == 0:
        # Save the pruned HF config
        bridge.hf_pretrained.save_artifacts(args.output_hf_path)
        hf_cfg = AutoConfig.from_pretrained(
            args.output_hf_path, trust_remote_code=args.trust_remote_code
        )
        if hasattr(mcore_cfg, "compress_ratios") and mcore_cfg.compress_ratios is not None:
            pruned_ratios = list(mcore_cfg.compress_ratios)[: mcore_cfg.num_layers]
            try:
                hf_cfg.compress_ratios = pruned_ratios
            except AttributeError:
                pass
            if hasattr(hf_cfg, "layer_types"):
                _ratio_to_type = {
                    0: "sliding_attention",
                    4: "compressed_sparse_attention",
                    128: "heavily_compressed_attention",
                }
                hf_cfg.layer_types = [
                    _ratio_to_type.get(r, "sliding_attention") for r in pruned_ratios
                ]
        elif hasattr(hf_cfg, "layer_types"):
            hf_cfg.layer_types = hf_cfg.layer_types[: mcore_cfg.num_layers]
        if hasattr(hf_cfg, "mlp_layer_types") and hf_cfg.mlp_layer_types is not None:
            hf_cfg.mlp_layer_types = hf_cfg.mlp_layer_types[: mcore_cfg.num_layers]
        hf_cfg.hidden_size = mcore_cfg.hidden_size
        hf_cfg.num_hidden_layers = mcore_cfg.num_layers
        if hasattr(hf_cfg, "n_routed_experts"):
            hf_cfg.n_routed_experts = mcore_cfg.num_moe_experts
        if hasattr(hf_cfg, "moe_intermediate_size"):
            hf_cfg.moe_intermediate_size = mcore_cfg.moe_ffn_hidden_size
        hf_cfg.save_pretrained(args.output_hf_path)

        # Patch config.json with head_dim (post-prune slicing target).
        # AutoConfig doesn't persist unknown fields reliably, so we do it
        # on-disk. convert_pruned_to_hf.py reads head_dim from this file to
        # decide how aggressively to slice wq_b / wkv / wo_a / compressor.
        if args.head_dim is not None:
            cfg_path_patch = os.path.join(args.output_hf_path, "config.json")
            with open(cfg_path_patch) as _f:
                _cfg_patch = json.load(_f)
            _cfg_patch["head_dim"] = args.head_dim
            # Recalculate partial_rotary_factor to preserve qk_rope_head_dim.
            # transformers' DeepseekV4Config.__post_init__ recomputes:
            #   qk_rope_head_dim = int(head_dim * partial_rotary_factor)
            # Original: 512 * 0.125 = 64. After pruning head_dim to 256,
            # partial_rotary_factor must become 64/256 = 0.25.
            _qk_rope = _cfg_patch.get("qk_rope_head_dim", 64)
            _new_prf = _qk_rope / args.head_dim
            _cfg_patch["partial_rotary_factor"] = _new_prf
            # Also update nested rope_parameters.*.partial_rotary_factor —
            # all attention types share the same MLA rope dimension.
            _rope_params = _cfg_patch.get("rope_parameters")
            if isinstance(_rope_params, dict):
                for _rp_section in _rope_params.values():
                    if isinstance(_rp_section, dict) and "partial_rotary_factor" in _rp_section:
                        _rp_section["partial_rotary_factor"] = _new_prf
            with open(cfg_path_patch, "w") as _f:
                json.dump(_cfg_patch, _f, indent=2)
            print_rank_0(
                f"Patched head_dim={args.head_dim} into {cfg_path_patch} "
                f"(partial_rotary_factor={_new_prf})"
            )

        # Patch config.json with fields that AutoConfig doesn't persist
        cfg_path = os.path.join(args.output_hf_path, "config.json")
        with open(cfg_path) as f:
            cfg_json = json.load(f)
        # compress_ratios drives SGLang's CSA/HCA compressor instantiation
        # (see config.compress_ratios[layer_id] in DeepseekV4Attention). When
        # mcore_cfg doesn't carry it (e.g. the model was built without the
        # experimental_attention_variant transformer config), fall back to the
        # original HF config's compress_ratios or layer_types so the SGLang
        # params_dict matches the saved state_dict.
        if hasattr(mcore_cfg, "compress_ratios") and mcore_cfg.compress_ratios is not None:
            cfg_json["compress_ratios"] = list(mcore_cfg.compress_ratios)[: mcore_cfg.num_layers]
        else:
            orig_cfg_path = os.path.join(args.hf_model_name_or_path, "config.json")
            if os.path.exists(orig_cfg_path):
                with open(orig_cfg_path) as f:
                    orig_json = json.load(f)
                orig_ratios = orig_json.get("compress_ratios")
                if orig_ratios and len(orig_ratios) >= mcore_cfg.num_layers:
                    cfg_json["compress_ratios"] = list(orig_ratios)[: mcore_cfg.num_layers]
                else:
                    _type_to_ratio = {
                        "sliding_attention": 0,
                        "compressed_sparse_attention": 4,
                        "heavily_compressed_attention": 128,
                    }
                    src_types = orig_json.get("layer_types") or cfg_json.get("layer_types")
                    if src_types:
                        cfg_json["compress_ratios"] = [
                            _type_to_ratio.get(t, 0)
                            for t in src_types[: mcore_cfg.num_layers]
                        ]
        if "mlp_layer_types" in cfg_json:
            cfg_json["num_hash_layers"] = sum(
                1 for t in cfg_json["mlp_layer_types"] if t == "hash_moe"
            )
        orig_cfg_path = os.path.join(args.hf_model_name_or_path, "config.json")
        if os.path.exists(orig_cfg_path):
            with open(orig_cfg_path) as f:
                orig_json = json.load(f)
            if "rope_scaling" in orig_json and "rope_scaling" not in cfg_json:
                cfg_json["rope_scaling"] = orig_json["rope_scaling"]
        if "torch_dtype" not in cfg_json:
            cfg_json["torch_dtype"] = "bfloat16"
        with open(cfg_path, "w") as f:
            json.dump(cfg_json, f, indent=2)

        # ── Save best_config.json sidecar with val_loss for NAS ranking ──
        # The NAS script consumes `<output>_best_config.json`; write a
        # val_loss-bearing sidecar so the ranking step can use the real
        # validation loss instead of the heuristic quality_score.
        best_config = {
            "hidden_size": mcore_cfg.hidden_size,
            "num_layers": mcore_cfg.num_layers,
            "num_moe_experts": mcore_cfg.num_moe_experts,
            "moe_ffn_hidden_size": mcore_cfg.moe_ffn_hidden_size,
            "moe_shared_expert_intermediate_size": mcore_cfg.moe_shared_expert_intermediate_size,
            "head_dim": args.head_dim,  # post-prune slicing target; None = keep original
            "params_ratio": None,  # filled by NAS, if needed
            "val_loss": val_loss if val_loss == val_loss else None,  # NaN-safe
            "val_num_samples": val_num_samples,
            "compression": "mcore_minitron",
        }
        best_config_path = args.output_hf_path + "_best_config.json"
        with open(best_config_path, "w") as f:
            json.dump(best_config, f, indent=2)
        print_rank_0(f"Saved best_config sidecar to {best_config_path}")

        print_rank_0(f"Saved pruned HF config to {args.output_hf_path}")
        print_rank_0(
            f"Run merge_pruned_state_dicts.py to combine {args.pp_size} rank files into one."
        )

    print_rank_0("Done!")


if __name__ == "__main__":
    dist.setup()
    args = parse_args()
    try:
        main(args)
    finally:
        dist.cleanup()

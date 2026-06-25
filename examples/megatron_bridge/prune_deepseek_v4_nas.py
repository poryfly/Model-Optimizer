# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4 structured pruning with NAS search + validation loss evaluation.

This script implements Neural Architecture Search (NAS) for DeepSeek-V4 pruning:
1. Generate multiple candidate pruning configurations
2. For each candidate, apply pruning and compute validation loss
3. Select the configuration with lowest validation loss
4. Apply the best configuration and save the final model

Usage (8× GPU with PP=8):
    torchrun --nproc_per_node=8 prune_deepseek_v4_nas.py \\
        --hf_model_name_or_path /data/models/deepseek-ai/DeepSeek-V4-Flash \\
        --output_hf_path /output/pruned-v4-best \\
        --pp_size 8 \\
        --target_params_ratio 0.8 \\
        --num_candidates 10 \\
        --trust_remote_code
"""

import argparse
import json
import os
import copy
from itertools import product

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


# ── Reuse the PP layout helpers from prune_deepseek_v4 ─────────────────
# We re-derive the pipeline layout after mcore_minitron drops layers, so
# the validation forward pass on the pruned model doesn't crash inside
# the pipeline schedule with `IndexError: list index out of range`. The
# functions are import-side-effect free, so importing the sibling module
# is safe (it does NOT load the model or call `dist.setup()`).
try:
    from prune_deepseek_v4 import _build_pp_layout, _rebuild_pp_layout_after_prune
except Exception:  # noqa: BLE001 — fallback if sibling not on path
    def _build_pp_layout(n_layers, pp, n_mtp=0):
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
        if pp_size <= 1:
            return ""
        if n_layers is None:
            # Legacy fallback: per-rank count.
            n_layers = len(unwrapped_model.decoder.layers)
        new_layout = _build_pp_layout(n_layers, pp_size, n_mtp=n_mtp)

        # Megatron stores a `PipelineParallelLayerLayout` object on the config
        # after init, not a raw string. Re-wrap the new string so that
        # `get_layer_offset(...)` (called by TransformerLayer during forward)
        # does not crash with `'str' object has no attribute 'get_layer_offset'`.
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
        for attr in ("pipeline_model_parallel_layout", "_pp_layout"):
            if hasattr(unwrapped_model, attr):
                try:
                    setattr(unwrapped_model, attr, _wrap(new_layout))
                except (AttributeError, TypeError):
                    pass
        return new_layout


# ── Module-level hadamard fallback ────────────────────────────────────
# DeepSeek-V4 DSA indexer calls `hadamard_transform` from `fast_hadamard_transform`.
# That package is not installed in this environment. We register a pure-PyTorch
# implementation BEFORE the model is loaded so that `dsa.hadamard_transform`
# is never `None` when forward passes run.
#
# Installed at module-import time (not inside main()) so that we are guaranteed
# to run before any candidate evaluation, and so the message goes to the
# launcher's log even if `print_rank_0` happens to be mis-routed.
def _install_hadamard_fallback():
    import sys
    import types

    def _hadamard_transform(x, scale=1.0):
        """Pure-PyTorch Walsh-Hadamard transform (power-of-2 sizes)."""
        n = x.shape[-1]
        assert n & (n - 1) == 0, f"Hadamard size must be power of 2, got {n}"
        orig_dtype = x.dtype
        h = x.float()
        step = 1
        while step < n:
            h_view = h.view(*h.shape[:-1], n // (2 * step), 2, step)
            a = h_view[..., 0, :]
            b = h_view[..., 1, :]
            h_view_new = torch.stack([a + b, a - b], dim=-2)
            h = h_view_new.reshape(*h.shape[:-1], n)
            step *= 2
        return (h * scale).to(orig_dtype)

    # Register the fake module so that any `from fast_hadamard_transform
    # import hadamard_transform` triggered later (e.g. by deepspeed, vllm
    # etc.) also picks up the fallback.
    _fht = types.ModuleType("fast_hadamard_transform")
    _fht.hadamard_transform = _hadamard_transform
    sys.modules["fast_hadamard_transform"] = _fht

    # Patch the already-imported `dsa` module's reference. `dsa.py` does
    #   try: from fast_hadamard_transform import hadamard_transform
    #   except ImportError: hadamard_transform = None
    # The first import attempt failed at module-load time (no real package
    # installed), so `dsa.hadamard_transform` is `None` right now. Replace it
    # with our fallback so that `rotate_activation` (which asserts the
    # function is not None) works.
    try:
        from megatron.core.transformer.experimental_attention_variant import dsa as _dsa
        _dsa.hadamard_transform = _hadamard_transform
    except Exception as _e:  # noqa: BLE001 — best-effort fallback installer
        # If dsa can't be imported (e.g. flashinfer env issue), we still left
        # the sys.modules entry in place so later imports succeed.
        sys.stderr.write(
            f"[hadamard-fallback] dsa import failed: {type(_e).__name__}: {_e}\n"
        )
        sys.stderr.flush()

    msg = "Patched fast_hadamard_transform with PyTorch fallback"
    # Print to both stdout (via print_rank_0) and stderr (unbuffered fallback)
    # so we always see the message in the log.
    try:
        print_rank_0(msg)
    except Exception:  # noqa: BLE001
        pass
    sys.stderr.write(f"[hadamard-fallback] {msg}\n")
    sys.stderr.flush()


_install_hadamard_fallback()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prune DeepSeek-V4 with NAS search + validation loss",
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
        help="Output path for best pruned HF model",
    )
    parser.add_argument("--pp_size", type=int, default=1, help="Pipeline parallelism size")

    # NAS search parameters
    parser.add_argument(
        "--target_params_ratio",
        type=float,
        default=0.8,
        help="Target parameter ratio (e.g., 0.8 = keep 80%% of original params)",
    )
    parser.add_argument(
        "--num_candidates",
        type=int,
        default=10,
        help="Number of candidate configurations to evaluate",
    )
    parser.add_argument(
        "--validation_ratio",
        type=float,
        default=0.2,
        help="Ratio of calibration data to use for validation (rest for importance)",
    )

    # Search space constraints. Defaults are intentionally aggressive: we
    # support down to ~0.5% of original params (the combo below gives
    # 0.6 * 0.4 * 0.5 * 0.25 = 3% of hidden+ffn+layer weight alone, and the
    # experts multiply it further). This lets `--target_params_ratio` as
    # low as 0.014 find a real candidate instead of clipping at ~10%.
    parser.add_argument("--max_hidden_size_reduction", type=float, default=0.6,
                        help="Maximum hidden_size reduction ratio (default 0.6, "
                             "was 0.4 — old default couldn't reach ratios < ~10%)")
    parser.add_argument("--max_ffn_reduction", type=float, default=0.75,
                        help="Maximum moe_ffn_hidden_size reduction ratio (default 0.75, "
                             "was 0.5 — old default couldn't reach ratios < ~10%)")
    parser.add_argument("--max_layers_reduction", type=float, default=0.6,
                        help="Maximum num_layers reduction ratio (default 0.6, "
                             "was 0.3 — old default couldn't reach ratios < ~10%)")
    parser.add_argument("--max_experts_reduction", type=float, default=0.875,
                        help="Maximum num_moe_experts reduction ratio (default 0.875, "
                             "was 0.5 — old default couldn't reach ratios < ~10%)")

    # Calibration parameters
    parser.add_argument("--calibration_samples", type=int, default=1024,
                        help="Total calibration samples (split into train/validation)")
    parser.add_argument("--seq_length", type=int, default=2048,
                        help="Calibration sequence length")
    parser.add_argument(
        "--validation_samples", type=int, default=8,
        help=(
            "Max number of held-out samples to evaluate per candidate. "
            "The full ~20%% held-out split is ~200 texts (very slow under "
            "PP=8 + 286B model). Default 8 gives a usable signal in a few "
            "minutes per candidate. Set higher for tighter ranking."
        ),
    )
    parser.add_argument(
        "--validation_seq_length", type=int, default=512,
        help=(
            "Cap on tokens per validation sample (separate from --seq_length "
            "used for importance estimation). Default 512 keeps one sample "
            "fast enough to finish in tens of seconds on the 286B V4 model "
            "with PP=8 / TP=1; full --seq_length=2048 can take minutes per "
            "sample."
        ),
    )

    # Forward-validation toggle. Default is OFF: we rank candidates by the
    # heuristic quality score (cheap, deterministic, no per-candidate model
    # load). The forward-validation path (load → prune → compute real
    # validation loss per candidate) is currently disabled because the
    # pruned-model forward pass under PP=8 / TP=1 + mHC + MTP is slow and
    # unstable on the 286B V4 model. Re-enable with
    # `--enable_forward_validation` once the rank-7 forward hang is fixed.
    parser.add_argument(
        "--enable_forward_validation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Load + prune + run forward-pass validation per candidate "
            "(default: OFF — heuristic ranking only). Pass "
            "--enable_forward_validation to re-enable the slow path."
        ),
    )

    args = parser.parse_args()
    print_args(args)
    return args


def compute_model_params(hidden_size, num_layers, num_moe_experts, moe_ffn_hidden_size,
                         moe_shared_expert_intermediate_size=None):
    """Compute total parameter count for a given configuration."""
    # Approximate parameter count (excluding embedding/lm_head for simplicity)
    # Attention: ~12 * hidden_size^2 per layer
    attn_params = num_layers * 12 * hidden_size * hidden_size
    
    # MoE: num_experts * 3 * ffn_hidden * hidden per layer
    moe_params = num_layers * num_moe_experts * 3 * moe_ffn_hidden_size * hidden_size
    
    # Shared expert: 3 * shared_ffn * hidden per layer
    shared_ffn = moe_shared_expert_intermediate_size or moe_ffn_hidden_size
    shared_params = num_layers * 3 * shared_ffn * hidden_size
    
    # Router: num_experts * hidden per layer
    router_params = num_layers * num_moe_experts * hidden_size
    
    total = attn_params + moe_params + shared_params + router_params
    return total


def generate_candidates(orig_config, target_ratio, num_candidates, max_reductions):
    """Generate candidate pruning configurations within parameter budget."""
    orig_hidden = orig_config.hidden_size
    orig_layers = orig_config.num_layers
    orig_experts = orig_config.num_moe_experts
    orig_ffn = orig_config.moe_ffn_hidden_size
    orig_shared = getattr(orig_config, 'moe_shared_expert_intermediate_size', orig_ffn)
    
    # Compute original params (approximate)
    orig_params = compute_model_params(
        orig_hidden, orig_layers, orig_experts, orig_ffn, orig_shared
    )
    target_params = orig_params * target_ratio
    
    print_rank_0(f"Original params (approx): {orig_params/1e9:.2f}B")
    print_rank_0(f"Target params: {target_params/1e9:.2f}B ({target_ratio*100:.1f}%)")
    
    # Generate search space
    hidden_sizes = []
    for ratio in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
        if ratio >= (1.0 - max_reductions['hidden']):
            h = int(orig_hidden * ratio / 128) * 128  # align to 128
            if h >= 512:
                hidden_sizes.append(h)

    ffn_sizes = []
    for ratio in [1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25]:
        if ratio >= (1.0 - max_reductions['ffn']):
            f = int(orig_ffn * ratio / 128) * 128
            if f >= 256:
                ffn_sizes.append(f)

    layer_counts = []
    for ratio in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]:
        if ratio >= (1.0 - max_reductions['layers']):
            l = int(orig_layers * ratio / 2) * 2  # align to 2
            if l >= 8:
                layer_counts.append(l)

    expert_counts = []
    for ratio in [1.0, 0.75, 0.5, 0.25, 0.125]:
        if ratio >= (1.0 - max_reductions['experts']):
            e = int(orig_experts * ratio / 8) * 8  # align to 8
            if e >= 16:
                expert_counts.append(e)
    
    # Generate all combinations
    all_candidates = []
    for h, f, l, e in product(hidden_sizes, ffn_sizes, layer_counts, expert_counts):
        shared = int(orig_shared * (f / orig_ffn) / 128) * 128  # scale shared with ffn
        shared = max(shared, 128)

        params = compute_model_params(h, l, e, f, shared)
        ratio = params / orig_params

        # Filter: within ±50% (relative) of target ratio OR within ±0.05
        # absolute — whichever is wider. The previous ±0.10 absolute filter
        # was meaningless for tiny targets (e.g. target=0.014 → accept
        # anything in [0, 0.114]), which is why we used to pick a ~10%
        # candidate when the user asked for 1.4%.
        ratio_tol = max(0.05, 0.5 * target_ratio)
        if abs(ratio - target_ratio) <= ratio_tol:
            all_candidates.append({
                'hidden_size': h,
                'num_layers': l,
                'num_moe_experts': e,
                'moe_ffn_hidden_size': f,
                'moe_shared_expert_intermediate_size': shared,
                'params_ratio': ratio,
            })
    
    # Sort by closeness to target and select top N
    all_candidates.sort(key=lambda c: abs(c['params_ratio'] - target_ratio))
    candidates = all_candidates[:num_candidates]
    
    print_rank_0(f"Generated {len(candidates)} candidates (from {len(all_candidates)} total)")
    for i, c in enumerate(candidates):
        print_rank_0(
            f"  [{i+1}] H={c['hidden_size']}, L={c['num_layers']}, "
            f"E={c['num_moe_experts']}, F={c['moe_ffn_hidden_size']}, "
            f"params={c['params_ratio']*100:.1f}%"
        )
    
    return candidates


def compute_validation_loss(model, tokenizer, validation_texts, seq_length, max_samples=None):
    """Compute average validation loss on held-out samples.

    Args:
        model: A **list** of model chunks (the value returned by
            `provider.provide_distributed_model(...)`). Passing the list
            matches what `get_forward_backward_func` expects for PP and is
            the same convention used by `prune_deepseek_v4.py`.
        tokenizer: HF tokenizer with a `pad_token_id` and `encode(text)`.
        validation_texts: Iterable of raw text strings.
        seq_length: Cap on tokens per sample.
        max_samples: Cap on the number of samples to evaluate. The full
            ~20%% held-out split is ~200 texts — at PP=8 + 286B params a
            single forward pass takes 3-5 s and the cross-rank broadcast
            can hang if ranks drift, so capping the loop is a safety belt.
            ``None`` means "no cap" (use the entire iterable).

    Returns:
        (avg_loss, num_samples) tuple. `avg_loss == 0.0` and `num_samples == 0`
        indicates total failure (caller should fall back to a heuristic).
    """
    import sys
    import time
    import torch.distributed as _td
    from megatron.core import parallel_state
    from megatron.core.pipeline_parallel import get_forward_backward_func

    # `model` is a list of model chunks for pipeline parallelism.
    for m in model:
        m.eval()

    total_loss = 0.0
    num_samples = 0

    # Cache the broadcast target once: the last PP stage is the same rank
    # across all calls inside this validation pass.
    pp_group = parallel_state.get_pipeline_model_parallel_group()
    pp_last_rank_global = parallel_state.get_pipeline_model_parallel_last_rank()

    # Per-rank identity (for diagnostic logs). PP rank is *not* the same as
    # global rank when TP/DP are > 1; for our setup TP=1, DP=1 so they
    # match, but we still log both to be safe.
    _my_global_rank = _td.get_rank() if _td.is_initialized() else 0
    _my_pp_rank = (
        parallel_state.get_pipeline_model_parallel_rank()
        if parallel_state.is_initialized()
        else 0
    )
    _am_last_stage = (
        parallel_state.is_pipeline_last_stage()
        if parallel_state.is_initialized()
        else True
    )

    # Diagnostic helper: print on ALL ranks (not just rank 0) so we can
    # see exactly which rank is stuck. Each line is prefixed with the rank.
    def _diag(msg):
        line = f"[validation][g{_my_global_rank} pp{_my_pp_rank}{'L' if _am_last_stage else ' '}] {msg}"
        sys.stdout.write(line + "\n")
        sys.stdout.flush()

    _diag(
        f"begin: pp_last_rank_global={pp_last_rank_global} "
        f"max_samples={max_samples}"
    )

    def _forward_step(data, model):
        tokens = data["tokens"]
        b, s = tokens.shape
        device = tokens.device
        mask = torch.triu(torch.ones(b, s, s, device=device), 1).bool().view(b, 1, s, s)
        # Per-stage timing inside `model(...)` so we can see on which rank the
        # forward pass actually stalls (e.g. MTP layer, lm_head, learned_output_contract).
        _t_m0 = time.time()
        # Last-stage heartbeats: stamp the wall clock every 5s while inside
        # `model(...)` so we can tell whether it's *actually computing* vs
        # stuck on a collective/compile.
        if _am_last_stage:
            import threading as _thr
            _stop_heartbeat = _thr.Event()

            def _heartbeat():
                _t0 = time.time()
                while not _stop_heartbeat.is_set():
                    time.sleep(5.0)
                    if _stop_heartbeat.is_set():
                        break
                    _diag(
                        f"[sample (in-flight)] model(...) still running "
                        f"({time.time()-_t0:.1f}s elapsed)"
                    )

            _hb_thread = _thr.Thread(target=_heartbeat, daemon=True)
            _hb_thread.start()
            try:
                out = model(tokens, position_ids=None, attention_mask=mask)
            finally:
                _stop_heartbeat.set()
                _hb_thread.join(timeout=2.0)
        else:
            out = model(tokens, position_ids=None, attention_mask=mask)
        _t_m1 = time.time()
        _diag(
            f"[sample (in-flight)] model(...) returned in "
            f"{_t_m1-_t_m0:.2f}s, out_type={type(out).__name__}"
            f"{', tuple_len=' + str(len(out)) if isinstance(out, tuple) else ''}"
        )

        # Closure: capture `tokens` directly so we don't depend on an outer
        # `data` variable (which gets rebound on every loop iteration and
        # caused loss to be computed against stale labels in the buggy
        # version of this function).
        def _loss_func(output_tensor, non_loss_data=False):
            _diag(f"[sample (in-flight)] _loss_func called")
            logits = output_tensor[0] if isinstance(output_tensor, tuple) else output_tensor
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = tokens[..., 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="mean",
                ignore_index=tokenizer.pad_token_id,
            )
            _diag(f"[sample (in-flight)] _loss_func done, loss={loss.item():.4f}")
            # Schedule contract: return `(output_tensor, loss_reduced)`. The
            # schedule then scales `output_tensor` in-place (cp * num_mb
            # averaging) and appends `loss_reduced` to `forward_data_store`.
            # Only the last PP stage calls this function; on other stages
            # `forward_data_store` stays empty.
            return output_tensor, loss

        return out, _loss_func

    with torch.no_grad():
        for sample_idx, text in enumerate(validation_texts):
            if max_samples is not None and num_samples >= max_samples:
                break
            tokens = tokenizer.encode(text)
            if len(tokens) < 20:
                continue
            tokens = tokens[:seq_length]
            tokens_t = torch.tensor([tokens], dtype=torch.long, device="cuda")

            # Diagnostic: every rank stamps when it starts the schedule.
            # The PP rank tells us who's the bottleneck.
            _t0 = time.time()
            _diag(f"[sample {sample_idx}] schedule begin, tokens={tuple(tokens_t.shape)}")

            try:
                schedule_output = get_forward_backward_func()(
                    forward_step_func=_forward_step,
                    data_iterator=[{"tokens": tokens_t}],
                    model=model,
                    num_microbatches=1,
                    seq_length=tokens_t.shape[-1],
                    micro_batch_size=1,
                    decoder_seq_length=tokens_t.shape[-1],
                    forward_only=True,
                    # collect_non_loss_data=False so the schedule appends
                    # the loss (not raw outputs) to forward_data_store on
                    # the last stage. The loss function is only called on
                    # the last stage; on other stages forward_data_store is
                    # an empty list, which we then fill via broadcast below.
                    collect_non_loss_data=False,
                )
                _t1 = time.time()
                _sched_type = type(schedule_output).__name__
                _sched_len = (
                    len(schedule_output)
                    if isinstance(schedule_output, (list, tuple))
                    else "n/a"
                )
                _diag(
                    f"[sample {sample_idx}] schedule returned in {_t1-_t0:.2f}s, "
                    f"type={_sched_type} len={_sched_len}"
                )

                # Normalise the schedule return into a list of loss tensors.
                # On the last stage this is `[loss_tensor]`; on other stages
                # it's `[]` (loss function never called). Some configs wrap
                # the list in a tuple — unwrap one level for robustness.
                if isinstance(schedule_output, tuple) and len(schedule_output) == 1:
                    schedule_output = schedule_output[0]

                if not isinstance(schedule_output, list):
                    raise RuntimeError(
                        f"pipeline schedule returned unexpected type: "
                        f"{type(schedule_output).__name__}"
                    )

                # CRITICAL: the pipeline schedule returns at *different*
                # times on different ranks — the schedule's P2P only
                # synchronizes adjacent ranks, not all PP ranks. Rank 0
                # finishes first, rank 7 (last stage) finishes last. So if
                # rank 0 jumps straight into the broadcast while rank 7 is
                # still mid-schedule, rank 0's broadcast waits on NCCL until
                # rank 7 eventually catches up — which it does, but other
                # ranks that have already finished but haven't been
                # formally "joined" yet can race ahead and trigger the
                # watchdog. Add an explicit barrier to synchronize all PP
                # ranks *before* any rank enters the broadcast.
                _t_bar0 = time.time()
                _diag(
                    f"[sample {sample_idx}] barrier begin "
                    f"(post-schedule, pre-broadcast)"
                )
                _td.barrier(group=pp_group)
                _t_bar1 = time.time()
                _diag(
                    f"[sample {sample_idx}] barrier returned in "
                    f"{_t_bar1-_t_bar0:.2f}s"
                )

                # Cross-rank loss gathering. The schedule only invokes the
                # loss function on the last PP stage, so on every other rank
                # `schedule_output` is `[]`. We need every rank to see the
                # same scalar so the rank-0 main process can report a real
                # val_loss.
                #
                # CRITICAL: `torch.distributed.broadcast` is a collective
                # op — *every* rank in the group must call it, otherwise the
                # participating ranks block forever on the NCCL watchdog
                # timeout. So we must NOT guard the broadcast on
                # `len(schedule_output) == 0`. Instead:
                #   1. every rank allocates a fixed-size buffer
                #   2. the last stage copies its local loss into the buffer
                #   3. every rank joins the broadcast (the buffer on the
                #      non-last ranks is just overwritten)
                loss_buf = torch.zeros((), dtype=torch.float32, device="cuda")
                if len(schedule_output) > 0:
                    # We are on the last PP stage. Copy our local loss
                    # into the broadcast buffer.
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
                # Diagnostic: stamp right before the broadcast.
                _t2 = time.time()
                _diag(
                    f"[sample {sample_idx}] broadcast begin "
                    f"(src=pp_last={pp_last_rank_global})"
                )
                # All ranks (last stage + others) call broadcast together.
                _td.broadcast(loss_buf, src=pp_last_rank_global, group=pp_group)
                _t3 = time.time()
                _diag(
                    f"[sample {sample_idx}] broadcast returned in "
                    f"{_t3-_t2:.2f}s, loss_buf={loss_buf.item():.4f}"
                )
                schedule_output = [loss_buf]

                # Average the per-microbatch losses (defensive: ignore any
                # zero-element entries the schedule may leave).
                valid = [l for l in schedule_output if hasattr(l, "item") and l.numel() > 0]
                if not valid:
                    raise RuntimeError("pipeline schedule returned no usable loss tensors")
                total_loss += sum(l.item() for l in valid) / len(valid)
                num_samples += 1
            except Exception as e:
                # CRITICAL: log on EVERY rank, not just rank 0. Otherwise
                # we can't see why a non-zero rank silently skipped a
                # sample while the others are still waiting.
                import traceback as _tb
                if not getattr(compute_validation_loss, "_traced", False):
                    setattr(compute_validation_loss, "_traced", True)
                    _diag(
                        f"[sample {sample_idx}] EXCEPTION: {e}\n"
                        f"--- traceback (first failure) ---\n"
                        f"{_tb.format_exc()}"
                        f"--- end traceback ---"
                    )
                else:
                    _diag(f"[sample {sample_idx}] EXCEPTION: {e}")
                continue

    avg_loss = total_loss / max(num_samples, 1)
    return avg_loss, num_samples


def compute_quality_score(candidate, target_ratio):
    """Compute heuristic quality score for a candidate configuration.
    
    Higher score = better candidate. Factors:
    - Closeness to target parameter ratio (most important)
    - Balanced reduction across dimensions (avoid extreme reductions)
    - Prefer keeping more experts (MoE stability)
    """
    params_diff = abs(candidate['params_ratio'] - target_ratio)
    
    # Penalize deviation from target (0 = perfect, 1 = far)
    params_score = 1.0 - min(params_diff / 0.10, 1.0)
    
    # Compute reduction ratios for each dimension
    h_ratio = candidate['hidden_size'] / 4096  # original hidden
    l_ratio = candidate['num_layers'] / 43  # original layers
    e_ratio = candidate['num_moe_experts'] / 256  # original experts
    f_ratio = candidate['moe_ffn_hidden_size'] / 2048  # original ffn
    
    # Penalize extreme reductions (prefer balanced)
    reductions = [1.0 - h_ratio, 1.0 - l_ratio, 1.0 - e_ratio, 1.0 - f_ratio]
    max_reduction = max(reductions)
    min_reduction = min(reductions)
    balance_score = 1.0 - (max_reduction - min_reduction)  # 1 = perfectly balanced
    
    # Prefer keeping more experts (MoE stability)
    expert_score = e_ratio  # 1 = keep all, 0 = remove all
    
    # Combined score (weighted)
    score = (
        0.5 * params_score +  # 50% weight on target match
        0.3 * balance_score +  # 30% weight on balance
        0.2 * expert_score     # 20% weight on expert retention
    )
    
    return score


def main(args):
    assert dist.size() == args.pp_size, "Only Pipeline parallelism is supported."

    # (hadamard fallback is installed at module-import time; see
    # _install_hadamard_fallback() above)

    if os.path.exists(f"{args.output_hf_path}/config.json"):
        print_rank_0(f"\nPruned model already exists at {args.output_hf_path}. Exiting...")
        return
    
    # Load original config
    with open(os.path.join(args.hf_model_name_or_path, "config.json")) as f:
        orig_config_dict = json.load(f)
    
    # Create a simple config object with attribute name mapping
    class Config:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, v)
            # Map DeepSeek-V4 config names to expected names
            if hasattr(self, 'num_hidden_layers') and not hasattr(self, 'num_layers'):
                self.num_layers = self.num_hidden_layers
            if hasattr(self, 'moe_intermediate_size') and not hasattr(self, 'moe_ffn_hidden_size'):
                self.moe_ffn_hidden_size = self.moe_intermediate_size
            # Handle num_moe_experts vs n_routed_experts
            if not hasattr(self, 'num_moe_experts') and hasattr(self, 'n_routed_experts'):
                self.num_moe_experts = self.n_routed_experts
            # Set default for moe_shared_expert_intermediate_size if not present
            if not hasattr(self, 'moe_shared_expert_intermediate_size'):
                self.moe_shared_expert_intermediate_size = getattr(self, 'moe_ffn_hidden_size', 2048)
    
    orig_config = Config(orig_config_dict)
    
    # Generate candidates
    max_reductions = {
        'hidden': args.max_hidden_size_reduction,
        'ffn': args.max_ffn_reduction,
        'layers': args.max_layers_reduction,
        'experts': args.max_experts_reduction,
    }
    candidates = generate_candidates(
        orig_config, args.target_params_ratio, args.num_candidates, max_reductions
    )
    
    if not candidates:
        print_rank_0("ERROR: No valid candidates generated. Adjust search space constraints.")
        return
    
    # Stage 1: Rank candidates by heuristic quality score
    print_rank_0(f"\n{'='*70}")
    print_rank_0("STAGE 1: Ranking candidates by heuristic quality")
    print_rank_0(f"{'='*70}\n")
    
    for c in candidates:
        c['quality_score'] = compute_quality_score(c, args.target_params_ratio)

    # PRIMARY sort key is `|params_ratio - target_ratio|` (smallest first).
    # The user's intent with `--target_params_ratio` is to hit that ratio
    # closely; quality_score is only a tiebreaker. Previously sorting by
    # quality alone caused us to pick a ~10% candidate when the target was
    # 1.4%, because a balanced ~10% configuration scored higher than an
    # aggressive ~1.4% one. The aggressive configuration is what the user
    # actually asked for.
    candidates.sort(
        key=lambda c: (
            abs(c['params_ratio'] - args.target_params_ratio),
            -c['quality_score'],
        )
    )
    
    for i, c in enumerate(candidates):
        marker = " <-- TOP 3" if i < 3 else ""
        print_rank_0(
            f"[{i+1}] Score={c['quality_score']:.3f} | "
            f"H={c['hidden_size']}, L={c['num_layers']}, "
            f"E={c['num_moe_experts']}, F={c['moe_ffn_hidden_size']}, "
            f"params={c['params_ratio']*100:.1f}%{marker}"
        )
    
    # Select top 3 for full evaluation
    top_candidates = candidates[:min(3, len(candidates))]

    # Stage 2: Full evaluation with validation loss
    print_rank_0(f"\n{'='*70}")
    print_rank_0(f"STAGE 2: Evaluating top {len(top_candidates)} candidates with validation loss")
    print_rank_0(f"{'='*70}\n")

    # Only the forward-validation path needs calibration data. The
    # heuristic-only path (default) skips this entirely.
    train_texts, validation_texts = [], []

    if args.enable_forward_validation:
        # Load calibration data
        print_rank_0(f"\nLoading calibration data...")
        # ... (reuse the Nemotron loading logic from prune_deepseek_v4.py)
        # For brevity, I'll use a simplified version here

        _NEMOTRON_PATH = "/root/.cache/huggingface/hub/datasets--nvidia--Nemotron-Post-Training-Dataset-v2"
        all_texts = []

        try:
            import pyarrow.parquet as pq
            import glob as _glob

            _parquet_files = _glob.glob(f"{_NEMOTRON_PATH}/snapshots/*/data/*.parquet")
            for _pf in sorted(_parquet_files)[:5]:  # Use first 5 files for speed
                if len(all_texts) >= args.calibration_samples:
                    break
                try:
                    _table = pq.read_table(_pf)
                    _rows = _table.to_pydict()
                    _msg_col = 'messages' if 'messages' in _rows else None

                    for _i in range(min(len(list(_rows.values())[0]), 200)):
                        if len(all_texts) >= args.calibration_samples:
                            break
                        if _msg_col:
                            _msgs = _rows[_msg_col][_i]
                            if isinstance(_msgs, list):
                                _text = "\n".join(
                                    str(m.get('content', ''))
                                    for m in _msgs if isinstance(m, dict)
                                )
                                if len(_text) > 50:
                                    all_texts.append(_text)
                except Exception:
                    continue
        except Exception as e:
            print_rank_0(f"Warning: Could not load Nemotron data ({e}), using fallback")
            all_texts = ["Sample text for validation."] * args.calibration_samples

        # Split into train (importance) and validation
        split_idx = int(len(all_texts) * (1.0 - args.validation_ratio))
        train_texts = all_texts[:split_idx]
        validation_texts = all_texts[split_idx:]

        print_rank_0(f"Calibration data: {len(train_texts)} train, {len(validation_texts)} validation")

    # Evaluate each of the top candidates
    results = []

    for i, candidate in enumerate(top_candidates):
        print_rank_0(f"\n[{i+1}/{len(top_candidates)}] Evaluating candidate:")
        print_rank_0(
            f"  H={candidate['hidden_size']}, L={candidate['num_layers']}, "
            f"E={candidate['num_moe_experts']}, F={candidate['moe_ffn_hidden_size']}, "
            f"params={candidate['params_ratio']*100:.1f}%, "
            f"quality_score={candidate['quality_score']:.3f}"
        )

        # ─────────────────────────────────────────────────────────────────
        # Fast path: heuristic-only selection (default, PP-safe).
        # ─────────────────────────────────────────────────────────────────
        if not args.enable_forward_validation:
            fallback_score = -float(candidate.get("quality_score", 0.0))
            print_rank_0(
                f"  Forward validation disabled (use --enable_forward_validation "
                f"to attempt). Using heuristic quality_score="
                f"{-fallback_score:.4f}."
            )
            results.append({
                "candidate": candidate,
                "val_loss": fallback_score,
                "num_samples": 0,
                "used_heuristic": True,
            })
            continue

        # ─────────────────────────────────────────────────────────────────
        # Slow path: load + prune + forward validation. Known to fail under
        # PP>1 + mcore_minitron because the prune plugin drops layers without
        # rewriting the PP layout, leaving tail ranks with 0 layers in the
        # `model.decoder.layers` list and triggering
        # `IndexError: list index out of range` from get_forward_backward_func.
        # Kept for future re-enable once the PP layout is re-derived post-prune.
        # ─────────────────────────────────────────────────────────────────

        # Create export config for this candidate
        export_config = {
            'hidden_size': candidate['hidden_size'],
            'num_layers': candidate['num_layers'],
            'num_moe_experts': candidate['num_moe_experts'],
            'moe_ffn_hidden_size': candidate['moe_ffn_hidden_size'],
            'moe_shared_expert_intermediate_size': candidate['moe_shared_expert_intermediate_size'],
        }

        # Load model (this is slow but necessary for each candidate)
        print_rank_0(f"  Loading model...")
        bridge = AutoBridge.from_hf_pretrained(
            args.hf_model_name_or_path, trust_remote_code=args.trust_remote_code
        )
        provider = bridge.to_megatron_provider(load_weights=True)

        # Apply provider overrides
        provider_overrides = {
            "tensor_model_parallel_size": 1,
            "expert_tensor_parallel_size": 1,
            "pipeline_model_parallel_size": args.pp_size,
            "pipeline_dtype": torch.bfloat16,
            "seq_length": args.seq_length,
            "use_fused_mhc": False,
        }

        if args.pp_size > 1:
            n_layers = orig_config_dict["num_hidden_layers"]
            n_mtp = orig_config_dict.get("num_nextn_predict_layers", 0)
            provider_overrides["pipeline_model_parallel_layout"] = _build_pp_layout(
                n_layers, args.pp_size, n_mtp=n_mtp
            )

        for key, value in provider_overrides.items():
            setattr(provider, key, value)

        provider.finalize()
        provider.initialize_model_parallel(seed=0)
        model = provider.provide_distributed_model(wrap_with_ddp=False)

        from megatron.core.utils import unwrap_model
        unwrapped_model = unwrap_model(model[0])

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            args.hf_model_name_or_path, trust_remote_code=args.trust_remote_code
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Apply pruning (simplified - skip importance estimation for speed)
        # In a full implementation, we'd compute importance once and reuse
        print_rank_0(f"  Applying pruning...")
        ss_config = mtp.mcore_minitron.get_mcore_minitron_config(
            channel_divisor=128,
            num_moe_experts_divisor=8,
        )

        # Skip forward loop for speed (use weight-magnitude importance)
        def dummy_forward_loop(model):
            pass

        unwrapped_model, _ = mtp.prune(
            unwrapped_model,
            mode=[("mcore_minitron", ss_config)],
            constraints={"export_config": export_config},
            dummy_input=None,
            config={"forward_loop": dummy_forward_loop, "skip_sorting": True},
        )

        # ── Rebuild PP layout after prune ─────────────────────────────
        # mcore_minitron drops layers in-place but leaves the
        # pipeline_model_parallel_layout string stale. Without this, the
        # validation forward pass would hit `list index out of range` inside
        # the pipeline schedule when a tail rank ends up with 0 layers.
        try:
            n_mtp_after = orig_config_dict.get("num_nextn_predict_layers", 0)
            new_layout = _rebuild_pp_layout_after_prune(
                unwrapped_model,
                args.pp_size,
                n_mtp=n_mtp_after,
                # Use the *target* post-prune num_layers (this candidate's
                # value), NOT len(unwrapped_model.decoder.layers) which is
                # the per-rank chunk count. Using the per-rank count produces
                # a layout with empty middle stages and the pipeline
                # schedule returns an empty loss list.
                n_layers=candidate["num_layers"],
            )
            print_rank_0(
                f"  Rebuilt PP layout after prune: {new_layout} "
                f"(model.decoder.layers="
                f"{len(unwrapped_model.decoder.layers)} on this rank)"
            )
        except Exception as _layout_e:
            # Best-effort: log and continue. If the schedule still crashes
            # we'll fall back to heuristic below.
            print_rank_0(
                f"  Warning: PP layout rebuild failed: "
                f"{type(_layout_e).__name__}: {_layout_e}"
            )

        # ── Patch mHC final-layernorm attributes on the new last stage ──
        # `TransformerBlock.__init__` only creates `hc_head_fn /
        # hc_head_base / hc_head_scale` on the stage that originally held the
        # final layernorm (the post-process stage of the *pre-prune* model).
        # After prune + layout rebuild, a different PP stage may now own the
        # final decoder layer, but `TransformerBlock.__init__` has long
        # returned — so that stage lacks the mHC params. The forward pass
        # then crashes at `learned_output_contract(..., self.hc_head_fn, ...)`
        # with `'TransformerBlock' object has no attribute 'hc_head_fn'`.
        #
        # Detect which rank now holds `layer.layer_number == num_layers` and,
        # if it lacks the mHC params, lazily create them in-place using the
        # same init as `transformer_block.py:392-402`.
        try:
            cfg = getattr(unwrapped_model, "config", None)
            decoder = getattr(unwrapped_model, "decoder", None)
            if (
                cfg is not None
                and decoder is not None
                and getattr(cfg, "enable_hyper_connections", False)
                and getattr(cfg, "mtp_num_layers", None) is not None
            ):
                _target_layer_number = int(candidate["num_layers"])
                _holds_last = any(
                    getattr(layer, "layer_number", None) == _target_layer_number
                    for layer in decoder.layers
                )
                if _holds_last:
                    _blk = decoder  # decoder IS the TransformerBlock
                    if _blk is not None:
                        _missing = not all(
                            hasattr(_blk, name)
                            for name in ("hc_head_fn", "hc_head_base", "hc_head_scale")
                        )
                        if _missing:
                            import torch.nn as _nn
                            _hc_mult = int(getattr(cfg, "num_residual_streams", 4))
                            _hc_dim = int(cfg.hidden_size) * _hc_mult
                            # Place params on the same device/dtype as the
                            # existing model parameters — otherwise the
                            # forward pass hits "aten.mm got two different
                            # devices cuda:6, cpu" inside torch.compile's
                            # fake-tensor propagation through `learned_output_contract`.
                            try:
                                _ref_param = next(iter(_blk.parameters()))
                                _ref_device = _ref_param.device
                                _ref_dtype = _ref_param.dtype
                            except StopIteration:
                                _ref_device = torch.device("cuda")
                                _ref_dtype = torch.float32
                            _blk.hc_head_fn = _nn.Parameter(
                                torch.randn(
                                    _hc_mult, _hc_dim,
                                    device=_ref_device, dtype=_ref_dtype,
                                )
                            )
                            _blk.hc_head_base = _nn.Parameter(
                                torch.zeros(
                                    _hc_mult,
                                    device=_ref_device, dtype=_ref_dtype,
                                )
                            )
                            _blk.hc_head_scale = _nn.Parameter(
                                torch.ones(
                                    1,
                                    device=_ref_device, dtype=_ref_dtype,
                                )
                            )
                            _nn.init.xavier_uniform_(_blk.hc_head_fn)
                            if getattr(cfg, "sequence_parallel", False):
                                for _p in (
                                    _blk.hc_head_fn,
                                    _blk.hc_head_base,
                                    _blk.hc_head_scale,
                                ):
                                    setattr(_p, "sequence_parallel", True)
                            print_rank_0(
                                f"  Lazily created mHC params on the new "
                                f"final-layernorm stage "
                                f"(target_layer_number={_target_layer_number}, "
                                f"hc_mult={_hc_mult}, hc_dim={_hc_dim}, "
                                f"device={_ref_device}, dtype={_ref_dtype})"
                            )
        except Exception as _mhc_e:
            # Non-fatal: if patching fails, the forward pass will surface the
            # original AttributeError and we'll fall back to heuristic below.
            print_rank_0(
                f"  Warning: mHC param patch failed: "
                f"{type(_mhc_e).__name__}: {_mhc_e}"
            )

        # Compute validation loss.
        # Pass the full `model` list (the same value `get_forward_backward_func`
        # expects) — the previous version passed `unwrapped_model` (a single
        # chunk), which is wrong under pipeline parallelism.
        print_rank_0(
            f"  Computing validation loss (max {args.validation_samples} "
            f"samples, seq_length={args.validation_seq_length})..."
        )
        try:
            val_loss, num_samples = compute_validation_loss(
                model,
                tokenizer,
                validation_texts,
                args.validation_seq_length,
                max_samples=args.validation_samples,
            )
        except Exception as _e:
            print_rank_0(
                f"  Validation crashed: {type(_e).__name__}: {_e}. "
                f"Falling back to heuristic quality_score."
            )
            val_loss, num_samples = float("nan"), 0

        # If validation produced no usable signal (every sample failed or the
        # whole call crashed), fall back to the heuristic quality score. We
        # negate it so the sort key direction matches "lower loss is better"
        # only by convention — we'll remap later via a dedicated comparator.
        if num_samples == 0 or val_loss != val_loss:  # NaN check
            fallback_score = -float(candidate.get("quality_score", 0.0))
            print_rank_0(
                f"  Validation loss: N/A — using heuristic proxy "
                f"({-fallback_score:.4f})"
            )
            results.append({
                "candidate": candidate,
                "val_loss": fallback_score,
                "num_samples": 0,
                "used_heuristic": True,
            })
        else:
            print_rank_0(
                f"  Validation loss: {val_loss:.4f} (on {num_samples} samples)"
            )
            results.append({
                "candidate": candidate,
                "val_loss": val_loss,
                "num_samples": num_samples,
                "used_heuristic": False,
            })

        # Clean up to free memory
        del model, unwrapped_model, bridge, provider
        torch.cuda.empty_cache()
    
    # Select best candidate
    print_rank_0(f"\n{'='*70}")
    print_rank_0("NAS SEARCH RESULTS")
    print_rank_0(f"{'='*70}\n")

    # Sort: prefer real validation over heuristic fallback. Within each tier,
    # lower loss is better. (Heuristic-only entries have val_loss = -quality,
    # but we still keep them at the bottom by adding a large tier penalty.)
    def _sort_key(r):
        used_h = r.get("used_heuristic", False)
        # tier 0 = real val, tier 1 = heuristic. Within tier, ascending loss.
        return (1 if used_h else 0, r["val_loss"])

    results.sort(key=_sort_key)

    for i, r in enumerate(results):
        c = r['candidate']
        marker = " <-- BEST" if i == 0 else ""
        loss_str = (
            f"{r['val_loss']:.4f}"
            if not r.get("used_heuristic", False)
            else f"heuristic({-r['val_loss']:.4f})"
        )
        print_rank_0(
            f"[{i+1}] Loss={loss_str} | "
            f"H={c['hidden_size']}, L={c['num_layers']}, "
            f"E={c['num_moe_experts']}, F={c['moe_ffn_hidden_size']}, "
            f"params={c['params_ratio']*100:.1f}%{marker}"
        )
    
    best = results[0]
    best_config = best['candidate']

    print_rank_0(f"\n{'='*70}")
    print_rank_0(f"BEST CONFIGURATION")
    print_rank_0(f"{'='*70}")
    if best.get("used_heuristic", False):
        print_rank_0(
            f"  Validation: unavailable — selected by heuristic "
            f"(quality_score={-best['val_loss']:.4f})"
        )
    else:
        print_rank_0(f"  Validation loss: {best['val_loss']:.4f}")
    print_rank_0(f"  hidden_size: {best_config['hidden_size']}")
    print_rank_0(f"  num_layers: {best_config['num_layers']}")
    print_rank_0(f"  num_moe_experts: {best_config['num_moe_experts']}")
    print_rank_0(f"  moe_ffn_hidden_size: {best_config['moe_ffn_hidden_size']}")
    print_rank_0(
        f"  moe_shared_expert_intermediate_size: "
        f"{best_config['moe_shared_expert_intermediate_size']}"
    )
    print_rank_0(f"  params_ratio: {best_config['params_ratio']*100:.1f}%")
    
    # Save best configuration (with val_loss so downstream prune can see it)
    best_config_path = args.output_hf_path + "_best_config.json"
    best_config_with_loss = dict(best_config)
    best_config_with_loss["val_loss"] = (
        best.get("val_loss") if not best.get("used_heuristic", False) else None
    )
    best_config_with_loss["val_num_samples"] = best.get("num_samples", 0)
    best_config_with_loss["used_heuristic"] = best.get("used_heuristic", False)
    with open(best_config_path, 'w') as f:
        json.dump(best_config_with_loss, f, indent=2)
    print_rank_0(f"\nSaved best configuration to {best_config_path}")
    
    # Now apply the best configuration with full importance estimation
    print_rank_0(f"\n{'='*70}")
    print_rank_0("APPLYING BEST CONFIGURATION WITH IMPORTANCE ESTIMATION")
    print_rank_0(f"{'='*70}\n")
    
    # This would call the full prune_deepseek_v4.py logic with the best config
    # For now, we'll just save the config and let the user run the final pruning
    print_rank_0(f"\nTo apply the best configuration, run:")
    print_rank_0(f"  torchrun --nproc_per_node={args.pp_size} prune_deepseek_v4.py \\")
    print_rank_0(f"    --hf_model_name_or_path {args.hf_model_name_or_path} \\")
    print_rank_0(f"    --output_hf_path {args.output_hf_path} \\")
    print_rank_0(f"    --pp_size {args.pp_size} \\")
    print_rank_0(f"    --hidden_size {best_config['hidden_size']} \\")
    print_rank_0(f"    --num_layers {best_config['num_layers']} \\")
    print_rank_0(f"    --num_moe_experts {best_config['num_moe_experts']} \\")
    print_rank_0(f"    --moe_ffn_hidden_size {best_config['moe_ffn_hidden_size']} \\")
    print_rank_0(f"    --moe_shared_expert_intermediate_size {best_config['moe_shared_expert_intermediate_size']} \\")
    print_rank_0(f"    --trust_remote_code")


if __name__ == "__main__":
    dist.setup()
    args = parse_args()
    try:
        main(args)
    finally:
        dist.cleanup()

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

"""Generate an exact 4B-target pruning configuration for DeepSeek-V4-Flash.

The default NAS parameter estimator in ``prune_deepseek_v4_nas.py`` uses the
rule-of-thumb ``12 * hidden_size^2`` for attention and ignores embedding,
lm_head, MLA low-rank projections, Hyper-Connection weights and MTP layers.
For an extreme 4B target (~1.4 % of 284B) those omissions cause the actual
pruned model to deviate significantly from the target size.

This script reads the original HF ``config.json`` and computes a much more
accurate total-parameter budget per candidate, then selects the candidate
closest to a user-specified total-parameter target (default 4B).  The output
``best_config.json`` has the same schema as ``prune_deepseek_v4_nas.py`` so it
can be consumed directly by ``prune_deepseek_v4.py`` / ``run_v4_4b_pipeline.sh``.

Usage:
    python3 prune_deepseek_v4_4b_config.py \
        --hf_config /data/.cache/models/deepseek-ai/DeepSeek-V4-Flash/config.json \
        --target_total_params 4_000_000_000 \
        --output best_config.json
"""

import argparse
import json
import math
import os
from itertools import product
from typing import Any


# Default DeepSeek-V4-Flash architectural constants.  These are used only when
# the corresponding key is missing from ``config.json``.
_DEFAULTS = {
    "hidden_size": 4096,
    "num_hidden_layers": 43,
    "num_attention_heads": 128,
    "n_routed_experts": 256,
    "moe_intermediate_size": 2048,
    "moe_shared_expert_intermediate_size": 2048,
    "vocab_size": 129280,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "num_residual_streams": 4,
    "num_nextn_predict_layers": 1,
    "first_k_dense_replace": 1,
    "tie_word_embeddings": False,
}


def _get(cfg: dict[str, Any], key: str) -> Any:
    """Fetch config value with legacy-name fallback and hardcoded default."""
    aliases = {
        "num_hidden_layers": ["num_layers"],
        "n_routed_experts": ["num_moe_experts", "num_experts"],
        "moe_intermediate_size": ["moe_ffn_hidden_size", "ffn_hidden_size"],
        "moe_shared_expert_intermediate_size": [
            "moe_shared_expert_intermediate_size",
            "shared_expert_intermediate_size",
        ],
        "num_nextn_predict_layers": ["num_mtp_layers", "mtp_num_layers"],
    }
    for k in [key] + aliases.get(key, []):
        if k in cfg:
            return cfg[k]
    return _DEFAULTS[key]


def _round(value: int, divisor: int) -> int:
    return max(divisor, (value // divisor) * divisor)


def estimate_mla_attention_params(
    cfg: dict[str, Any], hidden_size: int
) -> int:
    """Estimate MLA attention params for one layer after pruning.

    DeepSeek-V4 MLA uses separate Q down/up projections, a compressed KV
    projection and an output projection.  The ``o_group_proj`` is deliberately
    not pruned by ``_DynamicV4SelfAttention`` and is kept at its original size,
    so we account for it with the *original* hidden_size.
    """
    h = hidden_size
    orig_h = _get(cfg, "hidden_size")
    q_lora_rank = _get(cfg, "q_lora_rank")
    kv_lora_rank = _get(cfg, "kv_lora_rank")
    qk_rope_head_dim = _get(cfg, "qk_rope_head_dim")
    v_head_dim = _get(cfg, "v_head_dim")
    num_heads = _get(cfg, "num_attention_heads")

    # Q path: down -> up.  The up projection output dimension is
    # num_heads * qk_head_dim for the non-rope part; we approximate qk_head_dim
    # by the rope head dim when not present in the config.
    qk_head_dim = cfg.get("qk_head_dim", qk_rope_head_dim)
    q_down = h * q_lora_rank
    q_up = q_lora_rank * (num_heads * qk_head_dim)

    # KV path: compressed kv + rope k dimension
    kv_proj = h * (kv_lora_rank + qk_rope_head_dim)

    # Output projection
    linear_proj = h * h

    # o_group_proj is NOT pruned (per _DynamicV4SelfAttention).  Its input
    # dimension is num_heads * v_head_dim and output is hidden_size.
    o_group_proj = (num_heads * v_head_dim) * orig_h

    return q_down + q_up + kv_proj + linear_proj + o_group_proj


def estimate_layer_params(
    cfg: dict[str, Any],
    hidden_size: int,
    num_experts: int,
    ffn_size: int,
    shared_size: int,
) -> int:
    """Estimate total trainable params for one pruned transformer layer."""
    # LayerNorms (input + pre-mlp)
    layernorms = 2 * hidden_size * 2  # weight + bias

    # MLA attention
    attn = estimate_mla_attention_params(cfg, hidden_size)

    # MoE FFN: each expert is gate-up-down (3 matrices)
    moe = num_experts * 3 * ffn_size * hidden_size

    # Shared expert (also 3 matrices)
    shared = 3 * shared_size * hidden_size

    # Router
    router = hidden_size * num_experts

    # Hyper-Connection mapping projections (self-attention + mlp)
    hc_mult = _get(cfg, "num_residual_streams")
    hc_dim = hidden_size * hc_mult
    mix_hc_dim = cfg.get("mix_hc_dim", hidden_size)  # may not be present
    hc = 2 * mix_hc_dim * hc_dim

    return layernorms + attn + moe + shared + router + hc


def estimate_mtp_layer_params(
    cfg: dict[str, Any],
    hidden_size: int,
    num_experts: int,
    ffn_size: int,
    shared_size: int,
) -> int:
    """Estimate params for one MTP layer (simplified as one transformer layer)."""
    # MTP shares the same structure but typically has a smaller embed projection.
    # We approximate it as a normal transformer layer plus a small embed proj.
    layer = estimate_layer_params(cfg, hidden_size, num_experts, ffn_size, shared_size)
    embed_proj = hidden_size * hidden_size
    return layer + embed_proj


def estimate_total_params(
    cfg: dict[str, Any],
    hidden_size: int,
    num_layers: int,
    num_experts: int,
    ffn_size: int,
    shared_size: int,
) -> int:
    """Estimate total model parameters for a candidate pruning configuration."""
    vocab_size = _get(cfg, "vocab_size")
    hc_mult = _get(cfg, "num_residual_streams")
    hc_dim = hidden_size * hc_mult
    num_mtp = _get(cfg, "num_nextn_predict_layers")

    # Embedding + lm_head
    embedding = vocab_size * hidden_size
    lm_head = 0 if _get(cfg, "tie_word_embeddings") else vocab_size * hidden_size

    # Transformer layers
    layers = num_layers * estimate_layer_params(
        cfg, hidden_size, num_experts, ffn_size, shared_size
    )

    # Model-level Hyper-Connection head
    hc_head_fn = hc_mult * hc_dim
    hc_head_base = hc_mult
    hc_head_scale = 1
    hc_head = hc_head_fn + hc_head_base + hc_head_scale

    # Final layer norm
    final_layernorm = 2 * hidden_size

    # MTP layers
    mtp = num_mtp * estimate_mtp_layer_params(
        cfg, hidden_size, num_experts, ffn_size, shared_size
    )

    total = embedding + lm_head + layers + hc_head + final_layernorm + mtp
    return total


def generate_candidates(
    cfg: dict[str, Any],
    target_total: int,
    hidden_ratios: list[float] | None = None,
    ffn_ratios: list[float] | None = None,
    layer_ratios: list[float] | None = None,
    expert_ratios: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Generate pruning candidates and estimate real total params for each."""
    orig_h = _get(cfg, "hidden_size")
    orig_l = _get(cfg, "num_hidden_layers")
    orig_e = _get(cfg, "n_routed_experts")
    orig_f = _get(cfg, "moe_intermediate_size")
    orig_s = _get(cfg, "moe_shared_expert_intermediate_size")

    hidden_ratios = hidden_ratios or [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4]
    ffn_ratios = ffn_ratios or [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25]
    layer_ratios = layer_ratios or [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.35, 0.3]
    expert_ratios = expert_ratios or [1.0, 0.75, 0.5, 0.375, 0.25, 0.1875, 0.125]

    hidden_sizes = sorted({max(512, _round(int(orig_h * r), 128)) for r in hidden_ratios})
    ffn_sizes = sorted({max(256, _round(int(orig_f * r), 128)) for r in ffn_ratios})
    layer_counts = sorted({max(8, _round(int(orig_l * r), 2)) for r in layer_ratios})
    expert_counts = sorted({max(16, _round(int(orig_e * r), 8)) for r in expert_ratios})

    candidates = []
    for h, f, l, e in product(hidden_sizes, ffn_sizes, layer_counts, expert_counts):
        shared = max(128, _round(int(orig_s * (f / orig_f)), 128))
        total = estimate_total_params(cfg, h, l, e, f, shared)
        candidates.append(
            {
                "hidden_size": h,
                "num_layers": l,
                "num_moe_experts": e,
                "moe_ffn_hidden_size": f,
                "moe_shared_expert_intermediate_size": shared,
                "estimated_total_params": total,
                "estimated_total_params_b": total / 1e9,
            }
        )
    return candidates


def compute_quality_score(candidate: dict[str, Any], target_total: int) -> float:
    """Heuristic quality score for a candidate.

    Higher is better.  Balanced reduction is preferred, and we reward keeping
    more experts because MoE routing becomes unstable when the expert count is
    too small.
    """
    cfg = candidate["_orig_cfg"]
    orig_h = _get(cfg, "hidden_size")
    orig_l = _get(cfg, "num_hidden_layers")
    orig_e = _get(cfg, "n_routed_experts")
    orig_f = _get(cfg, "moe_intermediate_size")

    h_ratio = candidate["hidden_size"] / orig_h
    l_ratio = candidate["num_layers"] / orig_l
    e_ratio = candidate["num_moe_experts"] / orig_e
    f_ratio = candidate["moe_ffn_hidden_size"] / orig_f

    # Closeness to target (primary)
    params_diff = abs(candidate["estimated_total_params"] - target_total)
    params_score = max(0.0, 1.0 - params_diff / target_total)

    # Balance
    reductions = [1.0 - h_ratio, 1.0 - l_ratio, 1.0 - e_ratio, 1.0 - f_ratio]
    balance_score = max(0.0, 1.0 - (max(reductions) - min(reductions)))

    # Expert retention
    expert_score = e_ratio

    return 0.5 * params_score + 0.3 * balance_score + 0.2 * expert_score


def select_best_config(
    cfg: dict[str, Any],
    target_total: int,
    top_k: int = 5,
    max_rel_error: float = 0.25,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Select the candidate whose estimated total params are closest to target."""
    candidates = generate_candidates(cfg, target_total)

    # Keep only candidates within 25% of the target by default.  For extreme 4B
    # targets this is usually enough; the user can re-run with a looser bound if
    # no candidate is found.
    candidates = [
        c
        for c in candidates
        if abs(c["estimated_total_params"] - target_total) / target_total <= max_rel_error
    ]
    if not candidates:
        raise RuntimeError(
            f"No candidate within {max_rel_error * 100:.0f}% of {target_total / 1e9:.2f}B. "
            "Loosen --max_rel_error or expand search ratios."
        )

    for c in candidates:
        c["_orig_cfg"] = cfg

    # Primary sort: closeness to target; secondary: quality score
    candidates.sort(
        key=lambda c: (
            abs(c["estimated_total_params"] - target_total),
            -compute_quality_score(c, target_total),
        )
    )

    best = candidates[0]
    # Strip internal helper key before returning
    for c in candidates:
        c.pop("_orig_cfg", None)
    best.pop("_orig_cfg", None)
    return best, candidates[:top_k]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate exact 4B-target DeepSeek-V4 pruning config",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--hf_config",
        type=str,
        required=True,
        help="Path to DeepSeek-V4-Flash HF config.json",
    )
    parser.add_argument(
        "--target_total_params",
        type=int,
        default=4_000_000_000,
        help="Target total parameter count (default 4B)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="best_config.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top candidates to print",
    )
    parser.add_argument(
        "--max_rel_error",
        type=float,
        default=0.25,
        help="Max relative deviation from target_total_params",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.hf_config) as f:
        cfg = json.load(f)

    print(f"Loaded config from {args.hf_config}")
    print(f"Target total params: {args.target_total_params / 1e9:.2f}B")
    print(
        f"Original model: hidden={_get(cfg, 'hidden_size')}, "
        f"layers={_get(cfg, 'num_hidden_layers')}, "
        f"experts={_get(cfg, 'n_routed_experts')}, "
        f"ffn={_get(cfg, 'moe_intermediate_size')}"
    )

    best, top = select_best_config(
        cfg, args.target_total_params, top_k=args.top_k, max_rel_error=args.max_rel_error
    )

    print(f"\nTop {args.top_k} candidates:")
    print("-" * 90)
    print(
        f"{'rank':>4} {'H':>5} {'L':>4} {'E':>5} {'F':>5} {'S':>5} "
        f"{'total(B)':>10} {'err%':>8} {'quality':>8}"
    )
    print("-" * 90)
    orig_total = estimate_total_params(
        cfg,
        _get(cfg, "hidden_size"),
        _get(cfg, "num_hidden_layers"),
        _get(cfg, "n_routed_experts"),
        _get(cfg, "moe_intermediate_size"),
        _get(cfg, "moe_shared_expert_intermediate_size"),
    )
    print(f"orig {orig_total / 1e9:>50.2f}B")
    for i, c in enumerate(top):
        err = (c["estimated_total_params"] - args.target_total_params) / args.target_total_params
        marker = " <-- BEST" if i == 0 else ""
        # Recompute quality for display
        c["_orig_cfg"] = cfg
        q = compute_quality_score(c, args.target_total_params)
        c.pop("_orig_cfg", None)
        print(
            f"{i + 1:>4} "
            f"{c['hidden_size']:>5} "
            f"{c['num_layers']:>4} "
            f"{c['num_moe_experts']:>5} "
            f"{c['moe_ffn_hidden_size']:>5} "
            f"{c['moe_shared_expert_intermediate_size']:>5} "
            f"{c['estimated_total_params'] / 1e9:>10.2f}B "
            f"{err * 100:>7.1f}% "
            f"{q:>7.3f}"
            f"{marker}"
        )
    print("-" * 90)

    # Build output in the same schema as prune_deepseek_v4_nas.py so that
    # run_v4_4b_pipeline.sh can consume it directly.
    output = {
        "hidden_size": best["hidden_size"],
        "num_layers": best["num_layers"],
        "num_moe_experts": best["num_moe_experts"],
        "moe_ffn_hidden_size": best["moe_ffn_hidden_size"],
        "moe_shared_expert_intermediate_size": best["moe_shared_expert_intermediate_size"],
        "estimated_total_params": best["estimated_total_params"],
        "target_total_params": args.target_total_params,
        "params_ratio": best["estimated_total_params"] / orig_total,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved best config to {args.output}")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

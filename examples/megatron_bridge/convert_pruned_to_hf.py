#!/usr/bin/env python3
"""Convert merged pruned DeepSeek-V4 Megatron state_dict to SGLang-compatible safetensors.

Outputs keys in DeepSeek on-disk format (not HF format), which SGLang's
remap_weight_name_to_dpsk_hf_format() converts to internal model names.

Usage:
    python3 convert_pruned_to_hf.py \
        --megatron_ckpt /data/output/v4-flash-full-v4_megatron/pruned_model.pt \
        --hf_config /data/output/v4-flash-full-v4/config.json \
        --output_dir /data/output/v4-flash-full-v4-hf
"""

import argparse
import json
import math
import textwrap
import os
import re

import torch
from safetensors.torch import save_file


def quantize_fp8_blockwise(weight, block_size=128):
    """Quantize a BF16 weight to FP8 (e4m3) with block-wise e8m0 scales.

    Matches the DeepSeek-V4 on-disk format consumed by SGLang's DeepGEMM
    kernels.  Returns ``(weight_fp8, scale_e8m0)`` where:

    * ``weight_fp8`` has the same shape as *weight*, dtype ``float8_e4m3fn``
    * ``scale_e8m0`` has shape ``[M/bs, N/bs]``, dtype ``float8_e8m0fnu``
    """
    M, N = weight.shape
    assert M % block_size == 0 and N % block_size == 0, (
        f"Weight shape [{M}, {N}] must be divisible by block_size {block_size}"
    )
    FP8_MAX = 448.0  # max representable in float8_e4m3fn

    w = weight.float()
    blocks = w.reshape(M // block_size, block_size, N // block_size, block_size)
    amax = blocks.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-12)

    # Continuous scale: real_value = fp8_value * scale
    scale_cont = amax / FP8_MAX  # [Mb, 1, Nb, 1]

    # Round to nearest power of 2 (e8m0 stores exponents only)
    log2_s = torch.log2(scale_cont)
    exp_biased = (log2_s.ceil() + 127).clamp(0, 254)  # e8m0 bias = 127
    scale_pow2 = torch.exp2(exp_biased - 127)

    # Quantize
    w_q = (w.reshape_as(blocks) / scale_pow2).clamp(-FP8_MAX, FP8_MAX)
    weight_fp8 = w_q.reshape(M, N).to(torch.float8_e4m3fn)

    # Store scale as float8_e8m0fnu (power-of-2 values)
    scale_e8m0 = scale_pow2.squeeze().to(torch.float8_e8m0fnu)

    return weight_fp8, scale_e8m0


# MXFP4 (e2m1) quantization — matches SGLang's MXFP4QuantizeUtil exactly.
_MXFP4_E2M1_MAX = 6.0
_MXFP4_BOUNDS = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]


def quantize_mxfp4(weight, block_size=32):
    """Quantize a BF16 weight to MXFP4 (int8-packed e2m1) with e8m0 scales.

    Matches ``sglang.srt.layers.quantization.mxfp4_tensor.MXFP4QuantizeUtil``
    so that the output is byte-compatible with the Marlin MoE kernel used
    by SGLang for DeepSeek V4 expert weights.

    Returns ``(packed_weight, scale)`` where:

    * ``packed_weight``: int8 tensor ``[M, N/2]`` — two FP4 nibbles per byte
      (even-indexed value in low nibble, odd-indexed in high nibble).
    * ``scale``: float8_e8m0fnu tensor ``[M, N/block_size]``.
    """
    M, N = weight.shape
    assert N % block_size == 0, f"N={N} must be divisible by block_size={block_size}"

    orig_shape = (M, N)
    w = weight.float().view(-1, block_size)  # [total_blocks, block_size]

    # Per-block amax → e8m0 scale
    amax = w.abs().amax(dim=-1, keepdim=True)  # [total_blocks, 1]
    descale = amax / _MXFP4_E2M1_MAX
    min_val = torch.tensor(-127.0, device=w.device)
    e8m0_exp = torch.ceil(torch.maximum(torch.log2(descale), min_val))
    # Dequantize scale factor
    w = (w / torch.exp2(e8m0_exp)).view(orig_shape)

    # Cast to FP4 nibbles (sign-magnitude encoding)
    sign = torch.sign(w)
    sign_bit = ((2 - sign) // 2).to(torch.uint8)  # 0=pos, 1=neg
    abs_w = w.abs()
    # Ordinal: count how many bounds the value exceeds
    ord_ = torch.zeros_like(abs_w, dtype=torch.uint8)
    for b in _MXFP4_BOUNDS:
        ord_ = ord_ + (abs_w > b).to(torch.uint8)
    fp4_nibbles = (sign_bit * 0x08 + ord_)  # [M, N]

    # Pack pairs of nibbles: even indices → low nibble, odd → high nibble
    lo = fp4_nibbles[..., 0::2]   # even
    hi = fp4_nibbles[..., 1::2]   # odd
    packed = ((hi.clone() << 4) + lo).to(torch.uint8)  # high | low
    packed = packed.view(torch.int8).reshape(M, N // 2)

    # Scale: biased exponent stored as float8_e8m0fnu
    e8m0_biased = (e8m0_exp + 127).clamp(0, 254).to(torch.uint8)
    scale_out = e8m0_biased.reshape(M, N // block_size).view(torch.float8_e8m0fnu)

    return packed, scale_out


def dequantize_mxfp4(packed_weight, scale, block_size=32):
    """Dequantize MXFP4 (int8-packed e2m1) weights back to BF16.

    Reverses the quantize_mxfp4 operation to unpack expert weights.

    Args:
        packed_weight: int8 tensor [M, N/2] — two FP4 nibbles per byte
        scale: float8_e8m0fnu tensor [M, N/block_size]
        block_size: quantization block size (default 32)

    Returns:
        BF16 tensor [M, N]
    """
    M, N_half = packed_weight.shape
    N = N_half * 2

    # Unpack nibbles: low nibble (even indices), high nibble (odd indices)
    packed_uint8 = packed_weight.view(torch.uint8)
    lo = packed_uint8 & 0x0F  # low nibble
    hi = (packed_uint8 >> 4) & 0x0F  # high nibble

    # Interleave: even indices from lo, odd indices from hi
    fp4_nibbles = torch.zeros(M, N, dtype=torch.uint8, device=packed_weight.device)
    fp4_nibbles[..., 0::2] = lo
    fp4_nibbles[..., 1::2] = hi

    # Decode FP4 sign-magnitude: sign_bit in bit 3, ordinal in bits 0-2
    sign_bit = (fp4_nibbles >> 3) & 0x01
    ordinal = fp4_nibbles & 0x07

    # Map ordinal to FP4 values
    # Ordinal 0-7 maps to: 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
    fp4_values = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                               dtype=torch.float32, device=packed_weight.device)
    abs_w = fp4_values[ordinal.long()]

    # Apply sign: sign_bit=0 means positive, sign_bit=1 means negative
    sign = 1.0 - 2.0 * sign_bit.float()
    w = abs_w * sign

    # Apply scale (dequantize)
    # scale is [M, N/block_size], need to broadcast to [M, N]
    scale_float = scale.view(torch.uint8).float()
    e8m0_exp = scale_float - 127.0  # unbias
    scale_factor = torch.exp2(e8m0_exp)  # [M, N/block_size]

    # Reshape and broadcast scale to [M, N]
    scale_factor = scale_factor.unsqueeze(-1).expand(-1, -1, block_size)
    scale_factor = scale_factor.reshape(M, N)

    w = w * scale_factor

    return w.to(torch.bfloat16)


def convert_state_dict(megatron_sd, target_hidden_size=None, target_n_experts=None,
                       target_moe_ffn=None, target_head_dim=None,
                       num_attention_heads=64, o_groups=8):
    sd = {}
    skipped = []
    expert_fc1 = {}  # (layer_idx, expert_idx) -> tensor
    expert_fc2 = {}
    hc_alphas = {}   # (layer_idx, "attn"|"ffn") -> {pre/post/res: tensor}

    # ── Global params (on-disk format) ──
    global_map = {
        "embedding.word_embeddings.weight": "embed.weight",
        "output_layer.weight": "head.weight",
        "decoder.final_layernorm.weight": "norm.weight",
        "decoder.hc_head_fn": "hc_head_fn",
        "decoder.hc_head_base": "hc_head_base",
        "decoder.hc_head_scale": "hc_head_scale",
    }

    for mg_key, tensor in megatron_sd.items():
        if "_extra_state" in mg_key:
            continue

        if mg_key in global_map:
            sd[global_map[mg_key]] = tensor
            continue

        # ── Per-layer params ──
        m = re.match(r"decoder\.layers\.(\d+)\.(.*)", mg_key)
        if not m:
            skipped.append(mg_key)
            continue

        li = int(m.group(1))
        sfx = m.group(2)

        # Megatron → on-disk format mapping
        per_layer_map = {
            # Layer norms
            "input_layernorm.weight": "attn_norm.weight",
            "pre_mlp_layernorm.weight": "ffn_norm.weight",
            # MLA attention projections (on-disk names)
            "self_attention.linear_q_down_proj.weight": "attn.wq_a.weight",
            "self_attention.q_layernorm.weight": "attn.q_norm.weight",
            "self_attention.linear_q_up_proj.weight": "attn.wq_b.weight",
            "self_attention.linear_kv_proj.weight": "attn.wkv.weight",
            "self_attention.kv_layernorm.weight": "attn.kv_norm.weight",
            "self_attention.linear_o_group_proj": "attn.wo_a.weight",
            "self_attention.linear_proj.weight": "attn.wo_b.weight",
            "self_attention.core_attention.attn_sink": "attn.attn_sink",
            # Compressor (on-disk: wkv + wgate, SGLang fuses into wkv_gate)
            "self_attention.core_attention.compressor.linear_wkv.weight": "attn.compressor.wkv.weight",
            "self_attention.core_attention.compressor.linear_wgate.weight": "attn.compressor.wgate.weight",
            "self_attention.core_attention.compressor.norm.weight": "attn.compressor.norm.weight",
            "self_attention.core_attention.compressor.ape": "attn.compressor.ape",
            # Indexer
            "self_attention.core_attention.indexer.linear_wq_b.weight": "attn.indexer.wq_b.weight",
            "self_attention.core_attention.indexer.linear_weights_proj.weight": "attn.indexer.weights_proj.weight",
            "self_attention.core_attention.indexer.compressor.linear_wkv.weight": "attn.indexer.compressor.wkv.weight",
            "self_attention.core_attention.indexer.compressor.linear_wgate.weight": "attn.indexer.compressor.wgate.weight",
            "self_attention.core_attention.indexer.compressor.norm.weight": "attn.indexer.compressor.norm.weight",
            "self_attention.core_attention.indexer.compressor.ape": "attn.indexer.compressor.ape",
            # MoE router
            "mlp.router.weight": "ffn.gate.weight",
            "mlp.router.expert_bias": "ffn.gate.bias",
            "mlp.router.tid2eid": "ffn.gate.tid2eid",
            # Shared expert down proj
            "mlp.shared_experts.linear_fc2.weight": "ffn.shared_experts.w2.weight",
        }

        if sfx in per_layer_map:
            sd[f"layers.{li}.{per_layer_map[sfx]}"] = tensor
            continue

        # --- Shared expert FC1: split gate (w1) / up (w3) ---
        if sfx == "mlp.shared_experts.linear_fc1.weight":
            half = tensor.shape[0] // 2
            sd[f"layers.{li}.ffn.shared_experts.w1.weight"] = tensor[:half].contiguous()
            sd[f"layers.{li}.ffn.shared_experts.w3.weight"] = tensor[half:].contiguous()
            continue

        # --- Expert FC weights: collect for per-expert saving ---
        fc1_match = re.match(r"mlp\.experts\.linear_fc1\.weight(\d+)", sfx)
        if fc1_match:
            expert_fc1[(li, int(fc1_match.group(1)))] = tensor
            continue

        fc2_match = re.match(r"mlp\.experts\.linear_fc2\.weight(\d+)", sfx)
        if fc2_match:
            expert_fc2[(li, int(fc2_match.group(1)))] = tensor
            continue

        # --- HC mapping_proj / bias → flat params ---
        hc_mapped = False
        for mg_hc, disk_hc in [("self_attention_hyper_connection", "hc_attn"),
                                ("mlp_hyper_connection", "hc_ffn")]:
            if sfx == f"{mg_hc}.mapping_proj.weight":
                sd[f"layers.{li}.{disk_hc}_fn"] = tensor
                hc_mapped = True
                break
            if sfx == f"{mg_hc}.bias":
                sd[f"layers.{li}.{disk_hc}_base"] = tensor
                hc_mapped = True
                break
        if hc_mapped:
            continue

        # --- HC alpha scalars: collect for concat ---
        alpha_match = re.match(
            r"(self_attention_hyper_connection|mlp_hyper_connection)\.alpha_(pre|post|res)", sfx
        )
        if alpha_match:
            hc_type = alpha_match.group(1)
            alpha_name = alpha_match.group(2)
            hc_kind = "attn" if "self_attention" in hc_type else "ffn"
            key = (li, hc_kind)
            if key not in hc_alphas:
                hc_alphas[key] = {}
            hc_alphas[key][alpha_name] = tensor
            continue

        skipped.append(mg_key)

    # --- Save expert weights as per-expert on-disk format ---
    # fc1 = [2*ffn, hidden] → split into w1 [ffn, hidden] (gate) + w3 [ffn, hidden] (up)
    for (li, ei), t in expert_fc1.items():
        half = t.shape[0] // 2
        sd[f"layers.{li}.ffn.experts.{ei}.w1.weight"] = t[:half].contiguous()
        sd[f"layers.{li}.ffn.experts.{ei}.w3.weight"] = t[half:].contiguous()

    # fc2 = [hidden, ffn] → w2 [hidden, ffn] (down)
    for (li, ei), t in expert_fc2.items():
        sd[f"layers.{li}.ffn.experts.{ei}.w2.weight"] = t.contiguous()

    # --- Concat HC alpha scalars → scale [3] ---
    for (li, hc_kind), alphas in hc_alphas.items():
        parts = [alphas.get("pre", torch.zeros(1)),
                 alphas.get("post", torch.zeros(1)),
                 alphas.get("res", torch.zeros(1))]
        sd[f"layers.{li}.hc_{hc_kind}_scale"] = torch.cat(parts).contiguous()

    # --- Post-processing: fix dimensions that DynamicModule export missed ---
    if target_hidden_size is not None:
        for k in ("embed.weight", "head.weight"):
            if k in sd and sd[k].shape[-1] > target_hidden_size:
                sd[k] = sd[k][:, :target_hidden_size].contiguous()
        # Slice weights whose input dim depends on hidden_size.
        # mcore_minitron does not prune V4's CSA/HCA compressor weights
        # (wkv/wgate) or the indexer weights_proj, so we must slice their
        # input dimension here.
        # NOTE: wo_a's input is n_heads*head_dim/n_groups (NOT hidden_size)
        # — it must NOT be sliced.
        for pattern in [r"^layers\.\d+\.attn\.wq_a\.weight$",
                        r"^layers\.\d+\.attn\.wkv\.weight$",
                        r"^layers\.\d+\.attn\.compressor\.wkv\.weight$",
                        r"^layers\.\d+\.attn\.compressor\.wgate\.weight$",
                        r"^layers\.\d+\.attn\.indexer\.weights_proj\.weight$"]:
            for k in list(sd.keys()):
                if re.match(pattern, k) and sd[k].shape[-1] > target_hidden_size:
                    sd[k] = sd[k][:, :target_hidden_size].contiguous()
        # wo_b output dim = hidden_size
        for k in list(sd.keys()):
            if re.match(r"^layers\.\d+\.attn\.wo_b\.weight$", k) and sd[k].shape[0] > target_hidden_size:
                sd[k] = sd[k][:target_hidden_size, :].contiguous()
        # Indexer compressor wkv/wgate: input dim = hidden_size
        for pattern in [r"^layers\.\d+\.attn\.indexer\.compressor\.wkv\.weight$",
                        r"^layers\.\d+\.attn\.indexer\.compressor\.wgate\.weight$"]:
            for k in list(sd.keys()):
                if re.match(pattern, k) and sd[k].shape[-1] > target_hidden_size:
                    sd[k] = sd[k][:, :target_hidden_size].contiguous()
        # HC fn weights: shape [mix_hc, hc_dim] where hc_dim = hc_mult * hidden_size
        hc_mult = 4  # from config
        target_hc_dim = hc_mult * target_hidden_size
        for k in list(sd.keys()):
            if re.match(r"^layers\.\d+\.hc_(attn|ffn)_fn$", k):
                t = sd[k]
                if t.shape[1] > target_hc_dim:
                    sd[k] = t[:, :target_hc_dim].contiguous()
        # hc_head_fn: same shape pattern [hc_mult, hc_mult * hidden_size]
        if "hc_head_fn" in sd:
            t = sd["hc_head_fn"]
            if t.shape[1] > target_hc_dim:
                sd["hc_head_fn"] = t[:, :target_hc_dim].contiguous()

    # ── MLA per-head slicing: shrink head_dim (512 → target, e.g. 128) ──
    # DeepSeek-V4 attention weights carry a [n_heads * head_dim] axis that
    # the Megatron-side mcore_minitron prune does NOT touch (head_dim is
    # fixed). For aggressive (4B-scale) pruning head_dim=512 is way too
    # wide relative to hidden_size=2048, so we slice the first target_head_dim
    # dims of each head here. Affects:
    #   wq_b:   [n_heads * hd, q_lora_rank]
    #   wkv:    [hd, hidden_size]
    #   wo_a:   [n_heads * hd // o_groups, o_groups * o_lora_rank]  (grouped)
    #   kv_norm.weight: [hd]
    #   compressor.{wkv,wgate}.weight: [coff * hd, hidden_size]
    #   compressor.ape: [compress_ratio, coff * hd]  (or flat)
    #   compressor.norm.weight: [hd]
    if target_head_dim is not None:
        _n_heads = int(num_attention_heads)
        _o_groups = int(o_groups)
        _target_hd = int(target_head_dim)
        _hd_count = 0

        # Detect orig_head_dim from the first wq_b we see. All attention
        # layers share the same head_dim, so one probe is enough.
        _orig_hd = None
        for _k in sd:
            if re.match(r"^layers\.\d+\.attn\.wq_b\.weight$", _k):
                _orig_hd = sd[_k].shape[0] // _n_heads
                break

        if _orig_hd is not None and _orig_hd > _target_hd:
            # wq_b: [n_heads * hd, q_lora_rank] → slice along hd axis
            for k in list(sd.keys()):
                if re.match(r"^layers\.\d+\.attn\.wq_b\.weight$", k):
                    t = sd[k]
                    q_lr = t.shape[1]
                    t2 = t.view(_n_heads, _orig_hd, q_lr)[:, :_target_hd, :]
                    sd[k] = t2.reshape(_n_heads * _target_hd, q_lr).contiguous()
                    _hd_count += 1

            # wkv: [hd, hidden_size] → slice along hd axis
            for k in list(sd.keys()):
                if re.match(r"^layers\.\d+\.attn\.wkv\.weight$", k):
                    t = sd[k]
                    if t.shape[0] > _target_hd:
                        sd[k] = t[:_target_hd, :].contiguous()
                        _hd_count += 1

            # wo_a: Megatron stores linear_o_group_proj.weight as
            # [o_groups * o_lora_rank, n_heads * hd // o_groups]  (OUT, IN).
            # The INPUT axis is laid out as [o_groups, heads_per_group, hd]
            # along the trailing dim, so we reshape as
            # [o_groups, o_lora_rank, heads_per_group, hd] to slice hd
            # while keeping group + heads-per-group structure intact.
            for k in list(sd.keys()):
                if re.match(r"^layers\.\d+\.attn\.wo_a\.weight$", k):
                    t = sd[k]
                    heads_per_group = _n_heads // _o_groups
                    out_dim = t.shape[0]
                    o_lora_rank = out_dim // _o_groups
                    t2 = t.view(_o_groups, o_lora_rank, heads_per_group, _orig_hd)
                    t2 = t2[:, :, :, :_target_hd]
                    sd[k] = t2.reshape(
                        _o_groups * o_lora_rank,
                        _o_groups * heads_per_group * _target_hd // _o_groups,
                    ).contiguous()
                    _hd_count += 1

            # kv_norm.weight: [hd]
            for k in list(sd.keys()):
                if re.match(r"^layers\.\d+\.attn\.kv_norm\.weight$", k):
                    t = sd[k]
                    if t.shape[0] > _target_hd:
                        sd[k] = t[:_target_hd].contiguous()
                        _hd_count += 1

            # compressor.{wkv,wgate}.weight: [coff * hd, hidden_size]
            for pattern in (
                r"^layers\.\d+\.attn\.compressor\.wkv\.weight$",
                r"^layers\.\d+\.attn\.compressor\.wgate\.weight$",
            ):
                for k in list(sd.keys()):
                    if re.match(pattern, k):
                        t = sd[k]
                        if t.dim() == 2 and t.shape[0] > _target_hd:
                            coff = t.shape[0] // _orig_hd
                            if coff * _orig_hd == t.shape[0]:
                                last_dim = t.shape[1]
                                t2 = t.view(coff, _orig_hd, last_dim)
                                t2 = t2[:, :_target_hd, :]
                                sd[k] = t2.reshape(
                                    coff * _target_hd, last_dim
                                ).contiguous()
                                _hd_count += 1

            # compressor.norm.weight: [hd]
            for k in list(sd.keys()):
                if re.match(r"^layers\.\d+\.attn\.compressor\.norm\.weight$", k):
                    t = sd[k]
                    if t.dim() == 1 and t.shape[0] > _target_hd:
                        sd[k] = t[:_target_hd].contiguous()
                        _hd_count += 1

            # compressor.ape: [compress_ratio, coff * hd] (2-D) or flat 1-D
            for k in list(sd.keys()):
                if re.match(r"^layers\.\d+\.attn\.compressor\.ape$", k):
                    t = sd[k]
                    if t.dim() == 2 and t.shape[-1] > _target_hd:
                        if t.shape[-1] % _orig_hd == 0:
                            coff = t.shape[-1] // _orig_hd
                            cr = t.shape[0]
                            t2 = t.view(cr, coff, _orig_hd)[:, :, :_target_hd]
                            sd[k] = t2.reshape(cr, coff * _target_hd).contiguous()
                            _hd_count += 1
                    elif t.dim() == 1 and t.shape[0] > _target_hd:
                        # Flat [cr * coff * hd] — slice assuming coff=1,
                        # so shape reduces to [cr * target_hd].
                        if t.shape[0] % _orig_hd == 0:
                            cr = t.shape[0] // _orig_hd
                            t2 = t.view(cr, _orig_hd)[:, :_target_hd]
                            sd[k] = t2.reshape(cr * _target_hd).contiguous()
                            _hd_count += 1

            print(
                f"  Sliced head_dim {_orig_hd} → {_target_hd} on "
                f"{_hd_count} MLA attention weights"
            )

    if target_n_experts is not None:
        for k in list(sd.keys()):
            m_e = re.match(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.", k)
            if m_e and int(m_e.group(2)) >= target_n_experts:
                del sd[k]

    if target_moe_ffn is not None:
        for k in list(sd.keys()):
            # Expert w1/w3: [ffn, hidden] → slice ffn dim
            if re.match(r"^layers\.\d+\.ffn\.experts\.\d+\.w[13]\.weight$", k):
                t = sd[k]
                if t.shape[0] > target_moe_ffn:
                    sd[k] = t[:target_moe_ffn, :].contiguous()
            # Expert w2: [hidden, ffn] → slice ffn dim
            elif re.match(r"^layers\.\d+\.ffn\.experts\.\d+\.w2\.weight$", k):
                t = sd[k]
                if t.shape[1] > target_moe_ffn:
                    sd[k] = t[:, :target_moe_ffn].contiguous()
        # Shared experts
        for k in list(sd.keys()):
            if k.endswith(".shared_experts.w1.weight") or k.endswith(".shared_experts.w3.weight"):
                t = sd[k]
                needs_ffn = t.shape[0] > target_moe_ffn
                needs_hid = target_hidden_size and t.shape[1] > target_hidden_size
                if needs_ffn or needs_hid:
                    s0 = target_moe_ffn if needs_ffn else t.shape[0]
                    s1 = target_hidden_size if needs_hid else t.shape[1]
                    sd[k] = t[:s0, :s1].contiguous()
            elif k.endswith(".shared_experts.w2.weight"):
                t = sd[k]
                needs_hid = target_hidden_size and t.shape[0] > target_hidden_size
                needs_ffn = t.shape[1] > target_moe_ffn
                if needs_hid or needs_ffn:
                    s0 = target_hidden_size if needs_hid else t.shape[0]
                    s1 = target_moe_ffn if needs_ffn else t.shape[1]
                    sd[k] = t[:s0, :s1].contiguous()

    # Also slice expert hidden dim (w1/w3 col dim, w2 row dim)
    if target_hidden_size is not None:
        for k in list(sd.keys()):
            if re.match(r"^layers\.\d+\.ffn\.experts\.\d+\.w[13]\.weight$", k):
                t = sd[k]
                if t.shape[1] > target_hidden_size:
                    sd[k] = t[:, :target_hidden_size].contiguous()
            elif re.match(r"^layers\.\d+\.ffn\.experts\.\d+\.w2\.weight$", k):
                t = sd[k]
                if t.shape[0] > target_hidden_size:
                    sd[k] = t[:target_hidden_size, :].contiguous()
        # Shared experts hidden_size slicing (independent of moe_ffn)
        for k in list(sd.keys()):
            if k.endswith(".shared_experts.w1.weight") or k.endswith(".shared_experts.w3.weight"):
                t = sd[k]
                if t.shape[1] > target_hidden_size:
                    ff = target_moe_ffn if (target_moe_ffn and t.shape[0] > target_moe_ffn) else t.shape[0]
                    sd[k] = t[:ff, :target_hidden_size].contiguous()
            elif k.endswith(".shared_experts.w2.weight"):
                t = sd[k]
                if t.shape[0] > target_hidden_size:
                    ff = target_moe_ffn if (target_moe_ffn and t.shape[1] > target_moe_ffn) else t.shape[1]
                    sd[k] = t[:target_hidden_size, :ff].contiguous()

    return sd, skipped


def load_megatron_state_dict(path):
    """Load a single .pt file or merge all ``pruned_model_rank*.pt`` shards in a dir.

    prune_deepseek_v4.py with PP>1 writes one .pt per PP rank
    (e.g. ``pruned_model_rank{0..7}.pt``). The shards are NOT clean
    tensor-parallel slices — they overlap on most keys (because
    ``unwrapped_model.state_dict()`` is rank-local) and have different
    tensor values for the same key across ranks. The most reliable
    reconstruction is:

    1. Load every shard.
    2. Use the **largest** shard (most keys) as the base — it owns the most
       parameters and tends to be the last PP stage which also holds the
       final norm + output layer.
    3. Fill any keys missing from the base by taking them from whichever
       other shard has them.

    For PP=1 / single-rank runs the legacy path ``pruned_model.pt`` is still
    accepted.
    """
    if os.path.isfile(path):
        print(f"Loading Megatron checkpoint: {path}")
        sd = torch.load(path, map_location="cpu", weights_only=True)
        print(f"  {len(sd)} keys loaded (single file)")
        return sd

    if os.path.isdir(path):
        shard_names = sorted(
            f for f in os.listdir(path)
            if f.startswith("pruned_model_rank") and f.endswith(".pt")
        )
        if not shard_names:
            raise FileNotFoundError(
                f"No pruned_model_rank*.pt shards found in directory: {path}"
            )

        print(f"Loading Megatron checkpoint dir: {path}")
        # Sort by PP rank (numerical) — critical for layer-index remapping below.
        shard_names = sorted(
            shard_names,
            key=lambda n: int(n.replace("pruned_model_rank", "").replace(".pt", "")),
        )
        loaded = []
        for name in shard_names:
            shard_path = os.path.join(path, name)
            sd = torch.load(shard_path, map_location="cpu", weights_only=True)
            n_layers_in_rank = len({
                int(k.split(".")[2])
                for k in sd if k.startswith("decoder.layers.")
            })
            print(f"  + {name}: {len(sd)} keys ({n_layers_in_rank} rank-relative layers)")
            loaded.append((name, sd, n_layers_in_rank))

        # ── PP-aware merge with layer-index remapping ──────────────────
        # CRITICAL: each PP rank's state_dict uses RANK-RELATIVE layer
        # indices. So `decoder.layers.0.*` in rank 7 is a different
        # global layer than `decoder.layers.0.*` in rank 0. Treating
        # them as the same key corrupts the merged result — e.g. rank
        # 0's SWA layer 0 (compress_ratio=0) gets clobbered by rank 7's
        # CSA layer 0 (compress_ratio=4).
        #
        # The PP layout distributes a contiguous block of global layers
        # to each rank. The block size equals the number of rank-relative
        # layers in that shard. We compute the global offset of each
        # rank by accumulating block sizes in rank order, then rewrite
        # `decoder.layers.{rl}.*` → `decoder.layers.{rl + offset}.*`.
        merged = {}
        global_offset = 0
        rank_layer_counts = []
        for name, sd, n_layers_in_rank in loaded:
            rank_layer_counts.append(n_layers_in_rank)
            for k, v in sd.items():
                m = re.match(r"^decoder\.layers\.(\d+)\.(.*)$", k)
                if m:
                    rl = int(m.group(1))
                    global_li = rl + global_offset
                    new_k = f"decoder.layers.{global_li}.{m.group(2)}"
                else:
                    new_k = k
                # If two ranks somehow contribute to the same global key
                # (shouldn't happen for PP-only shards), keep the first
                # (lowest rank) since it owns the lowest global layers.
                if new_k not in merged:
                    merged[new_k] = v
            global_offset += n_layers_in_rank
        print(
            f"  PP layout (rank → global-layer-count): "
            f"{rank_layer_counts} → {global_offset} total layers"
        )
        print(f"  {len(merged)} keys after PP-aware merge")
        return merged

    raise FileNotFoundError(f"Megatron checkpoint not found: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--megatron_ckpt",
        required=True,
        help="Either a single .pt file or a directory containing pruned_model_rank*.pt shards",
    )
    parser.add_argument("--hf_config", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    megatron_sd = load_megatron_state_dict(args.megatron_ckpt)

    print("Converting weight names to DeepSeek on-disk format...")
    with open(args.hf_config) as f:
        hf_cfg = json.load(f)
    out_sd, skipped = convert_state_dict(
        megatron_sd,
        target_hidden_size=hf_cfg.get("hidden_size"),
        target_n_experts=hf_cfg.get("n_routed_experts"),
        target_moe_ffn=hf_cfg.get("moe_intermediate_size"),
        target_head_dim=hf_cfg.get("head_dim"),
        num_attention_heads=hf_cfg.get("num_attention_heads", 64),
        o_groups=hf_cfg.get("o_groups", 8),
    )
    print(f"  {len(out_sd)} weights converted")
    if skipped:
        print(f"  {len(skipped)} keys skipped:")
        for k in skipped[:20]:
            print(f"    {k}")
        if len(skipped) > 20:
            print(f"    ... and {len(skipped) - 20} more")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── DeepSeek-V4: FP8 block-wise quantization for attention weights ─
    # The native DeepSeek-V4 checkpoint stores most attention and output
    # projections as FP8 (float8_e4m3fn) with block-wise e8m0 scales.
    # SGLang's DeepSeek-V4 code path expects this format and uses DeepGEMM
    # kernels for FP8 GEMM.  Without FP8 weights + scales, SGLang falls
    # through to BF16 paths that may trigger CUBLAS errors on certain
    # operations (e.g. the indexer's weights_proj).
    #
    # We quantize the same weight set as the native checkpoint:
    #   attn.wkv, attn.wq_a, attn.wq_b, attn.wo_a, attn.wo_b,
    #   attn.indexer.wq_b
    # Compressor / indexer BF16 weights (wkv, wgate, weights_proj, norms)
    # are left in BF16 — matching the native format.
    #
    # Shared experts: only quantize if moe_ffn is compatible with block_size=128
    # for TP=4 or TP=8 (the common deployment configs). Otherwise SGLang's FP8
    # validation fails with "output_partition_size not divisible by block_n".
    #
    # CRITICAL: If shared_experts are incompatible, we must NOT quantize ANY
    # weights to FP8. Otherwise the checkpoint will have FP8 weights with
    # .scale keys, but SGLang (without quantization_config) will create BF16
    # parameters and fail to load the FP8 scales.
    _FP8_WEIGHT_PATTERNS = []
    _moe_ffn_for_fp8 = hf_cfg.get("moe_intermediate_size", 2048)
    _shared_fp8_ok = any(
        (_moe_ffn_for_fp8 // tp) % 128 == 0 for tp in [4, 8]
    )
    if _shared_fp8_ok:
        _FP8_WEIGHT_PATTERNS = [
            re.compile(r"^layers\.\d+\.attn\.wkv\.weight$"),
            re.compile(r"^layers\.\d+\.attn\.wq_a\.weight$"),
            re.compile(r"^layers\.\d+\.attn\.wq_b\.weight$"),
            re.compile(r"^layers\.\d+\.attn\.wo_a\.weight$"),
            re.compile(r"^layers\.\d+\.attn\.wo_b\.weight$"),
            re.compile(r"^layers\.\d+\.attn\.indexer\.wq_b\.weight$"),
            re.compile(r"^layers\.\d+\.ffn\.shared_experts\.w[123]\.weight$"),
        ]
    else:
        print(f"  Disabling FP8 for ALL weights (moe_ffn={_moe_ffn_for_fp8} "
              f"not compatible with block_size=128 for TP=4/8). "
              f"Checkpoint will be BF16.")
    _fp8_count = 0
    for k in list(out_sd.keys()):
        if any(p.match(k) for p in _FP8_WEIGHT_PATTERNS):
            t = out_sd[k]
            if t.dtype == torch.bfloat16 or t.dtype == torch.float16:
                # Ensure dims are divisible by block_size (128)
                M, N = t.shape
                if M % 128 == 0 and N % 128 == 0:
                    w_fp8, s_e8m0 = quantize_fp8_blockwise(t)
                    out_sd[k] = w_fp8
                    scale_key = k.replace(".weight", ".scale")
                    out_sd[scale_key] = s_e8m0
                    _fp8_count += 1
    if _fp8_count:
        print(f"  Quantized {_fp8_count} attention weights to FP8 block-wise")

    # ── Expert weights: slice from native MXFP4 checkpoint ────────────
    # The native DeepSeek-V4 stores routed expert weights in MXFP4 format
    # (int8-packed e2m1, block_size=32, e8m0 scales) which the Marlin
    # backend uses directly after its own repacking step.  Rather than
    # re-quantising from scratch (which risks subtle format mismatches),
    # we load the native weights and slice to pruned dimensions:
    #   w1/w3 weight: [moe_ffn, hidden//2] → [pruned_ffn, pruned_hidden//2]
    #   w2    weight: [hidden, moe_ffn//2] → [pruned_hidden, pruned_ffn//2]
    #   scales sliced along both dims correspondingly.
    # Pruned layer i maps to native layer i (depth pruning keeps the
    # first N layers).  Pruned expert j maps to native expert j (expert
    # pruning keeps the first K experts).
    _NATIVE_MODEL_DIR = "/data/.cache/models/deepseek-ai/DeepSeek-V4-Flash"
    _target_moe_ffn = hf_cfg.get("moe_intermediate_size")
    _target_hidden = hf_cfg.get("hidden_size")
    _target_n_experts = hf_cfg.get("n_routed_experts")
    _n_layers = hf_cfg.get("num_hidden_layers", 0)

    _mxfp4_count = 0
    # Remove BF16 expert weights BEFORE adding native MXFP4 (same key names)
    for k in [k for k in out_sd if re.match(
            r"^layers\.\d+\.ffn\.experts\.\d+\.w[123]\.weight$", k)]:
        del out_sd[k]
    if _target_moe_ffn and _target_hidden and os.path.isdir(_NATIVE_MODEL_DIR):
        from safetensors.torch import load_file as _st_load
        _native_idx = os.path.join(_NATIVE_MODEL_DIR, "model.safetensors.index.json")
        if os.path.exists(_native_idx):
            with open(_native_idx) as _f:
                _nwm = json.load(_f)["weight_map"]

            # Native expert shapes: w1/w3=[moe_ffn, hidden//2], w2=[hidden, moe_ffn//2]
            # Packed: 2 FP4 values per byte along last dim.
            # Scales: [rows, cols//block_size] where block_size=32.
            _hp = _target_hidden // 2       # packed hidden = 1024
            _fp = _target_moe_ffn // 2      # packed ffn = 512
            _sh = _target_hidden // 32      # scale hidden = 64
            _sf = _target_moe_ffn // 32     # scale ffn = 32

            # Cache loaded native shards to avoid repeated I/O
            _shard_cache = {}

            for _lid in range(_n_layers):
                for _eid in range(_target_n_experts):
                    for _wn, _wr, _wc, _sr, _sc in [
                        ("w1", _target_moe_ffn, _hp, _target_moe_ffn, _sh),
                        ("w3", _target_moe_ffn, _hp, _target_moe_ffn, _sh),
                        ("w2", _target_hidden, _fp, _target_hidden, _sf),
                    ]:
                        _nk_w = f"layers.{_lid}.ffn.experts.{_eid}.{_wn}.weight"
                        _nk_s = f"layers.{_lid}.ffn.experts.{_eid}.{_wn}.scale"
                        if _nk_w not in _nwm or _nk_s not in _nwm:
                            continue
                        # Load native weight shard (cached)
                        _ws = _nwm[_nk_w]
                        _ss = _nwm[_nk_s]
                        if _ws not in _shard_cache:
                            _shard_cache[_ws] = _st_load(
                                os.path.join(_NATIVE_MODEL_DIR, _ws), device="cpu")
                        if _ss not in _shard_cache:
                            _shard_cache[_ss] = _st_load(
                                os.path.join(_NATIVE_MODEL_DIR, _ss), device="cpu")
                        _nw = _shard_cache[_ws][_nk_w]
                        _ns = _shard_cache[_ss][_nk_s]
                        # Slice to pruned dimensions
                        _nw_sliced = _nw[:_wr, :_wc].contiguous()
                        _ns_sliced = _ns[:_sr, :_sc].contiguous()
                        # Dequantize MXFP4 to BF16 for SGLang compatibility
                        _nw_bf16 = dequantize_mxfp4(_nw_sliced, _ns_sliced)
                        out_sd[f"layers.{_lid}.ffn.experts.{_eid}.{_wn}.weight"] = _nw_bf16
                        _mxfp4_count += 1

    if _mxfp4_count:
        print(f"  Dequantized {_mxfp4_count} expert weights from MXFP4 to BF16")
    else:
        print("  WARNING: Could not slice expert weights from native model")

    # ── Re-quantize expert weights to MXFP4 for compact deployment ────
    # After slicing in BF16 (which avoids nibble/scale alignment issues),
    # re-quantize to MXFP4 (int8-packed e2m1 + e8m0 scales) matching the
    # native on-disk format. SGLang's Marlin MoE kernel consumes this
    # directly when config has "expert_dtype": "fp4".
    _MXFP4_BLOCK = 32
    _requant_count = 0
    _requant_failed = 0
    for k in list(out_sd.keys()):
        if not re.match(r"^layers\.\d+\.ffn\.experts\.\d+\.w[123]\.weight$", k):
            continue
        t = out_sd[k]
        if t.dtype != torch.bfloat16:
            continue
        M, N = t.shape
        if N % _MXFP4_BLOCK != 0:
            _requant_failed += 1
            continue
        packed, scale = quantize_mxfp4(t, block_size=_MXFP4_BLOCK)
        out_sd[k] = packed
        out_sd[k.replace(".weight", ".scale")] = scale
        _requant_count += 1
    if _requant_count:
        print(f"  Re-quantized {_requant_count} expert weights to MXFP4 "
              f"(block_size={_MXFP4_BLOCK})")
        if _requant_failed:
            print(f"  WARNING: {_requant_failed} experts skipped "
                  f"(N not divisible by {_MXFP4_BLOCK})")
    _experts_are_mxfp4 = _requant_count > 0 and _requant_failed == 0

    # ── Fallback: fill non-expert params for layers missing from Megatron ──
    # When prune_deepseek_v4.py runs with PP>1 and num_layers>PP, each rank
    # only owns its PP-stage layers. The saved state_dict may lack non-expert
    # params for some pruned layers. Backfill from the native checkpoint
    # (DeepSeek-V4-Flash) which uses the same on-disk format as our output
    # (`layers.N.attn.wkv.weight`, `layers.N.attn_norm.weight`, etc.).
    _NATIVE_MODEL_DIR_BACKFILL = "/data/.cache/models/deepseek-ai/DeepSeek-V4-Flash"
    if os.path.isdir(_NATIVE_MODEL_DIR_BACKFILL):
        from safetensors.torch import load_file as _st_load_backfill
        with open(os.path.join(_NATIVE_MODEL_DIR_BACKFILL, "model.safetensors.index.json")) as _f:
            _nwm_backfill = json.load(_f)["weight_map"]
        _shard_cache_bf = {}
        # Determine which layers are missing non-expert params. Expert keys
        # are populated for ALL layers via native MXFP4 slicing above, so
        # checking for any layers.* key would falsely report nothing missing.
        # Instead, look for a representative non-expert key per layer.
        _present_layers = set()
        for _k in out_sd:
            if not _k.startswith("layers."):
                continue
            if ".ffn.experts." in _k:
                continue
            _m = re.match(r"^layers\.(\d+)\.", _k)
            if _m:
                _present_layers.add(int(_m.group(1)))
        _missing_layers = sorted(
            li for li in range(_n_layers) if li not in _present_layers
        )
        if _missing_layers:
            print(f"  Backfilling non-expert params for layers {_missing_layers} from native model")
            # Keys we want per layer (on-disk format). Native uses identical
            # naming, so we copy them through directly.
            _NON_EXPERT_SUFFIXES = (
                "attn.attn_sink", "attn.compressor.ape", "attn.compressor.norm.weight",
                "attn.compressor.wgate.weight", "attn.compressor.wkv.weight",
                "attn.indexer.compressor.ape", "attn.indexer.compressor.norm.weight",
                "attn.indexer.compressor.wgate.weight", "attn.indexer.compressor.wkv.weight",
                "attn.indexer.weights_proj.weight", "attn.indexer.wq_b.weight", "attn.indexer.wq_b.scale",
                "attn.kv_norm.weight", "attn.q_norm.weight",
                "attn.wkv.weight", "attn.wkv.scale",
                "attn.wo_a.weight", "attn.wo_a.scale", "attn.wo_b.weight", "attn.wo_b.scale",
                "attn.wq_a.weight", "attn.wq_a.scale", "attn.wq_b.weight", "attn.wq_b.scale",
                "attn_norm.weight", "ffn_norm.weight",
                "ffn.gate.weight", "ffn.gate.bias", "ffn.gate.tid2eid",
                "ffn.shared_experts.w1.weight", "ffn.shared_experts.w1.scale",
                "ffn.shared_experts.w2.weight", "ffn.shared_experts.w3.weight",
                "ffn.shared_experts.w3.scale",
            )
            _bf_count = 0
            for _li in _missing_layers:
                for _sfx in _NON_EXPERT_SUFFIXES:
                    _k = f"layers.{_li}.{_sfx}"
                    if _k in out_sd:
                        continue
                    if _k not in _nwm_backfill:
                        continue
                    _ws = _nwm_backfill[_k]
                    if _ws not in _shard_cache_bf:
                        _shard_cache_bf[_ws] = _st_load_backfill(
                            os.path.join(_NATIVE_MODEL_DIR_BACKFILL, _ws), device="cpu"
                        )
                    out_sd[_k] = _shard_cache_bf[_ws][_k].contiguous()
                    _bf_count += 1
            if _bf_count:
                print(f"  Backfilled {_bf_count} non-expert keys from native model")
            else:
                print(f"  WARNING: backfill found no matching keys for missing layers")

    # ── Hash MoE tid2eid recovery ──────────────────────────────────────
    # Megatron Bridge only loads tid2eid for the first hash layer.
    # For remaining hash layers, load tid2eid from native model and
    # remove the spurious expert_bias that Megatron initialized instead.
    with open(args.hf_config) as f:
        _cfg_for_hash = json.load(f)
    _n_hash = _cfg_for_hash.get("num_hash_layers", 0)
    _n_layers_pruned = _cfg_for_hash.get("num_hidden_layers", 0)
    _hash_fixed = 0
    if _n_hash > 0 and os.path.isdir(_NATIVE_MODEL_DIR):
        _native_idx_path = os.path.join(_NATIVE_MODEL_DIR, "model.safetensors.index.json")
        if os.path.exists(_native_idx_path):
            with open(_native_idx_path) as _f:
                _nwm = json.load(_f)["weight_map"]
            # Build set of original hash layer indices
            _orig_mlp_types = _cfg_for_hash.get("mlp_layer_types", [])
            # Native model: first num_hash_layers are hash_moe
            _orig_hash_indices = set(range(_n_hash))
            # Pruned model: need to figure out which pruned layers are hash
            # Based on mlp_layer_types from the config (already reconstructed)
            _pruned_mlp_types = _cfg_for_hash.get("mlp_layer_types", [])
            for _li in range(min(_n_layers_pruned, len(_pruned_mlp_types))):
                if _pruned_mlp_types[_li] != "hash_moe":
                    continue
                _tid_key = f"layers.{_li}.ffn.gate.tid2eid"
                if _tid_key in out_sd:
                    continue  # already present
                # Find corresponding native layer (same index since depth
                # pruning keeps first N layers)
                _native_tid_key = f"layers.{_li}.ffn.gate.tid2eid"
                if _native_tid_key not in _nwm:
                    continue
                _shard = _nwm[_native_tid_key]
                if _shard not in _shard_cache:
                    _shard_cache[_shard] = _st_load(
                        os.path.join(_NATIVE_MODEL_DIR, _shard), device="cpu")
                out_sd[_tid_key] = _shard_cache[_shard][_native_tid_key]
                # Remove spurious expert_bias for this hash layer
                _bias_key = f"layers.{_li}.ffn.gate.bias"
                if _bias_key in out_sd:
                    del out_sd[_bias_key]
                _hash_fixed += 1
    if _hash_fixed:
        print(f"  Recovered tid2eid for {_hash_fixed} hash MoE layers from native model")

    # ── tid2eid expert ID clamping ─────────────────────────────────────
    # When experts are pruned (e.g. 256 → 32), tid2eid still references
    # old expert IDs (up to 255). Clamp to actual expert count to prevent
    # out-of-bounds memory access in MoE fused gate kernel.
    _tid_clamped = 0
    for _k in list(out_sd.keys()):
        if not _k.endswith(".ffn.gate.tid2eid"):
            continue
        _li = int(_k.split(".")[1])
        # Count actual experts for this layer
        _exp_ids = set()
        for _ek in out_sd:
            _m = re.match(rf"^layers\.{_li}\.ffn\.experts\.(\d+)\.w1\.weight$", _ek)
            if _m:
                _exp_ids.add(int(_m.group(1)))
        if not _exp_ids:
            continue
        _max_eid = max(_exp_ids)
        _t = out_sd[_k]
        if _t.max().item() > _max_eid:
            out_sd[_k] = _t.clamp(max=_max_eid)
            _tid_clamped += 1
    if _tid_clamped:
        print(f"  Clamped tid2eid expert IDs for {_tid_clamped} hash layers")

    with open(args.hf_config) as f:
        config = json.load(f)

    # ── DeepSeek-V4: ship a custom config class via auto_map ──────────
    # The stock transformers DeepseekV4Config consumes the legacy
    # ``compress_ratios`` list in __post_init__ and does NOT expose it as
    # an instance attribute.  SGLang (and other inference frameworks) read
    # ``config.compress_ratios`` directly, which raises AttributeError on
    # current transformers versions.
    #
    # We solve this by shipping a tiny custom config class (via auto_map +
    # trust_remote_code) that inherits from the built-in class and adds
    # ``compress_ratios`` back as an attribute derived from layer_types.
    _CUSTOM_CONFIG_MODULE = "configuration_deepseek_v4_pruned"
    _CUSTOM_CONFIG_CLASS = "DeepseekV4PrunedConfig"
    _custom_config_src = textwrap.dedent("""\
        \"\"\"Custom DeepSeek-V4 config that preserves compress_ratios.

        Ships with pruned checkpoints so that inference frameworks (SGLang,
        vLLM, etc.) can read ``config.compress_ratios`` — an attribute the
        stock transformers DeepseekV4Config consumes during __post_init__
        but never stores.
        \"\"\"
        from transformers.models.deepseek_v4.configuration_deepseek_v4 import (
            DeepseekV4Config as _Base,
        )

        _TYPE_TO_RATIO = {
            "sliding_attention": 0,
            "compressed_sparse_attention": 4,
            "heavily_compressed_attention": 128,
        }


        class DeepseekV4PrunedConfig(_Base):
            def __post_init__(self, **kwargs):
                super().__post_init__(**kwargs)
                if not getattr(self, "compress_ratios", None) and self.layer_types:
                    rates = self.compress_rates or {}
                    self.compress_ratios = [
                        rates.get(t, _TYPE_TO_RATIO.get(t, 0))
                        for t in self.layer_types
                    ]
    """)
    _cfg_cls_path = os.path.join(args.output_dir, f"{_CUSTOM_CONFIG_MODULE}.py")
    with open(_cfg_cls_path, "w") as f:
        f.write(_custom_config_src)
    config.setdefault("auto_map", {})["AutoConfig"] = (
        f"{_CUSTOM_CONFIG_MODULE}.{_CUSTOM_CONFIG_CLASS}"
    )

    # Write quantization_config so SGLang creates Fp8Config and uses the
    # correct MoE kernel for MXFP4 experts (Fp8MoEMethod with is_fp4_expert).
    # Without quantization_config, SGLang falls back to UnquantizedFusedMoEMethod
    # which expects full-width BF16 weights and has no scale parameters.
    #
    # When attention weights were also FP8-quantized (_fp8_count > 0), only the
    # router gate (and possibly shared_experts) need ignoring.
    # When attention is BF16 (_fp8_count == 0) but experts are MXFP4, we must
    # ignore all BF16 linear layers (self_attn, shared_experts, gate) so SGLang
    # creates BF16 params for them while using FP4 MoE for routed experts.
    if _fp8_count > 0 or _experts_are_mxfp4:
        _moe_ffn = config.get("moe_intermediate_size", 2048)
        _ignored = ["mlp.gate"]

        if _fp8_count == 0:
            # Attention and shared_experts are BF16; tell SGLang to skip them.
            _ignored.extend(["self_attn", "shared_experts"])
            print(f"  quantization_config written for MXFP4 experts only; "
                  f"attention/shared_experts remain BF16 (ignored_layers)")
        else:
            # Attention is FP8. Check if shared_experts can also be FP8.
            _fp8_ok_tp = []
            for _tp in [2, 4, 8]:
                _part = _moe_ffn // _tp
                if _part % 128 == 0:
                    _fp8_ok_tp.append(_tp)
            if 8 not in _fp8_ok_tp and 4 not in _fp8_ok_tp:
                _ignored.append("shared_experts")
                print(f"  shared_experts excluded from FP8 (moe_ffn={_moe_ffn} not "
                      f"compatible with block_size=128 for TP<=8)")
            else:
                _max_tp = max(_fp8_ok_tp)
                print(f"  shared_experts FP8 OK for TP<={_max_tp} "
                      f"(moe_ffn={_moe_ffn}, partition={_moe_ffn // _max_tp})")

        config["quantization_config"] = {
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "quant_method": "fp8",
            "scale_fmt": "ue8m0",
            "weight_block_size": [128, 128],
            "ignored_layers": _ignored,
        }
    # Set expert_dtype based on whether re-quantization to MXFP4 succeeded.
    # SGLang uses "fp4" to select the Marlin MoE kernel for MXFP4 experts.
    if _experts_are_mxfp4:
        config["expert_dtype"] = "fp4"
    else:
        config.pop("expert_dtype", None)
    # Reconstruct compress_ratios and layer_types from the actual checkpoint.
    # The HF config's layer_types may be wrong after importance-based layer
    # dropping (it takes first N entries from original, but dropped layers
    # shift the positions). We detect each layer's type from its keys:
    #   - indexer present → CSA (compress_ratio=4)
    #   - compressor without indexer → HCA (compress_ratio=128)
    #   - neither → SWA (compress_ratio=0)
    n_layers = config.get("num_hidden_layers", 0)
    if n_layers > 0:
        _reconstructed_ratios = []
        _reconstructed_types = []
        for _li in range(n_layers):
            has_indexer = any(
                k.startswith(f"layers.{_li}.attn.indexer.")
                for k in out_sd
            )
            has_compressor = any(
                k.startswith(f"layers.{_li}.attn.compressor.")
                for k in out_sd
            )
            if has_indexer:
                _reconstructed_ratios.append(4)
                _reconstructed_types.append("compressed_sparse_attention")
            elif has_compressor:
                _reconstructed_ratios.append(128)
                _reconstructed_types.append("heavily_compressed_attention")
            else:
                _reconstructed_ratios.append(0)
                _reconstructed_types.append("sliding_attention")
        config["compress_ratios"] = _reconstructed_ratios
        config["layer_types"] = _reconstructed_types
        _n_csa = _reconstructed_ratios.count(4)
        _n_hca = _reconstructed_ratios.count(128)
        _n_swa = _reconstructed_ratios.count(0)
        print(f"  Reconstructed layer types: SWA={_n_swa}, CSA={_n_csa}, HCA={_n_hca}")

    # Reconstruct mlp_layer_types: hash_moe for first 3 MoE layers,
    # moe for the rest. The number of hash layers is from the original config.
    n_hash = config.get("num_hash_layers", 3)
    if n_layers > 0:
        _mlp_types = []
        _moe_count = 0
        for _li in range(n_layers):
            has_mlp = any(
                k.startswith(f"layers.{_li}.ffn.gate.")
                for k in out_sd
            )
            if has_mlp:
                if _moe_count < n_hash:
                    _mlp_types.append("hash_moe")
                else:
                    _mlp_types.append("moe")
                _moe_count += 1
            else:
                _mlp_types.append("moe")  # placeholder
        config["mlp_layer_types"] = _mlp_types
        config["num_hash_layers"] = n_hash
    # Carry over V4-specific architecture fields from original config
    # that are not modified by pruning but required by SGLang.
    _ORIG_MODEL_DIR = _NATIVE_MODEL_DIR if os.path.isdir(_NATIVE_MODEL_DIR) else None
    if _ORIG_MODEL_DIR:
        _orig_cfg_path = os.path.join(_ORIG_MODEL_DIR, "config.json")
        if os.path.exists(_orig_cfg_path):
            with open(_orig_cfg_path) as _f:
                _orig_cfg = json.load(_f)
            for _field in ("rope_parameters", "compress_rope_theta",
                           "q_lora_rank", "o_lora_rank", "o_groups",
                           "qk_rope_head_dim", "head_dim",
                           "index_n_heads", "index_head_dim", "index_topk",
                           "hc_mult", "hc_eps", "hc_sinkhorn_iters",
                           "swiglu_limit", "sliding_window",
                           "num_experts_per_tok", "num_key_value_heads",
                           "num_attention_heads", "partial_rotary_factor",
                           "rope_scaling", "routed_scaling_factor",
                           "scoring_func", "topk_method",
                           "norm_topk_prob", "n_shared_experts"):
                if _field in _orig_cfg and _field not in config:
                    config[_field] = _orig_cfg[_field]

    # Recalculate partial_rotary_factor if head_dim was changed by pruning.
    # transformers' DeepseekV4Config.__post_init__ ignores explicit
    # qk_rope_head_dim when partial_rotary_factor is set, recomputing:
    #   qk_rope_head_dim = int(head_dim * partial_rotary_factor)
    # Original: 512 * 0.125 = 64. After pruning head_dim to e.g. 256,
    # partial_rotary_factor must become 64/256 = 0.25 to preserve the
    # RoPE dimension.
    _final_head_dim = config.get("head_dim")
    _final_qk_rope = config.get("qk_rope_head_dim")
    if _final_head_dim and _final_qk_rope:
        _correct_prf = _final_qk_rope / _final_head_dim
        if config.get("partial_rotary_factor") != _correct_prf:
            print(
                f"  Fixing partial_rotary_factor: "
                f"{config.get('partial_rotary_factor')} → {_correct_prf} "
                f"(to preserve qk_rope_head_dim={_final_qk_rope} "
                f"with head_dim={_final_head_dim})"
            )
            config["partial_rotary_factor"] = _correct_prf
        # Also fix nested rope_parameters.*.partial_rotary_factor
        _rp = config.get("rope_parameters")
        if isinstance(_rp, dict):
            for _rp_sec in _rp.values():
                if isinstance(_rp_sec, dict) and "partial_rotary_factor" in _rp_sec:
                    _rp_sec["partial_rotary_factor"] = _correct_prf

    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    for fname in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        src = os.path.join(os.path.dirname(args.hf_config), fname)
        if os.path.exists(src):
            with open(src) as f_in:
                data = f_in.read()
            with open(os.path.join(args.output_dir, fname), "w") as f_out:
                f_out.write(data)

    print(f"Saving safetensors to {args.output_dir}...")
    max_shard = 5 * 1024**3
    shards = []
    current_shard = {}
    current_size = 0

    for key in sorted(out_sd.keys()):
        tensor = out_sd[key]
        tsz = tensor.numel() * tensor.element_size()
        if current_size + tsz > max_shard and current_shard:
            shards.append(current_shard)
            current_shard = {}
            current_size = 0
        current_shard[key] = tensor
        current_size += tsz
    if current_shard:
        shards.append(current_shard)

    weight_map = {}
    for i, shard in enumerate(shards):
        shard_name = f"model-{i+1:05d}-of-{len(shards):05d}.safetensors"
        save_file(shard, os.path.join(args.output_dir, shard_name))
        for key in shard:
            weight_map[key] = shard_name
        print(f"  {shard_name} ({len(shard)} tensors)")

    total_size = sum(t.numel() * t.element_size() for t in out_sd.values())
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(os.path.join(args.output_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)

    print(f"\nDone! {len(out_sd)} weights in {len(shards)} shards ({total_size/1e9:.1f} GB).")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()

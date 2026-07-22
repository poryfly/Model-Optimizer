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

"""Smoke test for DeepSeek-V4 pruning via mcore_minitron.

Requires V4 Bridge and MCore modules to be available.
Skips gracefully when V4 dependencies are not installed.
"""

from types import SimpleNamespace

import pytest

from modelopt.torch.nas.plugins.megatron import HAS_V4

pytestmark = pytest.mark.skipif(not HAS_V4, reason="V4 MCore modules not available")


def _make_v4_config(
    hidden_size=128,
    num_layers=6,
    num_attention_heads=4,
    num_moe_experts=8,
    moe_router_topk=2,
    moe_ffn_hidden_size=64,
    q_lora_rank=32,
    kv_lora_rank=32,
    qk_head_dim=16,
    v_head_dim=16,
    qk_rope_head_dim=8,
    compress_ratios=None,
):
    """Create a minimal V4-like config for testing."""
    if compress_ratios is None:
        compress_ratios = [0, 0, 4, 128, 4, 128]
    return SimpleNamespace(
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        num_query_groups=num_attention_heads,
        kv_channels=hidden_size // num_attention_heads,
        ffn_hidden_size=128,
        num_moe_experts=num_moe_experts,
        moe_router_topk=moe_router_topk,
        moe_ffn_hidden_size=moe_ffn_hidden_size,
        moe_shared_expert_intermediate_size=None,
        moe_shared_expert_gate=False,
        gated_linear_unit=True,
        add_bias_linear=False,
        normalization="RMSNorm",
        qk_layernorm=False,
        attention_output_gate=False,
        moe_layer_freq=1,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        sequence_parallel=False,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        # V4-specific
        compress_ratios=compress_ratios,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_head_dim=qk_head_dim,
        v_head_dim=v_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        o_lora_rank=None,
        o_groups=None,
        index_n_heads=4,
        index_head_dim=16,
        index_topk=16,
        hc_mult=0,
        mamba_num_heads=None,
        mamba_head_dim=None,
        mamba_num_groups=None,
        mamba_state_dim=None,
    )


class TestV4ParamCount:
    """Test V4 parameter counting formulas."""

    def test_mla_attn_layer_params(self):
        from modelopt.torch.nas.plugins.megatron_model_stats import _mla_attn_layer_params

        params = _mla_attn_layer_params(
            hidden_size=128,
            q_lora_rank=32,
            kv_lora_rank=32,
            num_attention_heads=4,
            qk_head_dim=16,
            v_head_dim=16,
            qk_rope_head_dim=8,
            add_bias_linear=False,
            normalization="RMSNorm",
        )
        assert params > 0

    def test_mcore_param_count_v4(self):
        from modelopt.torch.nas.plugins.megatron_model_stats import mcore_param_count

        config = _make_v4_config()
        total, active = mcore_param_count(config, vocab_size=256)
        assert total > 0
        assert active > 0
        assert total >= active


class TestV4DepthPruning:
    """Test V4 depth pruning with compress_ratios constraints."""

    def test_prune_v4_depth_pairs(self):
        from modelopt.torch.prune.plugins.mcore_minitron import _prune_v4_depth

        compress_ratios = [0, 0, 4, 128, 4, 128]
        sorted_layers = [5, 3, 6, 4, 1, 2]

        model_ns = SimpleNamespace(
            config=SimpleNamespace(compress_ratios=compress_ratios)
        )
        model_wrapper = SimpleNamespace()
        model_wrapper.inner = model_ns

        class FakeModule:
            def __init__(self, inner):
                self._inner = inner

            def named_modules(self):
                yield "inner", self._inner

            @property
            def config(self):
                return self._inner.config

        model = FakeModule(model_ns)

        from megatron.core.models.gpt import GPTModel

        # Patch isinstance check by using a direct approach
        import modelopt.torch.prune.plugins.mcore_minitron as mmod

        original_supported = mmod.SUPPORTED_MODELS
        try:
            mmod.SUPPORTED_MODELS = {type(model): "test"}
            layers_to_drop = _prune_v4_depth(model, sorted_layers, num_layers_active=4)
            assert len(layers_to_drop) == 2
            # Dropped layers should be a CSA/HCA pair
            for dropped in layers_to_drop:
                assert compress_ratios[dropped - 1] > 0
        finally:
            mmod.SUPPORTED_MODELS = original_supported


class TestV4HashRoutingRemap:
    """Test hash routing tid2eid remapping after expert pruning."""

    def test_remap_hash_routing(self):
        import torch

        from modelopt.torch.prune.plugins.mcore_minitron import _remap_hash_routing_experts

        class FakeHashRouter:
            def __init__(self):
                self.tid2eid = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])

        router = FakeHashRouter()

        class FakeModel:
            def modules(self):
                return [router]

        kept = [0, 2, 4, 6]
        _remap_hash_routing_experts(FakeModel(), kept)
        # All IDs should be remapped to valid indices (0-3)
        assert all(0 <= x.item() <= 3 for x in router.tid2eid)


class TestV4KVCache:
    """Test V4 MLA KV cache estimation."""

    def test_mla_kv_cache_smaller_than_mha(self):
        from modelopt.torch.nas.plugins.megatron_model_stats import mcore_memory_footprint_mb

        config_mha = _make_v4_config()
        config_mha.compress_ratios = None
        config_mha.q_lora_rank = None
        config_mha.kv_lora_rank = None

        config_mla = _make_v4_config()

        _, kv_mha, _, _ = mcore_memory_footprint_mb(
            config_mha, vocab_size=256, sequence_length=128, batch_size=1
        )
        _, kv_mla, _, _ = mcore_memory_footprint_mb(
            config_mla, vocab_size=256, sequence_length=128, batch_size=1
        )
        # MLA KV cache should be smaller due to compressed dimensions
        assert kv_mla < kv_mha

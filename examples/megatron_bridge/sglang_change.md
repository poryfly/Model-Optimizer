# SGLang 适配 head_dim=256 裁剪版 DeepSeek V4 模型改动总结

## 背景

原始 DeepSeek V4 模型：`head_dim=512`, `qk_nope_head_dim=448`, `qk_rope_head_dim=64`  
裁剪后模型：`head_dim=256`, `qk_nope_head_dim=192`, `qk_rope_head_dim=64`  
运行环境：RTX 5090 (SM 12.0 / Blackwell)，SGLang v0.5.14

---

## 改动 1：flash_mla_sm120.py — FlashMLA KV Cache 布局动态推导

**文件路径**：`sglang/srt/layers/attention/flash_mla_sm120.py`

### 问题

原始代码将所有 KV cache 页内布局常数硬编码为 DeepSeek V4 (head_dim=512) 的值：

| 常数 | 原始硬编码值 | 裁剪模型实际值 |
|------|-------------|--------------|
| `_NOPE_DIM` | 448 | 192 |
| `_NOPE_ROPE_STRIDE` | 576 | 256 |
| `_NUM_TILES` | 7 | 3 |
| `_SCALE_STRIDE` | 8 | 4 |
| `_D` | 512 | 256 |
| `bytes_per_token` | 584 | 324 |

导致 `_gather_and_dequant` 函数在读取 FP8 KV cache 时地址越界，dequantize 结果全为 NaN，最终 logits 全为 NaN。

### Diff

```diff
--- a/sglang/srt/layers/attention/flash_mla_sm120.py
+++ b/sglang/srt/layers/attention/flash_mla_sm120.py
@@ -18,21 +18,30 @@
 logger = logging.getLogger(__name__)
 
 # Page layout constants for DSv4-Flash (MODEL1):
-#   nope_dim = 448, rope_dim = 64, quantize_block_size = 64
-#   nope_rope_stride = 448 + 64*2 = 576 bytes per token
-#   scale_stride = ceil(448/64) + 1 = 8 bytes per token (7 scales + 1 pad)
-#   bytes_per_token = 448 + 128 + 8 = 584
-#   page_bytes = ceil_div(page_size * 584, 576) * 576
-
-_NOPE_DIM = 448
+# These are dynamically computed from the k_cache shape.
+# Original DSv4: nope_dim=448, rope_dim=64, head_dim=512, bytes_per_token=584
+# Pruned head_dim=256: nope_dim=192, rope_dim=64, head_dim=256, bytes_per_token=324
+
 _ROPE_DIM = 64
-_NOPE_ROPE_STRIDE = _NOPE_DIM + _ROPE_DIM * 2  # 576
 _TILE_SIZE = 64
-_NUM_TILES = _NOPE_DIM // _TILE_SIZE  # 7
-_SCALE_STRIDE = _NUM_TILES + 1  # 8 (7 scales + 1 pad)
-_D = _NOPE_DIM + _ROPE_DIM  # 512
 
 
+def _compute_layout(bytes_per_token: int):
+    """Compute page layout constants from bytes_per_token.
+
+    bytes_per_token = dim_nope + dim_rope*2 + (dim_nope // 64 + 1)
+    Since dim_nope = D - 64 and dim_nope must be divisible by 64:
+      bytes_per_token = D + 64 + D//64
+      => D = (bytes_per_token - 64) * 64 // 65
+    """
+    _D = (bytes_per_token - _ROPE_DIM) * _TILE_SIZE // (_TILE_SIZE + 1)
+    _NOPE_DIM = _D - _ROPE_DIM
+    _NOPE_ROPE_STRIDE = _NOPE_DIM + _ROPE_DIM * 2  # = _D
+    _NUM_TILES = _NOPE_DIM // _TILE_SIZE
+    _SCALE_STRIDE = _NUM_TILES + 1
+    return _NOPE_DIM, _NOPE_ROPE_STRIDE, _NUM_TILES, _SCALE_STRIDE, _D
+
+
 def _gather_and_dequant(k_cache, indices, page_size):
     """Gather KV entries from the paged buffer using correct page-internal addressing.
 
@@ -45,6 +54,10 @@
     Returns:
         kv: (..., _D) bfloat16, dequantized KV vectors
     """
+    # Compute layout dynamically from cache shape
+    bytes_per_token = k_cache.shape[3]
+    NOPE_DIM, NOPE_ROPE_STRIDE, NUM_TILES, SCALE_STRIDE, D = _compute_layout(bytes_per_token)
+
     idx_shape = indices.shape
     flat_idx = indices.reshape(-1)  # (N,)
     N = flat_idx.shape[0]
@@ -65,33 +78,33 @@
     )  # (num_pages, page_bytes) uint8
 
     # Compute byte offsets within each page
-    # NOPE: page[safe_page, safe_offset * 576 + 0:448]
-    # ROPE: page[safe_page, safe_offset * 576 + 448:576]
-    # SCALES: page[safe_page, page_size * 576 + safe_offset * 8 + 0:7]
-
-    nope_base = safe_offsets * _NOPE_ROPE_STRIDE  # (N,)
+    nope_base = safe_offsets * NOPE_ROPE_STRIDE  # (N,)
     nope_offsets = nope_base.unsqueeze(-1) + torch.arange(
-        _NOPE_DIM, device=device, dtype=torch.long
-    )  # (N, 448)
+        NOPE_DIM, device=device, dtype=torch.long
+    )  # (N, NOPE_DIM)
 
-    rope_base = nope_base + _NOPE_DIM  # (N,)
+    rope_base = nope_base + NOPE_DIM  # (N,)
     rope_offsets = rope_base.unsqueeze(-1) + torch.arange(
         _ROPE_DIM * 2, device=device, dtype=torch.long
-    )  # (N, 128)
+    )  # (N, ROPE_DIM*2)
 
-    scale_section_offset = page_size * _NOPE_ROPE_STRIDE  # 147456
-    scale_base = scale_section_offset + safe_offsets * _SCALE_STRIDE  # (N,)
+    scale_section_offset = page_size * NOPE_ROPE_STRIDE
+    scale_base = scale_section_offset + safe_offsets * SCALE_STRIDE  # (N,)
     scale_offsets = scale_base.unsqueeze(-1) + torch.arange(
-        _NUM_TILES, device=device, dtype=torch.long
-    )  # (N, 7)
+        NUM_TILES, device=device, dtype=torch.long
+    )  # (N, NUM_TILES)
 
     # Gather bytes per page
     page_idx_nope = safe_pages.unsqueeze(-1).expand_as(nope_offsets)
-    nope_bytes = raw_pages[page_idx_nope, nope_offsets]  # (N, 448) uint8
+    nope_bytes = raw_pages[page_idx_nope, nope_offsets]  # (N, NOPE_DIM)
 
     page_idx_rope = safe_pages.unsqueeze(-1).expand_as(rope_offsets)
-    rope_bytes = raw_pages[page_idx_rope, rope_offsets]  # (N, 128) uint8
+    rope_bytes = raw_pages[page_idx_rope, rope_offsets]  # (N, ROPE_DIM*2)
 
     page_idx_scale = safe_pages.unsqueeze(-1).expand_as(scale_offsets)
-    scale_bytes = raw_pages[page_idx_scale, scale_offsets]  # (N, 7) uint8
+    scale_bytes = raw_pages[page_idx_scale, scale_offsets]  # (N, NUM_TILES)
 
     # Reinterpret dtypes
-    nope_fp8 = nope_bytes.view(torch.float8_e4m3fn)  # (N, 448)
-    rope_bf16 = rope_bytes.contiguous().view(torch.bfloat16)  # (N, 64)
-    scale_e8m0 = scale_bytes.view(torch.float8_e8m0fnu)  # (N, 7)
+    nope_fp8 = nope_bytes.view(torch.float8_e4m3fn)  # (N, NOPE_DIM)
+    rope_bf16 = rope_bytes.contiguous().view(torch.bfloat16)  # (N, ROPE_DIM)
+    scale_e8m0 = scale_bytes.view(torch.float8_e8m0fnu)  # (N, NUM_TILES)
 
     # Dequantize: nope_tile * scale_tile → bf16 (vectorized)
-    result = torch.empty(N, _D, dtype=torch.bfloat16, device=device)
-    result[:, :_NOPE_DIM] = (
+    result = torch.empty(N, D, dtype=torch.bfloat16, device=device)
+    result[:, :NOPE_DIM] = (
         (
-            nope_fp8.view(N, _NUM_TILES, _TILE_SIZE).float()
-            * scale_e8m0.view(N, _NUM_TILES, 1).float()
+            nope_fp8.view(N, NUM_TILES, _TILE_SIZE).float()
+            * scale_e8m0.view(N, NUM_TILES, 1).float()
         )
-        .view(N, _NOPE_DIM)
+        .view(N, NOPE_DIM)
         .to(torch.bfloat16)
     )
-    result[:, _NOPE_DIM:] = rope_bf16
+    result[:, NOPE_DIM:] = rope_bf16
 
-    return result.reshape(*idx_shape, _D)
+    return result.reshape(*idx_shape, D)
```

---

## 改动 2：model_config.py — 修复 qk_rope_head_dim 读取

**文件路径**：`sglang/srt/configs/model_config.py`

### 问题

transformers 库的 `DeepseekV4Config.__post_init__` 通过 `int(head_dim * partial_rotary_factor)` 计算 `qk_rope_head_dim`。  
原始模型 `partial_rotary_factor = 64/512 = 0.125`，裁剪后 `head_dim=256`，导致 `qk_rope_head_dim = int(256 * 0.125) = 32`（错误，应为 64）。

这会引起下游 attention 计算中 RoPE 维度不匹配，导致 reshape 错误和 logits 为 NaN。

### Diff

```diff
--- a/sglang/srt/configs/model_config.py
+++ b/sglang/srt/configs/model_config.py
@@ -772,7 +772,34 @@
         elif (
             "DeepseekV4ForCausalLM" in self.hf_config.architectures
             or "DeepseekV4ForCausalLMNextN" in self.hf_config.architectures
         ):
-            self.qk_rope_head_dim = self.hf_config.qk_rope_head_dim
+            # transformers DeepseekV4Config computes qk_rope_head_dim as
+            # int(head_dim * partial_rotary_factor), which can be wrong for
+            # pruned models where head_dim changed but partial_rotary_factor
+            # wasn't updated. Read the raw value from config.json.
+            raw_qk_rope = None
+            try:
+                import json as _json
+                import os as _os
+
+                cfg_path = _os.path.join(self.model_path, "config.json")
+                if _os.path.isfile(cfg_path):
+                    with open(cfg_path) as f:
+                        raw_cfg = _json.load(f)
+                    raw_qk_rope = raw_cfg.get("qk_rope_head_dim")
+            except Exception:
+                pass
+            if raw_qk_rope is not None:
+                self.qk_rope_head_dim = int(raw_qk_rope)
+                # Fix the HF config object so downstream code sees the right value
+                self.hf_config.qk_rope_head_dim = self.qk_rope_head_dim
+                if self.hf_config is not self.hf_text_config:
+                    self.hf_text_config.qk_rope_head_dim = self.qk_rope_head_dim
+                # Also fix partial_rotary_factor for consistency
+                self.hf_config.partial_rotary_factor = (
+                    self.qk_rope_head_dim / self.hf_config.head_dim
+                )
+            else:
+                self.qk_rope_head_dim = self.hf_config.qk_rope_head_dim
             self.qk_nope_head_dim = self.hf_config.head_dim - self.qk_rope_head_dim
             self.window_size = self.hf_config.sliding_window
             self.head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
```

---

## 改动 3：sgl_infer_dpsk_v4.sh — 禁用 Triton 内核，使用 PyTorch fallback

**文件路径**：`sgl_infer_dpsk_v4.sh`

### 问题

`flash_mla_sm120_triton.py` 中的 Triton kernel 同样硬编码了 `head_dim=512` 的页内布局（Triton `constexpr` 常量在编译期确定），无法适配裁剪模型。

### Diff

```diff
--- a/sgl_infer_dpsk_v4.sh
+++ b/sgl_infer_dpsk_v4.sh
@@ -16,6 +16,7 @@
 # export SGLANG_DSV4_FP4_EXPERTS=0  # 关闭 FP4 expert 路径
 export SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=0
+export SGLANG_SM120_TRITON_FLASHMLA=0  # 使用 PyTorch fallback，避免 Triton 内核的硬编码 head_dim=512 布局
```

---

## 改动总结

| # | 文件 | 改动类型 | 说明 |
|---|------|---------|------|
| 1 | `sglang/srt/layers/attention/flash_mla_sm120.py` | 核心修复 | 将硬编码的 KV cache 页内布局常数改为从 `bytes_per_token` 动态推导，新增 `_compute_layout()` 函数 |
| 2 | `sglang/srt/configs/model_config.py` | 配置修复 | DeepseekV4 分支新增从 `config.json` 直接读取 `qk_rope_head_dim` 的逻辑，绕过 transformers 的错误计算 |
| 3 | `sgl_infer_dpsk_v4.sh` | 启动脚本 | 添加 `SGLANG_SM120_TRITON_FLASHMLA=0` 环境变量，禁用同样有硬编码问题的 Triton 内核 |

## 布局公式推导

对于 DeepSeek V4 系列 FP8 KV cache，每 token 字节数公式为：

```
bytes_per_token = D + ROPE_DIM + NOPE_DIM // TILE_SIZE + 1
```

其中 `NOPE_DIM = D - ROPE_DIM`，且 `NOPE_DIM` 必须被 `TILE_SIZE(64)` 整除，化简为：

```
bytes_per_token = D + 64 + D // 64
```

反解 D：

```
D = (bytes_per_token - 64) * 64 // 65
```

验证：
- 原始 DSv4：`bytes_per_token=584` → `D = (584-64)*64//65 = 512` ✓
- 裁剪模型：`bytes_per_token=324` → `D = (324-64)*64//65 = 256` ✓

---

# 附录：H20 (SM90) 环境的额外适配

> 以上改动 1~3 均针对 **RTX 5090 (SM 12.0 / Blackwell)** 验证。
> 将同一 conda 环境迁移到 **H20 (SM 9.0 / Hopper)** 部署裁剪模型时，会遇到新的报错，
> 需要以下额外适配。原始改动 1~3 依然保留（H20 仍复用改动 1 的动态 layout 逻辑）。

## 背景：SM120 与 SM90 走不同的 attention 代码路径

运行环境：NVIDIA H20 (SM 9.0 / Hopper)，SGLang v0.5.14

H20 部署裁剪模型（head_dim=256）报错：

```
File ".../sgl_kernel/flash_mla.py", line 269, in _flash_mla_with_kvcache_sched_meta
    torch.ops.sgl_kernel.sparse_decode_fwd.default(...)
RuntimeError: Only head_size_k == 576 or 512 is supported for sparse decoding
```

根因在 `deepseek_v4_backend.py` 的架构分发逻辑：

```python
# line 71
_is_sm120 = is_sm120_supported()

# line 1414 (decode 路径)
if _is_sm120:                                    # RTX 5090 → True
    from ...flash_mla_sm120 import flash_mla_with_kvcache_sm120
    o = flash_mla_with_kvcache_sm120(...)        # 纯 PyTorch，可 patch (改动 1)
else:                                            # H20 → False
    import sgl_kernel.flash_mla as flash_mla
    o = flash_mla.flash_mla_with_kvcache(...)    # 编译型 CUDA kernel，硬编码 head_size_k==576/512
```

| | 源 (5090/SM120) | 目标 (H20/SM90) |
|---|---|---|
| `_is_sm120` | True | False |
| 走的路径 | `flash_mla_sm120.py`（纯 PyTorch，改动 1 生效） | `sgl_kernel.sparse_decode_fwd`（编译型，硬编码 576/512） |
| 裁剪模型 (head_dim=256) | 能跑 | 报 `head_size_k` 错误 |

改动 1~3 的 patch 全部位于 SM120 的纯 PyTorch 路径，H20 默认根本不走那条路，
而是走编译好的 `sparse_decode_fwd` CUDA kernel（无法通过改 Python 源码修复）。

---

## 改动 4：deepseek_v4_backend.py — 允许强制走 SM120 纯 PyTorch 路径

**文件路径**：`sglang/srt/layers/attention/deepseek_v4_backend.py`

### 问题

H20 (SM90) 默认走编译型 `sparse_decode_fwd`，硬编码 `head_size_k==576/512`，裁剪模型 (256) 被拒绝。
而 `flash_mla_with_kvcache_sm120` 的 torch 后端 (`_sm120_sparse_decode_fwd`) 是**纯 PyTorch、设备无关**实现，
H20 上同样能跑（只是比编译 kernel 慢）。因此只需让 H20 也进 `if _is_sm120:` 分支。

### Diff

```diff
--- a/sglang/srt/layers/attention/deepseek_v4_backend.py
+++ b/sglang/srt/layers/attention/deepseek_v4_backend.py
@@ -71 +71,3 @@
-_is_sm120 = is_sm120_supported()
+import os as _os
+# 允许通过环境变量强制走 SM120 纯 PyTorch 路径（供 H20/SM90 部署裁剪模型使用）
+_is_sm120 = is_sm120_supported() or _os.environ.get("SGLANG_FORCE_SM120_FLASHMLA", "0") == "1"
```

### 一致性说明

`_is_sm120` 在该文件有两处用到，强制 True 后均自洽：

| 位置 | 强制 True 后行为 | 是否正确 |
|------|-----------------|---------|
| line 123 `_create_flashmla_metadata()` | 返回 `None` | SM120 路径本就不用 metadata |
| line 1414 decode 分支 | 走 `flash_mla_with_kvcache_sm120`（纯 PyTorch） | 设备无关，H20 能跑 |

SM120 的 decode 分支直接用 `indices` / `topk_length`，不依赖 `flashmla_metadata`，因此 line 123 返回 `None` 无影响。

---

## 改动 5：启动脚本 — H20 专用环境变量

**文件路径**：H20 上的 SGLang 启动脚本（如 `sgl_infer_dpsk_v4.sh`）

### Diff

```diff
+# ---- H20 (SM90) 专用：强制走 SM120 纯 PyTorch fallback ----
+export SGLANG_FORCE_SM120_FLASHMLA=1   # 让 H20 的 _is_sm120=True，绕开编译型 sparse_decode_fwd
+export SGLANG_SM120_TRITON_FLASHMLA=0  # 在 SM120 分支内选纯 PyTorch，不用 Triton（Triton 有硬编码 head_dim=512）
```

| 环境变量 | 作用 |
|---|---|
| `SGLANG_FORCE_SM120_FLASHMLA=1` | 配合改动 4，让 H20 进 SM120 分支 |
| `SGLANG_SM120_TRITON_FLASHMLA=0` | 选纯 PyTorch `_sm120_sparse_decode_fwd`（改动 1 已 patch），避开硬编码的 Triton kernel |

---

## 改动 6：启动参数 — H20 禁用 CUDA graph

**文件路径**：H20 上的 SGLang 启动脚本

### 问题

强制走的 `_sm120_sparse_decode_fwd` 是纯 PyTorch + 数据依赖的 gather/index 操作，
与 SGLang 启动时的 **CUDA graph 捕获（prewarm）不兼容**，会在启动阶段卡死
（日志停在 `cutlass.cute.experimental` 警告后，GPU util 长时间无进展，收不到请求也不报错）。

### Diff

```diff
+# H20 纯 PyTorch sparse decode 与 CUDA graph 捕获不兼容，必须禁用
+--disable-cuda-graph
```

禁用后走 eager 执行，虽更慢，但纯 PyTorch 路径本就慢，先保证能跑通。

---

## H20 适配总结

| # | 文件 / 位置 | 改动类型 | 说明 |
|---|------|---------|------|
| 4 | `sglang/srt/layers/attention/deepseek_v4_backend.py` (line 71) | 分发修复 | `_is_sm120` 增加 `SGLANG_FORCE_SM120_FLASHMLA` 环境变量开关，让 H20 也走 SM120 纯 PyTorch 路径 |
| 5 | H20 启动脚本 | 环境变量 | `SGLANG_FORCE_SM120_FLASHMLA=1` + `SGLANG_SM120_TRITON_FLASHMLA=0` |
| 6 | H20 启动脚本 | 启动参数 | `--disable-cuda-graph`，避免纯 PyTorch sparse decode 在 CUDA graph 捕获时卡死 |

### 部署验证顺序（H20）

1. 按改动 4 修改 `deepseek_v4_backend.py`（改前先备份 `.bak`）
2. 启动脚本加改动 5 的两个环境变量 + 改动 6 的 `--disable-cuda-graph`
3. 启动后发 1 条请求小并发验证：不再报 `head_size_k==576/512`，输出正常（logits 非 NaN）
4. 确认正确性后再上量

### 硬件差异对照

| 项 | RTX 5090 | H20 |
|---|---|---|
| Compute Capability | 12.0 (SM120 / Blackwell) | 9.0 (SM90 / Hopper) |
| `_is_sm120`（默认） | True | False（需改动 4 强制 True） |
| sparse decode 实现 | SM120 纯 PyTorch / Triton | 默认编译 kernel（硬编码）→ 改后用 SM120 纯 PyTorch |
| CUDA graph | 可用 | 纯 PyTorch 路径需禁用 |

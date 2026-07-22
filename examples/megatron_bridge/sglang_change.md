# SGLang 适配 DeepSeek-V4 裁剪模型 (head_dim=256) 修改记录

## 背景

裁剪后的 DeepSeek-V4 模型将 head_dim 从 512 缩减至 256（qk_nope_head_dim=192, qk_rope_head_dim=64）。
SGLang 的 DSV4 后端针对 head_dim=512 做了大量硬编码优化，需要逐层解除限制。

## 验证结果

**服务已成功拉起。** 通过 `/data/dpsk-v4-run.sh` 启动后：
- 服务健康检查通过（HTTP 200）
- 请求时触发 "illegal instruction" CUDA 错误（已修复，见下方第二轮修改）

## 已完成的修改

### 1. Python 层断言移除

| 文件 | 修改内容 |
|------|----------|
| `srt/mem_cache/deepseek_v4_memory_pool.py` | 移除 `assert bytes_per_token == 448 + 64*2 + 8` 硬编码断言 |
| `srt/layers/attention/deepseek_v4_backend.py` | 移除 `assert head_dim == 512` 断言；添加 `qk_nope_head_dim`/`qk_rope_head_dim` 属性 |
| `srt/layers/attention/dsv4/index_buf_accessor.py` | 移除 `NopeFp8RopeBf16Pack.__post_init__` 中 shape==448/64/7 的断言 |

### 2. Triton kernel 参数化

| 文件 | 修改内容 |
|------|----------|
| `srt/layers/attention/dsv4/quant_k_cache.py` | `quant_to_nope_fp8_rope_bf16_pack_triton()` 新增 `dim_nope`/`dim_rope` 参数，从输入 tensor shape 自动推导 |
| `srt/layers/attention/dsv4/dequant_k_cache.py` | `dequantize_k_cache_paged()` 新增 `dim_nope`/`dim_rope` 参数，替代模块级常量 |
| `srt/layers/attention/deepseek_v4_backend.py` | `dequantize_k_cache_paged` 调用处传入实际 dims；`store_cache()` 强制使用 triton quant 路径 |

### 3. JIT CUDA kernel 绕行（第一轮）

| 文件 | 修改内容 |
|------|----------|
| `srt/models/deepseek_v4.py` | `_compute_q_b()` 对 head_dim!=512 使用 `fused_norm_rope_inplace_triton` 替代 JIT kernel |
| `srt/models/deepseek_v4.py` | `_compute_kv_to_cache()` 改为使用 `fused_norm_rope_inplace_triton` + `store_cache()` 替代 JIT `fused_k_norm_rope_flashmla` |
| `srt/layers/attention/dsv4/compressor_v2.py` | `forward_unified()` 强制走 `_forward_unified_hip` triton 路径；store 路径强制使用 triton quant |

### 4. FlashMLA 预编译 kernel 绕行（PyTorch fallback）

| 文件 | 修改内容 |
|------|----------|
| `srt/layers/attention/deepseek_v4_backend.py` | 新增 `_forward_decode_pytorch_fallback()` 方法；decode 路径检测 `head_dim_k not in (512, 576)` 时走 fallback |

**Fallback 实现逻辑**：
1. 使用 `dequantize_k_cache_paged()` 将 FP8 paged KV cache 反量化为 BF16
2. 拼接 SWA cache 和 extra (C4/C128) cache
3. 使用 PyTorch einsum 计算 attention scores：`scores = einsum("bhd,bkd->bhk")`
4. 应用 softmax_scale、attn_sink、length mask
5. Softmax + einsum 计算输出：`o = einsum("bhk,bkd->bhd")`

**关键修复**：
- `swa_topk_lengths`/`extra_topk_lengths` 为 1D tensor `[batch_size]`，使用 `.view(-1)` 而非 `.squeeze(1)`
- `extra_k_cache` 经 view 后非连续，使用 `.reshape().contiguous()` 确保内存连续
- `swa_page_indices` 为 page-level block table，需展开为 token-level indices：`token_id = block_id * page_size + offset`

### 5. JIT CUDA kernel 绕行（第二轮 — 修复请求崩溃）

服务拉起后请求时触发 "illegal instruction" CUDA 错误。根因：多个 JIT CUDA kernel 在 H20 GPU 上执行时产生非法指令。
通过 `CUDA_LAUNCH_BLOCKING=1` 定位后，逐一替换为 PyTorch/Triton 等价实现。

| 文件 | 修改内容 |
|------|----------|
| `jit_kernel/dsv4/compress.py` | `compress_forward()` 对 head_dim!=512 路由到 `_compress_forward_pytorch()` PyTorch fallback；优化 plan 数据预取到 CPU 避免逐元素 `.item()` CUDA 同步 |
| `srt/layers/attention/dsv4/indexer.py` | 新增 `_fused_q_indexer_rope_hadamard_quant_pytorch()` 函数（RoPE + Hadamard + FP8 quant）；`C4Indexer.compute_q()` 使用该 fallback 替代 JIT kernel |
| `srt/layers/attention/dsv4/indexer.py` | `topk_transform_512` 强制使用 `topk_transform_512_pytorch_vectorized` PyTorch fallback |
| `srt/layers/attention/dsa/dsa_indexer.py` | 新增 `_hadamard_pytorch()` 函数（Fast Walsh-Hadamard Transform）；`rotate_activation()` 使用该 fallback 替代 JIT `hadamard_transform` |

**`_fused_q_indexer_rope_hadamard_quant_pytorch` 实现逻辑**：
1. RoPE：对 q 的最后 rope_dim=64 维应用旋转位置编码
2. Hadamard：128 点 Fast Walsh-Hadamard 变换（无归一化）
3. FP8 量化：per-(token, head) 计算 max_abs → scale → clamp → float8_e4m3fn
4. 权重输出：`weights_out = weight * weight_scale * q_scale`

**`_compress_forward_pytorch` 实现逻辑**：
- Decode：从 plan_d 解析 write_loc/read_page，写入新 token，boundary 处做 softmax 加权压缩
- Prefill：从 plan_w 写入新 token 到 buffer，从 plan_c 读取 page 做 softmax 加权压缩

### 6. 修改原因

SGLang DSV4 后端有多层 JIT CUDA kernel：
- `main_norm_rope.cuh` — `FusedKNormRopeFlashMLAKernel`: 硬编码 `kHeadDim == 512`
- `main_norm_rope.cuh` — `FusedQIndexerRopeHadamardQuantKernel`: 硬编码 `kHeadDim == 128, kRopeDim == 64`
- `c4.cuh` / `c128.cuh` — `FlashCompress4/128Kernel`: 压缩 kernel
- `fused_norm_rope.cuh` — `FusedNormRopeKernel`: 压缩后 norm+rope
- `hadamard.cuh` — Hadamard 变换

FlashMLA 预编译二进制 kernel（`flashmla_ops.abi3.so`）：
- `sparse_decode_fwd`: 仅支持 head_size_k == 512 或 576
- `sparse_prefill_fwd`: 同上

在 H20 GPU (sm_90) 上，部分 JIT kernel 即使编译通过也会在执行时触发 "illegal instruction"。
所有 JIT kernel 均通过 Python 层绕行到 PyTorch/Triton 等价实现。

## FlashMLA 分析

### 是否使用 FlashMLA？

**是。** 虚拟环境 `/data/dpsk-v4/` 中的 `sgl_kernel` 包含预编译的 `flashmla_ops.abi3.so`（14MB），
通过 `torch.ops.sgl_kernel.sparse_decode_fwd` / `sparse_prefill_fwd` 调用。

`SGLANG_DISABLE_JIT_KERNEL=1` 环境变量在 SGLang 代码中**不存在**，无任何效果。

### 实际 MLA 逻辑路径

```
forward() → _forward_decode_pytorch_fallback()  [head_dim=256 时]
         → flash_mla_with_kvcache()             [head_dim=512/576 时]
             → torch.ops.sgl_kernel.sparse_decode_fwd  (预编译 CUDA)
```

## 修改文件清单

```
/usr/local/lib/python3.12/dist-packages/sglang/srt/mem_cache/deepseek_v4_memory_pool.py
/usr/local/lib/python3.12/dist-packages/sglang/srt/layers/attention/deepseek_v4_backend.py
/usr/local/lib/python3.12/dist-packages/sglang/srt/layers/attention/dsv4/index_buf_accessor.py
/usr/local/lib/python3.12/dist-packages/sglang/srt/layers/attention/dsv4/quant_k_cache.py
/usr/local/lib/python3.12/dist-packages/sglang/srt/layers/attention/dsv4/dequant_k_cache.py
/usr/local/lib/python3.12/dist-packages/sglang/srt/layers/attention/dsv4/compressor_v2.py
/usr/local/lib/python3.12/dist-packages/sglang/srt/layers/attention/dsv4/indexer.py
/usr/local/lib/python3.12/dist-packages/sglang/srt/layers/attention/dsa/dsa_indexer.py
/usr/local/lib/python3.12/dist-packages/sglang/srt/layers/deepseek_v4_rope.py
/usr/local/lib/python3.12/dist-packages/sglang/srt/models/deepseek_v4.py
/usr/local/lib/python3.12/dist-packages/sglang/jit_kernel/dsv4/compress.py
```

## 性能说明

PyTorch fallback 使用纯 PyTorch 算子实现，相比 JIT CUDA kernel：
- 无 kernel fusion（多步操作分离执行）
- 无 warp-level 优化（Hadamard 使用 tensor view 而非 shuffle）
- 适用于验证和功能测试，不适合生产部署

生产环境建议：
- 向 sgl_kernel 上游提交 head_dim=256 支持
- 或实现 triton-based sparse MLA attention kernel
- 或修复 H20 上 JIT kernel 的 "illegal instruction" 问题（可能是 PDL 相关）

## 测试方法

```bash
# 启动服务（建议加 CUDA_LAUNCH_BLOCKING=1 便于调试）
source /data/dpsk-v4/bin/activate
export CUDA_LAUNCH_BLOCKING=1
sglang serve --trust-remote-code --model-path /data/output/v4-pruned-nas-final-sglang \
  --tp 4 --moe-runner-backend marlin --disable-cuda-graph \
  --mem-fraction-static 0.75 --served-model-name dpsk_v4_width \
  --host 0.0.0.0 --port 30001

# 测试请求
curl http://localhost:30001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"dpsk_v4_width","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```

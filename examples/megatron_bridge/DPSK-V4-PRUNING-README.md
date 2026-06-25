# DPSK-V4 激活参数 L2 范数 NAS 剪枝流程

> 本文档整理 `examples/megatron_bridge/` 下的 DeepSeek-V4 剪枝工作流。
> 适用模型：**DeepSeek-V4-Flash**（约 284B 参数，MLA + MoE + mHC + MTP）。
> 核心思路：用 **激活参数的 L2 范数** 估计各通道重要性，通过 **NAS** 搜索在给定 `target_params_ratio` 下最佳的 `(hidden_size, num_layers, num_moe_experts, moe_ffn_hidden_size)` 组合。

---

## 1. 原理

### 1.1 重要性估计：激活 L2 范数

剪枝的核心是决定**哪些通道（dimension）可以丢掉**。本流程采用基于 activation 的 importance（而非纯 weight magnitude），具体公式：

```
对每个 transformer layer 的 input_layernorm / pre_mlp_layernorm 的输出：
    act = layernorm_output           # shape [seq, batch, hidden]
    act = act.abs().mean(dim=0)      # 在 seq 维度求平均  → [batch, hidden]
    importance[hidden] = (act ** 2).sum(dim=0)   # 跨 batch 求 L2 范数平方  → [hidden]
```

实现位于 `prune_deepseek_v4.py` 的 `forward_loop()`：

```python
def _layernorm_hook(mod, input, output):
    act = output.detach().float()
    if act.dim() == 3:                 # [seq, batch, hidden]
        act = act.abs().mean(dim=0)    # [batch, hidden]
    elif act.dim() == 2:
        act = act.abs()
    scores = act.pow(2).sum(dim=0)     # [hidden]
    ...
```

跨所有 calibration 样本累加后，再交给 `mcore_minitron` 的 `sort_parameters` 取 top-K 通道作为保留索引。这一步输出 `hidden_size_order`，后续 `_apply_v4_post_prune_slicing` 用来切 V4 特有的 Compressor / Indexer / Hyper-Connection 权重。

### 1.2 可剪枝维度

`mcore_minitron` 的 `SUPPORTED_HPARAMS` 对 DeepSeek-V4 实际生效的有 5 个：

| HParam                  | 含义                        | 范围（参考）      |
| ----------------------- | --------------------------- | ----------------- |
| `hidden_size`           | 宽度（MLA + MoE）           | 512..orig         |
| `num_layers`            | 深度                        | 8..orig           |
| `num_moe_experts`       | MoE 专家数                  | 16..orig（8 对齐）|
| `moe_ffn_hidden_size`   | Routed expert FFN 维度      | 256..orig（128 对齐）|
| `moe_shared_expert_intermediate_size` | Shared expert FFN 维度 | 跟随 ffn 缩放     |

**故意不剪**：`q_lora_rank`、`kv_lora_rank`、`o_lora_rank`（MLA LoRA）、`compress_ratios`（CSA/HCA 压缩比）、`num_residual_streams`（mHC 流数）。

### 1.3 NAS 搜索

`prune_deepseek_v4_nas.py` 用**启发式（heuristic）评分** + **ratio 贴近排序**：

1. **生成候选** (`generate_candidates`)：对每个维度独立枚举缩放档位（hidden 100%/90%/.../50%，ffn 100%/.../25%，layers 100%/.../40%，experts 100%/.../12.5%），笛卡尔积 ≈ 数百个组合。
2. **过滤** (`abs(ratio - target) <= max(0.05, 0.5 * target)`)：丢弃远离 target ratio 的组合，避免搜索空间爆炸。
3. **参数估算** (`compute_model_params`)：近似估算每个候选的参数量：

   ```
   attn     = L * 12 * H^2
   moe      = L * E * 3 * F * H
   shared   = L * 3 * S * H
   router   = L * E * H
   total    = attn + moe + shared + router
   ratio    = total / orig_total
   ```

4. **排序**：主键 `|ratio − target|`（越小越前），次键 `-quality_score`（越高越前，平局用）。
5. **quality_score**：综合 closeness-to-target、balance（避免单维度极端缩减）、expert retention（保留 MoE 稳定性）。
6. **选 top N** 作为最终评估候选，当前实现直接选 top 1（`num_candidates=10` 时前 3 个，`top_candidates = candidates[:3]`）。

> **注意**：当前默认 `--enable_forward_validation=False`，因此 NAS 只走启发式排序、不实际跑 pruned forward（rank 7 forward 在 mHC + MTP 链路上存在 hang 问题，详见 §5）。

### 1.4 实际剪枝：`mcore_minitron` 流程

`prune_deepseek_v4.py` 的核心调用：

```python
ss_config = {
    "hidden_size": target_hidden,
    "num_layers": target_layers,
    "num_moe_experts": target_experts,
    "moe_ffn_hidden_size": target_ffn,
    "moe_shared_expert_intermediate_size": target_shared,
}
unwrapped_model, _ = mtp.prune(
    unwrapped_model,
    mode=[("mcore_minitron", ss_config)],
    constraints={"export_config": export_config},
    dummy_input=None,
    config={"forward_loop": forward_loop, "skip_sorting": False},
)
```

执行顺序：
1. `forward_loop`：跑 calibration 数据，触发 `_layernorm_hook` 收集激活 L2 范数。
2. `sort_parameters`：按 importance 对 hidden 通道排序，输出 `hidden_size_order`。
3. 切 standard weights（embedding、MLA、MoE FFN、router）。
4. `drop_mcore_language_model_layers`：按 layer 重要性 drop 掉指定 layer，重编号 layer_number。
5. `_apply_v4_post_prune_slicing`：切 V4 特有权重（Compressor、Indexer、Hyper-Connection fn weights），用 `hidden_size_order` 取 top-K 通道。
6. `_rebuild_pp_layout_after_prune`：根据 prune 后 `num_layers` 重新算 `pipeline_model_parallel_layout`（`Etttt|...|tttmL` 形式），并通过 `PipelineParallelLayerLayout.from_str` 写到 `config` 上。
7. Lazy mHC 补丁：新持有 final layernorm 的 PP stage 补建 `hc_head_fn / hc_head_base / hc_head_scale`，并对齐到现有参数的 device/dtype。
8. 保存 per-rank shard `pruned_model_rank{r}.pt` 到 `${OUTPUT_BASE}-final_megatron/`。

### 1.5 HF 转换

`convert_pruned_to_hf.py`：
- 加载 `${OUTPUT_BASE}-final_megatron/pruned_model_rank{0..7}.pt` shards。
- 取 **最大的 shard 作 base** + 从其他 shard 补齐缺失 key（PP 切分下每 rank 持有不同 layer 子集，重叠 key 的最终值以 base 为准）。
- 从原始 MXFP4 checkpoint 恢复 routed expert weights（剪枝时不写 expert BF16，节省 ~98% state_dict 空间）。
- 输出 SGLang / vLLM 可直接加载的 HF 格式到 `${OUTPUT_BASE}-final-sglang/`。

---

## 2. 使用方式

### 2.1 一键运行（推荐）

```bash
bash /opt/Model-Optimizer/examples/megatron_bridge/run_nas_pruning.sh
```

`run_nas_pruning.sh` 包含三步：

| 步骤 | 脚本                                  | 作用                                       |
| ---- | ------------------------------------- | ------------------------------------------ |
| 1    | `prune_deepseek_v4_nas.py`            | NAS 搜索候选，输出 `best_config.json`      |
| 2    | `prune_deepseek_v4.py`                | 用 best config 真裁剪，写 per-rank shards  |
| 3    | `convert_pruned_to_hf.py`             | shards → HF 格式，SGLang/vLLM 可加载       |

默认参数（在脚本顶部）：
```bash
MODEL_PATH="/data/.cache/models/deepseek-ai/DeepSeek-V4-Flash"
OUTPUT_BASE="/data/output/v4-pruned-nas"
PP_SIZE=8
TARGET_RATIO=0.014     # 保留 1.4% 参数 (≈4B)
NUM_GPUS=8
```

### 2.2 自定义目标比例

修改 `run_nas_pruning.sh` 顶部 `TARGET_RATIO`：
```bash
TARGET_RATIO=0.20     # 保留 20% (≈57B)
```

或直接调 CLI：
```bash
torchrun --nproc_per_node=8 prune_deepseek_v4_nas.py \
    --hf_model_name_or_path /data/.cache/models/deepseek-ai/DeepSeek-V4-Flash \
    --output_hf_path /data/output/v4-pruned-nas-best \
    --pp_size 8 \
    --target_params_ratio 0.20 \
    --num_candidates 10 \
    --calibration_samples 1024 \
    --seq_length 2048 \
    --validation_ratio 0.2 \
    --trust_remote_code
```

### 2.3 关键 CLI 参数

`prune_deepseek_v4_nas.py`：
- `--target_params_ratio`：保留参数比例（NAS 主目标）。
- `--num_candidates`：候选数量（top N 参与排序评估）。
- `--calibration_samples`：激活收集样本数。
- `--seq_length`：calibration seq 长度。
- `--max_hidden_size_reduction` / `--max_ffn_reduction` / `--max_layers_reduction` / `--max_experts_reduction`：各维度最大缩减比例（默认 0.6/0.75/0.6/0.875，足以覆盖 target 低至 ~0.5%）。
- `--enable_forward_validation`：默认 OFF。开启后会真跑 pruned forward 算 loss（当前在 PP=8 + mHC + MTP 下会 hang，详见 §5）。

`prune_deepseek_v4.py`：
- `--validation_samples`：默认 0（关闭 pruned-model validation forward）。
- 其余参数由 `best_config.json` 自动注入。

### 2.4 输出文件

```
/data/output/
├── v4-pruned-nas-best_best_config.json     # NAS 选出的最佳 config
├── v4-pruned-nas-final/                    # prune_deepseek_v4 输出的 HF dir
│   ├── config.json                         # 修剪后的 HF config
│   └── ...
├── v4-pruned-nas-final_megatron/           # Megatron per-rank shards
│   ├── pruned_model_rank0.pt
│   ├── pruned_model_rank1.pt
│   └── ...pruned_model_rank7.pt
└── v4-pruned-nas-final-sglang/             # SGLang/vLLM 可加载的最终模型
    ├── config.json
    ├── tokenizer.json
    └── *.safetensors
```

---

## 3. 环境依赖

### 3.1 硬件

- **8× GPU**（NVIDIA H100/H200 80GB 或同级别）。PP=8 切分 286B 模型单卡约 35-50GB。
- 跨 GPU 高速互联（NVLink/IB），否则 NCCL collectives 会卡。
- 磁盘 ≥ 1TB：原模型 ~570GB + 剪枝中间产物 ~50GB + 输出 ~20-50GB。

### 3.2 软件栈

| 组件                    | 版本 / 来源                                                 |
| ----------------------- | ----------------------------------------------------------- |
| Python                  | 3.12                                                        |
| CUDA                    | 12.x                                                        |
| PyTorch                 | `/usr/local/lib/python3.12/dist-packages/torch`（含 CUDA 12）|
| venv                    | `/opt/venv/bin/python3`                                     |
| Megatron-LM             | `/opt/Megatron-Bridge/3rdparty/Megatron-LM/`                |
| ModelOpt                | `/opt/Model-Optimizer/`（`PYTHONPATH=/opt/Model-Optimizer`）|
| flashinfer              | 用 `FLASHINFER_DISABLE_VERSION_CHECK=1` 绕过版本检查         |
| fast_hadamard_transform | **未安装** — `prune_deepseek_v4_nas.py` 模块级 fallback      |

### 3.3 Hadamard fallback

`fast_hadamard_transform` 不在 venv 中。`prune_deepseek_v4_nas.py` 在 module 导入阶段注入纯 PyTorch fallback（`_install_hadamard_fallback()`），把 `fast_hadamard_transform` 替换成等价的 Walsh-Hadamard 实现。否则 DSA indexer 的 `rotate_activation` 会断言失败。

### 3.4 启动前环境检查

```bash
nvidia-smi                      # 确认 8 卡可见且无残留 python 进程占用
python3 -c "import torch; print(torch.cuda.device_count())"  # 应输出 8
```

`run_nas_pruning.sh` 自带 `preflight_gpu` 检查并强制清理残留进程（前一轮未正常退出时尤其重要）。

---

## 4. 故障排查

### 4.1 卡死 / NCCL timeout

常见原因：**PP=8 下 rank 之间 desync**。脚本已通过以下方式缓解：
- `compute_validation_loss` / `_compute_pruned_validation_loss` 在 schedule 返回后插 `torch.distributed.barrier(group=pp_group)`，强制所有 rank 对齐再进 broadcast（避免 schedule 内部 P2P 只同步相邻 rank 导致错位）。
- 所有 rank 用 `_diag` 输出实时进度，包含 `[sample N] schedule begin / barrier begin / broadcast begin`。
- `_loss_func` 处理 schedule contract：返回 `(output_tensor, loss)` 元组而非裸 loss。

### 4.2 `'TransformerBlock' object has no attribute 'hc_head_fn'`

prune + 重建 PP layout 后，**新的 final-layernorm stage 不是原来的 post_process stage**，但 `TransformerBlock.__init__` 早已运行、只为原 stage 创建 `hc_head_fn / hc_head_base / hc_head_scale`。修复：在 `_rebuild_pp_layout_after_prune` 后扫描 `decoder.layers`，找到 `layer.layer_number == candidate["num_layers"]` 的 rank，懒加载补建（按 `transformer_block.py:392-402` 同样 init）到现有 param 的 device/dtype。

### 4.3 `aten.mm got two different devices cuda:N, cpu`

懒加载的 `hc_head_fn` 默认在 CPU，而 hidden_states 在 cuda。修复：取一个现有 `_blk.parameters()` 的 device/dtype 作 reference，新 param 一致创建。

### 4.4 `dist.gather_object` AttributeError

`modelopt.torch.utils.distributed` 没有 `gather_object`，直接用 `torch.distributed.gather_object`。

### 4.5 HF 转换报 `No pruned_model_rank*.pt shards found`

`prune_deepseek_v4.py` 旧版在 PP>1 时让 rank 0 gather+merge 写单个 `pruned_model.pt`，但 `convert_pruned_to_hf.py` 期望 per-rank shards。已修复为每 rank 各自写 `pruned_model_rank{r}.pt`。

### 4.6 裁剪结果远大于 target ratio

`compute_model_params` 只近似估算 attn/moe/shared/router，**没考虑**：
- MLA attention 真实参数量 ≠ `12 * H²`（实际更小，V4 用 latent dim）。
- Compressor / Indexer / Hyper-Connection 等 V4 特有权重。
- Embedding / lm_head。

这导致 `params_ratio` 估值系统性偏低。如果实际裁剪后大小跟 target 不一致，**以 HF 转换后实际 storage 大小为准**。heuristic 排序已按 `|ratio − target|`（用近似 ratio）升序选最接近的 candidate。

### 4.7 `Used a dataset size of 0 samples` 的 validation loss

`prune_deepseek_v4.py` 默认 `--validation_samples=0`，跳过 pruned-model forward validation。这是为了避开 rank 7 在 mHC + MTP 链路下的 forward hang（详见 §5）。如果打开，会看到 `Pruned validation loss: nan (on 0 samples)` 或 GPU 利用率 0/100% 模式。

---

## 5. 后续改进

### 5.1 高优先级

1. **修复 pruned-model forward path**（rank 7 在 mHC contraction → lm_head → MTP 链路上 hang / 极慢）
   - 根因待定位：怀疑是 `learned_output_contract` 在 TP=1 下产生大中间 tensor + MTP + lm_head 串行；或 `torch.compile` 的 fake tensor 传播在某些路径 graph break。
   - 修复后重新打开 `--enable_forward_validation` / `--validation_samples > 0`，NAS 可以基于真 loss 排序，精度优于 heuristic。

2. **改进 `compute_model_params` 准确性**
   - 引入 MLA attention 真实参数量公式（`≈ 3 * H² + small` for V4）。
   - 加入 Compressor / Indexer / Hyper-Connection 的 hidden 维度贡献。
   - 加入 Embedding / lm_head 的固定贡献（vocab_size × hidden_size）。
   - 校准后，`|params_ratio − target|` 排序结果能更精确反映真实 storage。

3. **支持 `--target_params_ratio < 0.005`**（极致小模型）
   - 当前最深的 ratio 组合约 0.5%，继续往下需要：
     - 引入 `num_attention_heads` 缩减（`mcore_minitron.SUPPORTED_HPARAMS` 已支持）。
     - 允许 `q_lora_rank` / `kv_lora_rank` 缩减。
     - 把 alignment（128/8）改为可配置。

### 5.2 中优先级

4. **激活重要性采集优化**
   - 当前 hook 只在 `input_layernorm` + `pre_mlp_layernorm`。可考虑加入 `q_lora_norm` / `kv_lora_norm` / attention output 等更多 hook，覆盖 MLA 特有通道。
   - 跨 rank gather 时用 all-reduce 而非 `_activations[id(mod)] += scores.cpu()`，避免每个 hook 单独同步。

5. **NAS 多目标优化**
   - 当前 heuristic 综合分 4 因子。考虑 Pareto 前沿：在 `(closeness_to_target, quality_score)` 平面上选前 K 个，让下游 pruning step 评估每个点的真实 loss。

6. **进度可观测性**
   - 加 `tqdm` / `rich` 显示 NAS 候选进度。
   - 把 `_diag` 输出统一重定向到 logging module，方便过滤 / 重定向。

### 5.3 低优先级

7. **替换 PP layout 字符串协议**
   - 当前 `_build_pp_layout` 返回 string，再 `PipelineParallelLayerLayout.from_str` 解析。直接返回 `PipelineParallelLayerLayout` 对象可省一层转换。

8. **支持 TP>1 切分**
   - 当前 `pipeline_model_parallel_size=8, tensor_model_parallel_size=1`，单 TP 跑 lm_head + MTP 是瓶颈。TP=4 / TP=8 可以把大矩阵分到多卡，rank 7 forward 时间应下降 4-8 倍。

9. **fast_hadamard_transform 真实安装**
   - 当前用 PyTorch fallback（无优化）。安装官方 CUDA 实现可以加速 calibration forward loop。

---

## 6. 相关文件

| 文件                                               | 角色                                       |
| -------------------------------------------------- | ------------------------------------------ |
| `prune_deepseek_v4_nas.py`                         | NAS 搜索入口（heuristic 排序）             |
| `prune_deepseek_v4.py`                             | 单 config 真裁剪 + per-rank shard 保存     |
| `convert_pruned_to_hf.py`                          | Megatron shards → HF 转换                  |
| `run_nas_pruning.sh`                               | 一键 wrapper（preflight + 3 步 pipeline）  |
| `merge_pruned_state_dicts.py`                      | 辅助：合并 PP shards（手工排查时用）       |
| `/opt/Model-Optimizer/modelopt/torch/prune/plugins/mcore_minitron.py` | mcore_minitron 实现             |

---

## 7. 引用

- ModelOpt mcore_minitron：基于 activation-L2-norm 的重要性估计与 structured pruning。
- Megatron-LM `PipelineParallelLayerLayout`：PP layout 字符串协议。
- DeepSeek-V4-Flash：286B MoE + MLA + mHC + MTP 大模型。
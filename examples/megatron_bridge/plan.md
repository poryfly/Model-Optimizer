# DeepSeek-V4-Flash → DeepSeek-V4-4B-A1B 裁剪与恢复方案

## 1. 目标定义

- **基础模型**: DeepSeek-V4-Flash（总参数量 ~284B，激活参数量 ~37B，MLA + MoE + mHC + MTP）
- **目标模型**: DeepSeek-V4-4B-A1B
  - 总参数量 ≈ 4B（保留比例 ~1.4%）
  - 每 token 激活参数量 ≈ 1B
  - 保留 MLA + MoE + mHC + MTP 核心架构模块
- **能力目标**: 在通用语言、代码、数学、推理等基准上接近原模型，并超过 30B 级别 dense/MoE 模型
- **技术路线**: 结构化裁剪（Minitron + V4 后处理）→ 知识蒸馏（KD）→ 继续预训练（CPT）→ 指令微调（SFT）

## 2. 阶段一：结构化裁剪到 4B 规模

### 2.1 复用现有裁剪基础设施

基于现有提交 `093cc4b1` 已提供的工具链：

```
examples/megatron_bridge/
├── prune_deepseek_v4_nas.py      # NAS 搜索
├── prune_deepseek_v4.py          # 真裁剪
├── convert_pruned_to_hf.py       # 转 HF 格式
└── run_nas_pruning.sh            # 一键 wrapper
```

新增增强脚本：
- `prune_deepseek_v4_4b_config.py`：精确 4B 总参数量配置生成
- `run_v4_4b_pipeline.sh`：端到端裁剪流水线

### 2.2 目标配置（4B 总参数）

以 `TARGET_TOTAL_PARAMS=4_000_000_000` 运行精确配置生成，从候选中选择最接近 4B 的配置。

| 维度 | 原始值 | 目标值（示例） | 备注 |
|---|---|---|---|
| `hidden_size` | 4096 | 512–1536 | 128 对齐 |
| `num_layers` | 43 | 8–16 | 2 对齐 |
| `num_moe_experts` | 256 | 16–48 | 8 对齐 |
| `moe_ffn_hidden_size` | 2048 | 256–768 | 128 对齐 |
| `moe_shared_expert_intermediate_size` | 跟随 ffn | 256–768 | 跟随 ffn 缩放 |

> 注意：新增脚本 `prune_deepseek_v4_4b_config.py` 改进了参数估算，包含 embedding、lm_head、MLA 低秩投影、Hyper-Connection、MTP 等，避免原 `compute_model_params` 的系统性低估。

### 2.3 关键修复与参数调优

- `--enable_forward_validation=False`（默认）: 当前 PP=8 + mHC + MTP forward 会 hang，NAS 仍走启发式排序。
- `--validation_samples=0`: 裁剪后不做 pruned forward validation。
- 必须执行 `_apply_v4_post_prune_slicing` 切分 Compressor/Indexer/Hyper-Connection 权重。
- 必须执行 `_rebuild_pp_layout_after_prune` 重建 PP layout。
- 输出 `${OUTPUT_BASE}-final-sglang/` 作为后续训练起点。

### 2.4 输出物

- 裁剪后 HF 模型目录：`/data/output/v4-4b-a1b-final-sglang/`
- 包含 `config.json`、`tokenizer.json`、`*safetensors`
- 模型结构保留：MLA 低秩、MoE 路由、mHC 多残差流、MTP

## 3. 阶段二：知识蒸馏（Knowledge Distillation）

### 3.1 教师与学生

- **Teacher**: 原始 DeepSeek-V4-Flash（284B）或其 SGLang/vLLM 服务化 endpoint
- **Student**: 阶段一输出的 4B 裁剪模型

### 3.2 蒸馏目标函数

新增脚本：`examples/llm_distill/distill_deepseek_v4_4b.py`

组合以下损失：

1. **Soft logit 蒸馏**
   ```
   L_logit = KL(softmax(z_t / T) || softmax(z_s / T)) * T²
   ```
2. **Hidden-state 蒸馏**（中间层对齐 + 可学习投影）
   ```
   L_hidden = MSE(h_s, proj(h_t))
   ```
3. **MoE router 蒸馏**
   ```
   L_router = KL(router_logits_t || router_logits_s)
   ```
4. **可选 CE 损失**

总损失：
```
L_total = α_kd * L_logit + α_hid * L_hidden + α_router * L_router + α_ce * L_ce
```

### 3.3 蒸馏数据

- 高质量网页文本（Filtered Web Corpus）
- 代码数据（GitHub、Stack Overflow、The Stack）
- 数学推理数据（arXiv math、OpenWebMath）
- 多语言数据

数据量建议：**100B–300B tokens**。

## 4. 阶段三：继续预训练（Continual Pre-Training, CPT）

### 4.1 目标

在蒸馏后基础上，进一步恢复语言建模能力、MoE 专家负载平衡和 MTP 预测能力。

### 4.2 数据策略

新增脚本：`examples/megatron_bridge/cpt_deepseek_v4_4b.py`

支持单数据集或多数据集混合（`configs/cpt_data_mix.json`）：
- 通用文本：40%
- 代码：25%
- 数学/科学：20%
- 多语言：10%
- 领域数据：5%

课程学习：短序列 → 长序列（4096 → 8192）

总 token 量：**200B–500B tokens**

### 4.3 训练超参

- 学习率：1e-4 ~ 3e-4（warmup 5%，cosine decay）
- Batch size：4M–8M tokens
- Sequence length：4096 → 8192
- MoE load balancing loss：保留 aux_loss
- MTP loss：保留

## 5. 阶段四：指令微调（SFT）

### 5.1 数据

新增脚本：`examples/megatron_bridge/sft_deepseek_v4_4b.py`

支持 `messages` 格式指令数据与多数据集混合（`configs/sft_data_mix.json`）：
- 通用指令：OpenHermes、UltraChat
- 代码指令：CodeAlpaca
- 数学/推理：MetaMath、GSM8K augmented

总量：1M–5M 高质量样本

### 5.2 训练策略

- 学习率：1e-5 ~ 5e-5
- Epochs：2–3
- Sequence length：4096–8192
- Packing：启用
- Chat template：沿用 DeepSeek-V4-Flash

### 5.3 可选强化阶段

- **DPO**：在偏好数据对上优化
- **RLHF**：资源允许时使用 PPO/GRPO

## 6. 阶段五：量化与部署优化

- **NVFP4/FP8 权重量化**
- **FP8/KV cache 量化**
- 输出 TensorRT-LLM / SGLang / vLLM 可加载格式

参考入口：
- `examples/llm_ptq/hf_ptq.py`
- `examples/deepseek/deepseek_v4/quantize_to_nvfp4.py`

## 7. 评估与迭代

### 7.1 评估基准

新增脚本：`examples/megatron_bridge/eval_deepseek_v4_4b.py`

- **通用能力**：MMLU、MMLU-Pro、BBH、ARC-C
- **代码能力**：HumanEval、MBPP
- **数学推理**：GSM8K、MATH
- **多语言**：MGSM
- **长上下文**：Needle in a Haystack

### 7.2 迭代策略

1. 调优蒸馏损失权重
2. 扩大 CPT token 量
3. 调整裁剪配置
4. 任务-specific SFT/DPO

## 8. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|---|---|---|
| 1.4% 裁剪比信息损失过大 | 基础能力崩坏 | 增加蒸馏数据量、hidden-state 对齐、CPT 用原分布数据 |
| 参数估算不准 | 实际模型偏离 4B | 用 `prune_deepseek_v4_4b_config.py` 精确估算，以 storage 为准 |
| PP=8 forward hang | 无法 forward validation | 依赖启发式 NAS，修复后启用 validation |
| MoE 专家负载失衡 | 推理效率下降 | 保留 aux_loss，监控 expert load |
| MTP 退化 | 多 token 预测失效 | CPT 中保留 MTP loss |
| 长上下文能力损失 | 关键优势消失 | 评估 Needle in a Haystack，必要时长文 CPT |

## 9. 资源估算

> 说明：本节估算基于 **FLOPs = 6 × 每 token 激活参数量 × token 数** 的 standard transformer 训练公式，并假设 MoE MFU ≈ 35%、**全部使用 BF16 训练（不启用 FP8）**。蒸馏阶段若 teacher 与 student 同机加载，需额外计入 teacher 前向的 FLOPs；若 teacher offload 到 CPU，则瓶颈转为 CPU 推理吞吐，通常不推荐用于 100B 以上 token 的大规模 KD。

### 9.1 估算公式

- **CPT / SFT**（学生自回归训练）：
  ```
  FLOPs ≈ 6 × P_active × T
  ```
  其中 `P_active` 为每 token 激活参数量（目标 ~1B），`T` 为训练 token 数。

- **KD（本地加载 teacher）**：
  ```
  FLOPs ≈ 6 × P_student_active × T + 2 × P_teacher_active × T
  ```
  DeepSeek-V4-Flash 每 token 激活约 37B，因此本地 teacher 的前向开销约为学生训练本身的 12 倍。

- **KD（teacher 服务化）**：
  ```
  FLOPs ≈ 6 × P_student_active × T
  ```
  本地只算学生，但受 teacher 推理吞吐制约。

- **GPU 有效算力**（BF16，假设 MFU = 35%）：
  | GPU | BF16 峰值 | 有效算力 |
  |---|---|---|
  | H100 SXM | 989 TFLOPS | 346 TFLOPS |
  | H20 | 148 TFLOPS | 52 TFLOPS |

- **时间公式**：
  ```
  time(seconds) = FLOPs / (N × F_effective)
  ```
  `N` 为 GPU 数量，`F_effective` 为单卡有效算力。

### 9.2 H100/H200 方案（参考基线）

| 阶段 | GPU 需求 | 估算时间 | 备注 |
|---|---|---|---|
| 裁剪 | 8× H100/H200 80GB | 1–2 天 | PP=8，TP=1 |
| 蒸馏 | 8–32× H100/H200 | 2–4 周 | 本地 teacher（BF16）约 2–4 周；服务化 teacher 约 4–8 天 |
| CPT | 32–128× H100/H200 | 6–20 天 | 300B tokens / 64 GPUs / BF16 约 6 天 |
| SFT | 8–16× H100/H200 | 1–3 天 | 10B tokens / 8 GPUs / BF16 约 1.7 天 |
| DPO | 8–16× H100/H200 | 1–2 天 | |
| 量化/部署 | 4× H100 | 1 天 | |

**总估算**: 32–128 张 H100/H200，**3–7 周**完整周期。

### 9.3 H20 方案（当前可行方案）

H20 单卡显存 **96GB**（高于 H100 80GB），但 BF16/FP8 Tensor Core 有效算力约为 H100 的 **15% ~ 30%**。因此：

- **内存不是瓶颈**：284B 原始模型在 PP=8 下每卡约 35–50GB，H20 96GB 绰绰有余，可考虑 TP=2/TP=4 缓解 rank 7 瓶颈。
- **算力是瓶颈**：训练阶段耗时约为 H100 的 **3.3 ~ 6.7 倍**。

| 阶段 | GPU 需求 | H20 估算时间 | 计算依据 |
|---|---|---|---|
| 裁剪 | 8× H20 96GB | 1–3 天 | 内存充裕，可尝试 TP=2/4 |
| 蒸馏 | 8–32× H20 | **3–16 周** | 本地 BF16 teacher / 32 GPUs：100B tokens ≈ 8 周；**单机 FP8 teacher / 8 GPUs：100B tokens ≈ 12–16 周，50B tokens ≈ 6–9 周**；服务化 teacher ≈ 6–14 天 |
| CPT | 32–128× H20 | **4–8 周** | 300B tokens / 64 GPUs / BF16 ≈ 13 天；500B / 32 GPUs ≈ 7 周 |
| SFT | 8–16× H20 | **2–4 天** | 10B tokens / 8 GPUs / BF16 ≈ 3.3 天 |
| DPO | 8–16× H20 | **1–3 天** | |
| 量化/部署 | 4× H20 | 1–2 天 | BF16 部署，暂不启用 FP8 |

**总估算**: 32–128 张 H20，**10–24 周**完整周期（若蒸馏用服务化 teacher，可缩短至 **6–12 周**；若只有单机 8×H20 且 teacher 用 FP8，蒸馏 100B tokens 约 12–16 周，总周期约 14–20 周；蒸馏 50B tokens 约 6–9 周，总周期约 8–13 周）。

### 9.4 关于 teacher offload 到 CPU 的重要说明

若将 284B 的 DeepSeek-V4-Flash teacher **完全 offload 到 CPU 计算**，则 KD 阶段的瓶颈不再是 GPU FLOPs，而是 **CPU 推理吞吐**。粗略估算：

- 284B 模型 BF16 CPU 前向：高端服务器（64 核 + AVX-512）有效算力约 0.1–0.5 TFLOPS。
- 每 token 前向 FLOPs ≈ 2 × 37B = 74B。
- 单服务器吞吐 ≈ 0.1–0.5 TFLOPS / 74B ≈ **1–7 tokens/秒**。
- 100B tokens / 1 token/s ≈ **3170 年**；即使 1000 台服务器并行也超过 1 年。

**结论**：CPU 上直接跑 284B teacher 做 100B–300B token 的大规模 KD **不可行**。

若必须 CPU offload，只有两个可行路径：
1. **大幅缩减 KD 数据量**：例如只做 1B–10B token 的轻量蒸馏，耗时约 1–10 周（取决于 CPU 规模），但效果可能明显弱于完整 KD。
2. **对 teacher 做极致量化 + CPU 优化**：INT4/INT8 + llama.cpp 类推理框架，吞吐可提升 10–100 倍，但 100B token 仍需要数月到数年。

因此，**即使显存紧张，仍建议 teacher 放在 GPU 上**（本地加载或服务化），而不是 CPU offload。

### 9.5 关键修正说明

之前粗略按“H100 的 3–4 倍”给出的 12–24 周偏保守，但未充分考虑：
1. 目标模型每 token 仅激活 ~1B 参数，FLOPs 基数很小；
2. 不启用 FP8 后，H20 BF16 有效算力仅为 H100 的 15%；
3. teacher 本地加载时，284B teacher 前向是 KD 的主要开销；
4. teacher CPU offload 会彻底改变 KD 瓶颈，通常不推荐。

因此按 BF16 + 本地/服务化 teacher 重算后：
- **H20 + 本地 GPU teacher + BF16**：约 **10–24 周**
- **H20 + 服务化 teacher + BF16**：约 **6–12 周**

### 9.6 H20 优化建议（在「不启用 FP8、teacher 不上 CPU」约束下）

1. **裁剪阶段尝试 TP>1**
   - H20 显存充足，可将 `tensor_model_parallel_size` 从 1 提高到 2 或 4，缓解 PP=8 下 rank 7 的 lm_head + MTP 串行瓶颈。
   - 需在 `run_v4_4b_pipeline.sh` 中同步修改 `provider_overrides` 里的 `tensor_model_parallel_size`。

2. **蒸馏阶段 teacher 必须放在 GPU 上，且已支持服务化白盒蒸馏 / FP8 量化 teacher**
   - 由于已确认 teacher 不上 CPU，建议：
     - **本地 GPU 加载**：8×H20 节点同时加载 284B teacher + 4B student，显存仍足够（ teacher 约 568GB / 8 = 71GB/卡，student 很小）。
     - **FP8 量化 teacher + 单机 8×H20**：将 284B teacher 量化为 FP8（约 284GB），与 BF16 学生（约 8GB）一起加载到单机 8×H20（共 768GB 显存），无需额外 teacher 节点。已提供 `prepare_fp8_teacher.sh` 和 `distill_deepseek_v4_4b.py --teacher_quant_format fp8`。
     - **服务化 teacher**（更推荐）：单独 8×H20 节点部署 teacher，蒸馏节点只加载 student，可重叠 teacher/student 计算。已提供 `serve_teacher_deepseek_v4.py` 服务端和 `distill_deepseek_v4_4b.py --teacher_endpoint` 客户端，支持返回 logits、hidden states 和 MoE router logits，实现白盒蒸馏。

3. **CPT/SFT 使用 BF16 + 较大 batch size**
   - 不启用 FP8 后，用 H20 的 96GB 显存放更大 micro-batch，提高 MFU。
   - 建议 `per_device_train_batch_size=8` 起步，配合 gradient accumulation 达到全局 4M–8M tokens。

4. **蒸馏数据量与 teacher 部署联动选择**
   - 若只能本地 GPU teacher：KD 100B tokens 在 32×H20 BF16 下约 8 周，建议优先考虑 **50B tokens** 以控制时间。
   - 若可服务化 teacher：可放心做到 **100B–300B tokens**。

## 10. 新增/修改文件清单

| 文件 | 说明 |
|---|---|
| `examples/megatron_bridge/prune_deepseek_v4_4b_config.py` | 精确 4B 参数配置生成 |
| `examples/megatron_bridge/run_v4_4b_pipeline.sh` | 端到端裁剪流水线 |
| `examples/llm_distill/distill_deepseek_v4_4b.py` | MoE-aware 知识蒸馏（支持本地/远程/FP8 teacher） |
| `examples/llm_distill/serve_teacher_deepseek_v4.py` | 服务化白盒 teacher 服务端 |
| `examples/llm_distill/prepare_fp8_teacher.sh` | 教师模型 FP8 量化脚本 |
| `examples/megatron_bridge/cpt_deepseek_v4_4b.py` | 继续预训练 |
| `examples/megatron_bridge/sft_deepseek_v4_4b.py` | 指令微调 |
| `examples/megatron_bridge/eval_deepseek_v4_4b.py` | 基准评估 |
| `examples/megatron_bridge/configs/cpt_data_mix.json` | CPT 数据混合示例 |
| `examples/megatron_bridge/configs/sft_data_mix.json` | SFT 数据混合示例 |

## 11. 执行检查清单

- [ ] 运行 `run_v4_4b_pipeline.sh` 生成 4B 裁剪模型
- [ ] 验证 MLA/MoE/mHC/MTP 结构完整
- [ ] 运行 `distill_deepseek_v4_4b.py`
- [ ] 运行 `cpt_deepseek_v4_4b.py`
- [ ] 运行 `sft_deepseek_v4_4b.py`
- [ ] 运行 `eval_deepseek_v4_4b.py` 验证超越 30B 基线
- [ ] 运行 `examples/llm_ptq/hf_ptq.py` 量化并部署

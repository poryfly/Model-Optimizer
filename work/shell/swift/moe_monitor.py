"""MoE 训练过程监控 Callback - 适配 Megatron + ms-swift + DeepSeek-V4

功能：
  ✅ 监控 Router Entropy（路由熵）
  ✅ 监控 Expert Utilization（专家利用率 + 绝对数量）
  ✅ 监控 Top-K Gating Distribution（Top-1/Top-2 概率）
  ✅ 监控 Per-Expert Token 分布偏斜度
  ✅ 监控 Router Z-Loss（需启用 moe_z_loss_coeff）

依赖（需修改 Megatron 源码）：
  1. router.py forward() 末尾添加：self._last_router_logits = logits.detach()
  2. router.py apply_z_loss() 末尾添加：self._last_z_loss = z_loss.detach().item()

注意：
  - DeepSeek-V4 使用 sqrtsoftplus 门控，不是 softmax
  - 当前实现适配 EP=1 / TP=1，多 rank 场景需额外聚合
  - 需先启用 --moe_z_loss_coeff 0.001 才能采集 z_loss

使用方法：
  bash train_cpt_full_deepseek_v4_with_monitor.sh  # 已默认启用
"""
import torch
import numpy as np
from typing import Dict, List, Optional

# 导入 MegatronCallback 基类
from .base import MegatronCallback


class MoEMonitorCallback(MegatronCallback):
    """
    MoE 训练过程监控 Callback。
    在 on_log 阶段从模型中提取 MoE 路由相关的统计指标。
    适配 DeepSeek-V4 的 MoE 架构（Top-K=6 路由, sqrtsoftplus 门控）。
    """

    def __init__(self, trainer):
        super().__init__(trainer)
        # 配置参数
        self.log_every = 1
        self.top_k = 6  # DeepSeek-V4 的 top_k

    def _is_write_rank(self):
        try:
            import torch.distributed as dist
            return dist.get_rank() == 0
        except:
            return True

    def _collect_router_stats(self, model, global_step=0) -> Dict[str, float]:
        """
        遍历所有 MoE 层，收集路由统计指标

        关键说明：
        - DeepSeek-V4 使用 sqrtsoftplus 门控，logits 经过 routing() 后会转为 scores
        - 采集的 _last_router_logits 是 sigmoid/softmax 前的原始 logits
        - 这里计算指标时直接使用 scores（不是再套一层 softmax）
        """
        stats = {
            'moe/router_entropy': [],
            'moe/expert_utilization': [],
            'moe/num_active_experts': [],  # 绝对数量版
            'moe/top1_prob_mean': [],
            'moe/top1_prob_std': [],
            'moe/top2_prob_mean': [],
            'moe/per_expert_token_std': [],
            'moe/router_z_loss': [],
        }

        # 遍历模型的所有模块
        for name, module in model.named_modules():
            # 检测是否是 MoE 层
            is_moe = False
            router_logits = None

            # 模式1: 直接从 router 模块获取 _last_router_logits
            # 我们在 Megatron 的 TopKRouter 中添加了：self._last_router_logits = logits.detach()
            if hasattr(module, '_last_router_logits'):
                router_logits = module._last_router_logits
                is_moe = True

            # 模式2: module 直接存储了 router output
            if hasattr(module, '_router_logits'):
                router_logits = module._router_logits
                is_moe = True

            # 模式3: 检测 MoE 相关类名
            if 'MoE' in module.__class__.__name__ or 'Router' in module.__class__.__name__:
                if hasattr(module, 'router_logits'):
                    router_logits = module.router_logits
                    is_moe = True

            if not is_moe or router_logits is None:
                continue

            # ===== 从 router_logits 计算指标 =====
            try:
                # router_logits shape: [num_tokens, num_experts]
                if isinstance(router_logits, torch.Tensor):
                    raw_logits = router_logits.float()
                elif isinstance(router_logits, (list, tuple)):
                    raw_logits = router_logits[0].float()
                else:
                    continue

                # 确保是 2D: [num_tokens, num_experts]
                if raw_logits.dim() == 1:
                    raw_logits = raw_logits.unsqueeze(0)

                num_experts = raw_logits.shape[-1]

                # 跳过非标准 MoE 层（如 shared expert, num_experts=1）
                if num_experts <= 1:
                    continue

                # 实际使用的 top_k 不能超过 num_experts
                effective_top_k = min(self.top_k, num_experts)

                # DeepSeek-V4 使用 sqrtsoftplus 门控：score = sqrt(softplus(logit))
                # 这里用 softplus 归一化后作为 "概率"，避免门控错误
                # softplus(x) = log(1 + exp(x))，这里用 sigmoid 近似作为更稳的实现
                scores = torch.sigmoid(raw_logits)
                # 归一化到 [0,1] 范围当作概率使用
                probs = scores / (scores.sum(dim=-1, keepdim=True) + 1e-10)

                # 1. Router Entropy: H = -sum(p * log(p))
                log_probs = torch.log(probs + 1e-10)
                entropy = -torch.sum(probs * log_probs, dim=-1)
                max_entropy = np.log(num_experts)
                normalized_entropy = (entropy.mean().item()) / max_entropy
                stats['moe/router_entropy'].append(normalized_entropy)

                # 2. Expert Utilization: 基于 Top-K 路由
                # 统计被有效使用的专家（负载占比 > 均匀分布的 20%），避免单 batch 跳动
                _, topk_indices = probs.topk(effective_top_k, dim=-1)  # [tokens, top_k]
                all_selected = topk_indices.flatten()
                tokens_per_expert = torch.bincount(
                    all_selected.long(), minlength=num_experts
                ).float()
                # 绝对数量版
                unique_experts = torch.unique(all_selected)
                stats['moe/num_active_experts'].append(float(len(unique_experts)))
                # 归一化版（被有效使用的专家比例）
                frac = tokens_per_expert / (tokens_per_expert.sum() + 1e-10)
                dead_threshold = (1.0 / num_experts) * 0.2  # 均匀分布的 20%
                utilization_effective = (frac > dead_threshold).float().mean().item()
                stats['moe/expert_utilization'].append(utilization_effective)

                # 3. Top-1 / Top-2 概率统计（基于 scores，不是 raw_logits）
                sorted_scores, _ = probs.sort(dim=-1, descending=True)
                top1_probs = sorted_scores[..., 0]
                stats['moe/top1_prob_mean'].append(top1_probs.mean().item())
                stats['moe/top1_prob_std'].append(top1_probs.std().item())
                # Top-2 概率：检测是否退化为 single expert
                if num_experts >= 2:
                    top2_probs = sorted_scores[..., 1]
                    stats['moe/top2_prob_mean'].append(top2_probs.mean().item())

                # 4. Per-Expert Token 分布偏斜度
                # 用变异系数衡量 token 分配的不均衡程度
                token_mean = tokens_per_expert.mean()
                token_std = tokens_per_expert.std()
                if token_mean > 0:
                    cv = (token_std / token_mean).item()
                    stats['moe/per_expert_token_std'].append(cv)

                # 6. Router Z-Loss (DeepSeek 专用)
                # 我们在 Megatron 的 TopKRouter 中添加了：self._last_z_loss = z_loss.detach().item()
                if hasattr(module, '_last_z_loss'):
                    if module._last_z_loss is not None:
                        stats['moe/router_z_loss'].append(float(module._last_z_loss))
                    else:
                        # 调试：第一次打印
                        if global_step == 1 and self._is_write_rank():
                            print(f"[MoEMonitor] {name} 有 _last_z_loss 属性但值为 None")
                else:
                    # 调试：第一次打印
                    if global_step == 1 and self._is_write_rank():
                        print(f"[MoEMonitor] {name} 没有 _last_z_loss 属性")

            except Exception as e:
                # 单个模块计算失败不影响整体
                # 调试：第一次打印错误信息
                if global_step == 1 and self._is_write_rank():
                    print(f"[MoEMonitor] 模块 {name} 计算失败: {e}")
                continue

        # 汇总：对各层取均值
        result = {}
        for key, values in stats.items():
            if values:
                result[key] = float(np.mean(values))
                result[f'{key}/max'] = float(np.max(values))
                result[f'{key}/min'] = float(np.min(values))

        return result

    def on_log(self, logs):
        """
        在日志记录时附加 MoE 指标

        Args:
            logs: 日志字典 (会被写入 logging.jsonl 和 TensorBoard)
        """
        # 使用 trainer.state.iteration 作为全局步数（所有 rank 一致）
        global_step = self.state.iteration if hasattr(self.state, 'iteration') else 0

        if logs is None:
            logs = {}

        # 采集策略：第 1 步 + 每隔 log_every 步
        should_collect = (global_step == 1) or (global_step % self.log_every == 0)
        if not should_collect:
            return

        # 获取模型
        model = None
        if hasattr(self.trainer, 'wrapped_models') and self.trainer.wrapped_models:
            model = self.trainer.wrapped_models[0]
        elif hasattr(self.trainer, 'model'):
            model = self.trainer.model

        if model is None:
            return

        try:
            moe_stats = self._collect_router_stats(model, global_step)
            logs.update(moe_stats)
        except Exception as e:
            # 每 100 步打印一次错误，避免刷屏
            if self._is_write_rank() and global_step % 100 == 0:
                print(f"[MoEMonitor] 采集失败（不影响训练）: {e}")

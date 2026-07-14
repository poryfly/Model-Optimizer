"""MoE 训练过程监控 Callback - 适配 Megatron-Bridge + DeepSeek-V4

功能：
  ✅ 监控训练基础指标（iteration, consumed_samples, loss, grad_norm, lr, z_loss）
  ✅ 监控 z_loss (与 Megatron 原版 total_loss_dict["z_loss"] 完全一致, 从
     get_moe_metrics_tracker().report() 读, 不依赖源码 patch)
  ✅ 监控 Router Entropy (需 Megatron 源码 patch 提供 _last_router_logits)
  ✅ 监控 Expert Utilization (需 Megatron 源码 patch 提供 _last_routing_map)
  ✅ 监控 Top-K Gating Distribution (需 Megatron 源码 patch)

依赖 (需修改 Megatron 源码获得 router 指标, 供 _collect_router_stats 使用):
  megatron/core/transformer/moe/router.py  TopKRouter.forward() 需包含以下 3 行:
      self._last_z_loss = z_loss_mean.detach()      # apply_z_loss() 中保存, 用于 router_z_loss
      self._last_router_logits = logits.detach()   # gating 输出, 用于 router_entropy / top-k
      self._last_routing_map = routing_map.detach() # top-k 路由结果, 用于 expert_utilization

注意：
  - 不修改 Megatron 源码也能用基础指标 (loss, grad_norm, learning_rate, z_loss 等),
    因为它们都从 framework 的 metrics_tracker / optimizer / loss_dict 读取
  - DeepSeek-V4 使用 sqrtsoftplus 门控, 这里按相同公式计算 scores
  - "z_loss" 字段与原版一致: 直接调用 get_moe_metrics_tracker().report()
    (与原版 training_log 调 track_moe_metrics 完全相同), 写入 callback 自己的
    total_loss_dict, 取出后除以 advanced_iters, 与 Megatron 原版 total_loss_dict["z_loss"]
    / advanced_iters 的值完全一致
  - "router_z_loss" 字段是原始 z_loss_mean (每层均值, 未除以 num_microbatches),
    仅用于观察, 与原版 z_loss 数值不同

使用方法：
  export MOE_MONITOR_ENABLED=true
  bash run_pretrain_pruned.sh
"""
from __future__ import annotations

import os
import time
import torch
import numpy as np
from typing import Dict

from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.transformer.moe.moe_logging import get_moe_metrics_tracker
from megatron.bridge.training.callbacks import Callback, CallbackContext


class MoEMonitorCallback(Callback):
    """MoE 训练过程监控 Callback（适配 Megatron-Bridge 的 on_train_step_end 事件）

    输出与 Megatron 原本的 iteration log 同步：
      - 频率：使用 Megatron 的 logger.log_interval（默认 5）
      - 位置：在 Megatron 原本 log 行**正下方**一行（通过缩进对齐）
      - 格式：不重复 iteration 头， 不使用 [MoEMonitor] 前缀，
              直接接 Megatron 那一行的延续位置, 看起来就是同一行的延伸
    """

    def __init__(self, top_k: int = 6):
        """
        Args:
            top_k: DeepSeek-V4 的 top_k 路由数（默认 6）
        """
        super().__init__()
        self.top_k = top_k
        self._cached_log_interval: int | None = None
        self._cached_train_iters: int | None = None
        # 用于复现 Megatron 原版 training_log 的几个指标
        # (total_loss_dict 是 train.py 局部变量, 未暴露给 callback, 只能自计)
        self._skipped_count: int = 0
        self._nan_count: int = 0
        self._last_log_time: float | None = None
        self._last_logged_iter: int = 0
        # z_loss 与原版一致: 原版 z_loss = total_loss_dict["z_loss"] / advanced_iters
        # (advanced_iters = log 间隔内未被跳过的 iter 数)
        self._advanced_iter_count: int = 0
        # 对比模式: 当 Bridge 内置 training_log 也开启时, callback 只打印 MoE 路由指标,
        # 避免 loss/lr/grad_norm/z_loss 等基础指标两边数值不一致。
        self._compare_mode = os.environ.get("MOE_MONITOR_COMPARE_MODE", "false").lower() in (
            "1", "true", "yes", "on"
        )

    # ---------- helpers ----------

    def _is_write_rank(self) -> bool:
        try:
            import torch.distributed as dist
            return dist.get_rank() == 0
        except Exception:
            return True

    @staticmethod
    def _to_float(v) -> float:
        if v is None:
            return float("nan")
        if isinstance(v, torch.Tensor):
            return float(v.detach().item())
        return float(v)

    @staticmethod
    def _is_nan_or_inf(v) -> bool:
        """检查 loss 值是否为 NaN / +Inf / -Inf (参考 Megatron 原版 got_nan 逻辑)"""
        try:
            if isinstance(v, torch.Tensor):
                val = float(v.detach().item())
            else:
                val = float(v)
        except Exception:
            return False
        return val != val or val == float("inf") or val == -float("inf")

    def _loss_dict_has_nan(self, loss_dict) -> bool:
        """检查 loss_dict 里是否存在 NaN/Inf (与 train_utils.py 733-743 的 got_nan 逻辑一致)"""
        if not loss_dict:
            return False
        for v in loss_dict.values():
            if self._is_nan_or_inf(v):
                return True
        return False

    def _compute_elapsed_time_ms(self, iteration: int, context: CallbackContext) -> float:
        """计算 elapsed time per iteration (ms)

        参考 Megatron 原版 train_utils.py:1073-1074:
            elapsed_time = timers("interval-time").elapsed(barrier=True)
            elapsed_time_per_iteration = elapsed_time / total_iterations

        但 timers("interval-time") 是 Megatron 内部状态, 且 elapsed() 会重置计时器,
        改用 time.time() 自己记录上次 log 时刻, 精度足够 (仅用于日志显示)。
        """
        current_time = time.time()
        # 首次 log 时, 以 GlobalState.start_time 为基准, 与 Megatron 原版 (从训练开始计时) 保持一致
        if self._last_log_time is None:
            try:
                self._last_log_time = float(context.state.start_time)
            except Exception:
                self._last_log_time = current_time
        iter_count = iteration - self._last_logged_iter
        if iter_count > 0:
            elapsed_ms = (current_time - self._last_log_time) * 1000.0 / iter_count
        else:
            elapsed_ms = 0.0
        self._last_log_time = current_time
        self._last_logged_iter = iteration
        return elapsed_ms

    def _write_to_tensorboard(
        self,
        writer,
        iteration: int,
        consumed_samples: int,
        elapsed_time_per_iteration_ms: float,
        learning_rate: float,
        batch_size: int,
        lm_loss: float,
        grad_norm: float,
        loss_scale: float,
        temp_loss_dict: Dict,
        aux_load_balancing_loss: float,
        aux_seq_load_balancing_loss: float,
        aux_global_load_balancing_loss: float,
        z_loss: float,
        moe_stats: Dict[str, float],
    ) -> None:
        """把基础指标 + aux_loss + z_loss + MoE 路由指标写入 TensorBoard

        仅在 ``writer`` 非 None 的 rank (在 Bridge 框架中仅 last rank 会创建 SummaryWriter) 上写入,
        其他 rank 静默跳过。

        字段名与 Megatron 原版 ``training_log`` (train_utils.py:833-937) 保持一致,
        以确保 TensorBoard 上看到的指标顺序/名称与 Megatron 原生训练完全相同。

        关键点:
          - skip_train_metrics_log=True 时, 框架的 training_log 不会调,
            所有 add_scalar 都在本 callback 中完成, 等价于“接管”了 TensorBoard 写入。
          - 同时补上原版没有的 MoE 路由指标 (moe/router_entropy 等) 与 aux_losses,
            这些在原版中不输出到 TB, 但用户需要。
        """
        if writer is None:
            return
        try:
            # ---- 基础训练指标 (与 train_utils.py:891-925 一致) ----
            writer.add_scalar("lm loss", lm_loss, iteration)
            writer.add_scalar("lm loss vs samples", lm_loss, consumed_samples)
            writer.add_scalar("learning-rate", learning_rate, iteration)
            writer.add_scalar("learning-rate vs samples", learning_rate, consumed_samples)
            writer.add_scalar("batch-size", batch_size, iteration)
            writer.add_scalar("batch-size vs samples", batch_size, consumed_samples)
            writer.add_scalar("loss-scale", loss_scale, iteration)
            writer.add_scalar("grad-norm", grad_norm, iteration)
            writer.add_scalar(
                "elapsed time per iteration (ms)",
                elapsed_time_per_iteration_ms,
                iteration,
            )
            # ---- aux_losses (从 temp_loss_dict 提取, 与原版 total_loss_dict 字段名一致) ----
            # 原版 track_moe_metrics 会把这些 key 写到 total_loss_dict,
            # 本 callback 同源调用 get_moe_metrics_tracker().report(), 字段名相同。
            if "load_balancing_loss" in temp_loss_dict:
                writer.add_scalar("load_balancing_loss", aux_load_balancing_loss, iteration)
            if "seq_load_balancing_loss" in temp_loss_dict:
                writer.add_scalar("seq_load_balancing_loss", aux_seq_load_balancing_loss, iteration)
            if "global_load_balancing_loss" in temp_loss_dict:
                writer.add_scalar(
                    "global_load_balancing_loss",
                    aux_global_load_balancing_loss,
                    iteration,
                )
            if "z_loss" in temp_loss_dict:
                writer.add_scalar("z_loss", z_loss, iteration)
            # ---- MoE 路由指标 (原版 training_log 不输出, callback 补上) ----
            for key, value in moe_stats.items():
                # key 例: "moe/router_entropy", "moe/router_entropy/max", "moe/router_z_loss"
                # TensorBoard 会把 "/" 当 namespace 划分
                try:
                    writer.add_scalar(key, float(value), iteration)
                except Exception:
                    pass
            writer.flush()
        except Exception as e:
            # 写 TB 失败不能影响训练
            if self._is_write_rank():
                print(f"[moe_monitor_callback] TensorBoard write warning: {e}")

    def _collect_router_stats(self, model) -> Dict[str, float]:
        """遍历 MoE 层，收集路由统计指标

        需要 Megatron 源码在 router 模块设置以下属性:
          - _last_router_logits: gating 输出 (forward() 中保存)
          - _last_routing_map:   top-k 路由结果 (forward() 中保存)
          - _last_z_loss:        z_loss_mean 标量 (apply_z_loss() 中保存)

        如果没有, 对应 moe/* 指标会为空 dict（不报错）。
        """
        stats: Dict[str, list] = {
            "moe/router_entropy": [],
            "moe/expert_utilization": [],
            "moe/num_active_experts": [],
            "moe/top1_prob_mean": [],
            "moe/top1_prob_std": [],
            "moe/top2_prob_mean": [],
            "moe/per_expert_token_std": [],
            "moe/router_z_loss": [],
        }

        found_router_attr = False
        found_routing_map = False
        found_z_loss = False

        for name, module in model.named_modules():
            router_logits = getattr(module, "_last_router_logits", None)
            routing_map = getattr(module, "_last_routing_map", None)
            z_loss = getattr(module, "_last_z_loss", None)

            # 收集 z_loss (来自 apply_z_loss, 每个 MoE 层都有)
            if z_loss is not None:
                try:
                    if isinstance(z_loss, torch.Tensor):
                        stats["moe/router_z_loss"].append(float(z_loss.detach().item()))
                    else:
                        stats["moe/router_z_loss"].append(float(z_loss))
                    found_z_loss = True
                except Exception:
                    pass

            if router_logits is None and routing_map is None:
                continue
            if router_logits is not None:
                found_router_attr = True
            if routing_map is not None:
                found_routing_map = True

            try:
                num_experts = None
                if router_logits is not None:
                    # 统一成 [num_tokens, num_experts] 形状
                    if isinstance(router_logits, torch.Tensor):
                        raw = router_logits.detach().float()
                    elif isinstance(router_logits, (list, tuple)):
                        raw = router_logits[0].detach().float()
                    else:
                        raw = None
                    if raw is not None and raw.dim() == 1:
                        raw = raw.unsqueeze(0)
                    if raw is not None:
                        num_experts = raw.shape[-1]
                if routing_map is not None:
                    rm = routing_map.detach()
                    if rm.dim() == 1:
                        rm = rm.unsqueeze(0)
                    if num_experts is None:
                        num_experts = rm.shape[-1]
                if num_experts is None or num_experts <= 1:
                    continue  # 跳过 shared expert 之类

                effective_top_k = min(self.top_k, num_experts)

                # 1. Router Entropy - DeepSeek-V4 使用 sqrtsoftplus 门控
                if raw is not None:
                    scores = torch.nn.functional.softplus(raw.float()).sqrt()
                    probs = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)
                    log_probs = torch.log(probs + 1e-20)
                    entropy = -torch.sum(probs * log_probs, dim=-1)
                    max_entropy = np.log(num_experts)
                    stats["moe/router_entropy"].append(entropy.mean().item() / max_entropy)
                else:
                    probs = None

                # 2. Expert Utilization - 优先使用 _last_routing_map (更准, 直接取 top-k 选择)
                if routing_map is not None:
                    tokens_per_expert = rm.float().sum(dim=0)
                elif probs is not None:
                    _, topk_indices = probs.topk(effective_top_k, dim=-1)
                    all_selected = topk_indices.flatten()
                    tokens_per_expert = torch.bincount(
                        all_selected.long(), minlength=num_experts
                    ).float()
                else:
                    continue

                unique_experts = (tokens_per_expert > 0).nonzero(as_tuple=True)[0]
                stats["moe/num_active_experts"].append(float(len(unique_experts)))

                frac = tokens_per_expert / (tokens_per_expert.sum() + 1e-20)
                dead_threshold = (1.0 / num_experts) * 0.2
                stats["moe/expert_utilization"].append(
                    (frac > dead_threshold).float().mean().item()
                )

                # 3. Top-1 / Top-2 prob (仅当 raw 可用)
                if probs is not None:
                    sorted_scores, _ = probs.sort(dim=-1, descending=True)
                    top1 = sorted_scores[..., 0]
                    stats["moe/top1_prob_mean"].append(top1.mean().item())
                    stats["moe/top1_prob_std"].append(top1.std().item())
                    if num_experts >= 2:
                        stats["moe/top2_prob_mean"].append(sorted_scores[..., 1].mean().item())

                # 4. Per-Expert Token 分布偏斜度 (CV)
                tm = tokens_per_expert.mean()
                ts = tokens_per_expert.std()
                if tm > 0:
                    stats["moe/per_expert_token_std"].append((ts / tm).item())

            except Exception as e:
                if self._is_write_rank():
                    print(f"[MoEMonitor] {name} 路由统计失败: {e}")
                continue

        # 提示是否需要 patch (只在有缺失时一次性提示)
        if not found_router_attr or not found_routing_map or not found_z_loss:
            if self._is_write_rank():
                missing = []
                if not found_router_attr:
                    missing.append("_last_router_logits (moe/router_entropy / top1_prob_mean 等)")
                if not found_routing_map:
                    missing.append("_last_routing_map (moe/expert_utilization / per_expert_token_std 等)")
                if not found_z_loss:
                    missing.append("_last_z_loss (moe/router_z_loss)")
                if missing:
                    print(f"[MoEMonitor] 未检测到 router 属性: {', '.join(missing)}; 请 patch Megatron 源码 router.py")

        # 汇总
        result: Dict[str, float] = {}
        for key, values in stats.items():
            if values:
                result[key] = float(np.mean(values))
                result[f"{key}/max"] = float(np.max(values))
                result[f"{key}/min"] = float(np.min(values))
        return result

    # ---------- callback events ----------

    def on_train_start(self, context: CallbackContext) -> None:
        if not self._is_write_rank():
            return
        if self._compare_mode:
            print("\n[MoEMonitor] 已启用 - 对比模式: Bridge 内置 training_log 输出基础指标, "
                  "本 callback 只补充 MoE 路由指标 (router_entropy / expert_utilization / ...)\n")
        else:
            print("\n[MoEMonitor] 已启用 - 唯一一行输出, 完全仿照 Megatron 原本 log 格式, "
                  "包含全部基础 + 新增指标 (用 | 分隔)\n")

    def on_train_step_end(self, context: CallbackContext) -> None:
        """完全仿照 Megatron 原本 training_log 的 log_string 格式, 打印一行。

        关键设计:
          - 与 Megatron 原本的 iteration log 同步频率 (使用 logger.log_interval)
          - 包含 Megatron 原本的所有基础指标 (consumed samples, lr, batch size,
            lm loss, z_loss, loss scale, grad norm) + 额外 MoE 路由指标
          - 所有指标用 | 分隔, 格式与 Megatron 原本 log_string 完全一致
          - 为了只显示一行, 在 run_pretrain_pruned.sh 中设置 logger.skip_train_metrics_log=true
          - z_loss 来源: 调用 get_moe_metrics_tracker().report() (与原版
            training_log 调 track_moe_metrics 完全相同), 这样 z_loss 与原版一致
        """
        iteration = context.state.train_state.step
        if iteration is None or iteration <= 0:
            return

        # 1) 首次进入时从 Megatron config 拿到 log_interval 与 train_iters
        if self._cached_log_interval is None:
            try:
                self._cached_log_interval = int(
                    context.state.cfg.logger.log_interval
                )
            except Exception:
                self._cached_log_interval = 5  # 兜底
            try:
                self._cached_train_iters = int(
                    context.state.cfg.train.train_iters
                )
            except Exception:
                self._cached_train_iters = 0

        # 2) 累计 skipped / nan 计数 (不论本次是否要 log 都需更新, 与 Megatron 原版 total_loss_dict 逻辑一致)
        if context.skipped_iter:
            self._skipped_count += 1
        else:
            # 统计 log 间隔内未被跳过的 iter 数 (与原版 advanced_iters 一致)
            self._advanced_iter_count += 1
        loss_dict = context.loss_dict or {}
        if self._loss_dict_has_nan(loss_dict):
            self._nan_count += 1

        # 3) 频率: 与 Megatron logger.log_interval 保持一致
        if iteration % self._cached_log_interval != 0:
            return

        # 3.5) 对比模式 (与 Bridge 内置 training_log 同时开启):
        #      官方 training_log 会输出所有基础指标 + 官方 MoE 指标 (z_loss / aux_loss 等),
        #      callback 只补充 callback 独有的 MoE 路由指标, 避免两边 loss / lr / grad_norm
        #      / z_loss 因计算方式不同而不一致。
        if self._compare_mode:
            # 注意：Bridge 框架的 tensorboard_logger 通常在 last rank 上创建，
            # 而 stdout log 只在 rank 0 打印。因此 TB 写入和 log 打印要分开处理。
            writer = getattr(context.state, "tensorboard_logger", None)
            if writer is not None:
                moe_stats_tb: Dict[str, float] = {}
                if context.model and len(context.model) > 0:
                    moe_stats_tb = self._collect_router_stats(context.model[0])
                if moe_stats_tb:
                    for key, value in moe_stats_tb.items():
                        try:
                            writer.add_scalar(key, float(value), iteration)
                        except Exception:
                            pass
                    writer.flush()

            if self._is_write_rank():
                moe_stats: Dict[str, float] = {}
                if context.model and len(context.model) > 0:
                    moe_stats = self._collect_router_stats(context.model[0])
                if moe_stats:
                    from datetime import datetime
                    log_string = f" [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
                    log_string += f" iteration {iteration:8d}/{self._cached_train_iters:8d} |"
                    log_string += " MoE |"
                    for key in (
                        "moe/router_entropy",
                        "moe/expert_utilization",
                        "moe/num_active_experts",
                        "moe/top1_prob_mean",
                        "moe/top1_prob_std",
                        "moe/top2_prob_mean",
                        "moe/per_expert_token_std",
                        "moe/router_z_loss",
                    ):
                        if key in moe_stats:
                            short = key.split("/")[-1]
                            log_string += f" {short}: {moe_stats[key]:.4f} |"
                            max_key = f"{key}/max"
                            min_key = f"{key}/min"
                            if max_key in moe_stats and min_key in moe_stats:
                                log_string += f" {short}/max: {moe_stats[max_key]:.4f} |"
                                log_string += f" {short}/min: {moe_stats[min_key]:.4f} |"
                    print(log_string)

            self._advanced_iter_count = 0
            self._skipped_count = 0
            self._nan_count = 0
            return

        # 4) 调用 metrics_tracker.report() 拿到与原版相同的 z_loss 与 aux loss
        #    report() 内部会 all_reduce + aggregate, 必须在所有 rank 上调用,
        #    否则 PP 同步会 hang. 所以必须放在 write_rank 检查**之前**.
        #    report() 会把 ends with "loss" 的字段累加到 total_loss_dict.
        #    track_names 列表与原版 train_utils.py:1031-1041 保持一致.
        temp_loss_dict: Dict[str, torch.Tensor] = {}
        cfg_model = context.state.cfg.model
        moe_router_load_balancing_type = getattr(
            cfg_model, "moe_router_load_balancing_type", ""
        )
        if isinstance(moe_router_load_balancing_type, list):
            routing_type_str = " ".join(moe_router_load_balancing_type)
        else:
            routing_type_str = str(moe_router_load_balancing_type)
        track_names: list = []
        if "aux_loss" in routing_type_str:
            track_names.append("load_balancing_loss")
        if "seq_aux_loss" in routing_type_str:
            track_names.append("seq_load_balancing_loss")
        if "global_aux_loss" in routing_type_str:
            track_names.append("global_load_balancing_loss")
        if getattr(cfg_model, "moe_z_loss_coeff", None) is not None:
            track_names.append("z_loss")
        if track_names:
            # 与原版 train_utils.py:1043-1046 保持一致的 num_layers 计算
            if getattr(cfg_model, "is_hybrid_model", False):
                num_layers = getattr(cfg_model, "hybrid_layer_pattern", "").count("E")
            else:
                num_layers = getattr(cfg_model, "num_layers", None)
            get_moe_metrics_tracker().report(
                loss_scale=1.0 / get_num_microbatches(),
                iteration=iteration,
                writer=None,
                wandb_writer=None,
                total_loss_dict=temp_loss_dict,
                per_layer_logging=False,
                force_initialize=True,
                track_names=track_names,
                num_layers=num_layers,
                moe_layer_freq=getattr(cfg_model, "moe_layer_freq", None),
                mtp_num_layers=getattr(cfg_model, "mtp_num_layers", None),
                pg_collection=None,  # 用默认 parallel_state
            )

        # 5) 采集 MoE 路由指标 (router entropy, expert utilization, top-k 等)
        moe_stats: Dict[str, float] = {}
        if context.model and len(context.model) > 0:
            moe_stats = self._collect_router_stats(context.model[0])

        # 6) 从 loss_dict 提取 lm_loss
        lm_loss_t = loss_dict.get("lm loss", None)
        if isinstance(lm_loss_t, torch.Tensor):
            lm_loss_t = lm_loss_t.item()
        lm_loss = float(lm_loss_t) if lm_loss_t is not None else 0.0
        # z_loss 与原版一致: total_loss_dict["z_loss"] / advanced_iters
        # aux_losses (load_balancing_loss / seq_load_balancing_loss / global_load_balancing_loss)
        # 也是 total_loss_dict[...] / advanced_iters
        denom = self._advanced_iter_count if self._advanced_iter_count > 0 else 1

        def _aux_val(name: str) -> float:
            """取 temp_loss_dict 中某 aux loss / z_loss 的标量值, 0.0 表示不存在"""
            v = temp_loss_dict.get(name, None)
            if isinstance(v, torch.Tensor):
                return float(v.detach().item())
            if v is None:
                return 0.0
            return float(v)

        z_loss = _aux_val("z_loss") / denom
        aux_load_balancing_loss = _aux_val("load_balancing_loss") / denom
        aux_seq_load_balancing_loss = _aux_val("seq_load_balancing_loss") / denom
        aux_global_load_balancing_loss = _aux_val("global_load_balancing_loss") / denom

        # 7) grad_norm
        grad_norm = context.grad_norm if context.grad_norm is not None else 0.0

        # 8) learning rate
        learning_rate = 0.0
        if context.optimizer is not None:
            for param_group in context.optimizer.param_groups:
                if len(param_group) == 0:
                    continue
                if not param_group.get("is_decoupled_lr", False):
                    learning_rate = float(param_group["lr"])
                    break

        # 9) loss scale
        loss_scale = 1.0
        if context.optimizer is not None:
            try:
                loss_scale = float(context.optimizer.get_loss_scale().item())
            except Exception:
                pass

        # 10) global_batch_size
        try:
            batch_size = int(context.state.cfg.train.global_batch_size)
        except Exception:
            batch_size = 0

        # 11) consumed_samples
        consumed_samples = context.state.train_state.consumed_train_samples

        # 12) elapsed time per iteration (ms) —— 与 Megatron 原版 train_utils.py:1147 保持一致
        elapsed_time_per_iteration_ms = self._compute_elapsed_time_ms(iteration, context)

        # 13) TensorBoard 写入 (完全对齐原版 training_log + ms-swift 字段)
        #     GlobalState.tensorboard_logger 仅在 last rank 上实例化, 其他 rank 自动跳过。
        #     skip_train_metrics_log=True 时 Megatron 框架不会写 TB, 这里补上,
        #     保证 TensorBoard 上 loss / z_loss / aux_losses / MoE 路由指标全部可见。
        #     有 writer 的 rank (不一定是 rank 0) 负责写。
        writer = getattr(context.state, "tensorboard_logger", None)
        if writer is not None:
            self._write_to_tensorboard(
                writer=writer,
                iteration=iteration,
                consumed_samples=consumed_samples,
                elapsed_time_per_iteration_ms=elapsed_time_per_iteration_ms,
                learning_rate=learning_rate,
                batch_size=batch_size,
                lm_loss=lm_loss,
                grad_norm=grad_norm,
                loss_scale=loss_scale,
                temp_loss_dict=temp_loss_dict,
                aux_load_balancing_loss=aux_load_balancing_loss,
                aux_seq_load_balancing_loss=aux_seq_load_balancing_loss,
                aux_global_load_balancing_loss=aux_global_load_balancing_loss,
                z_loss=z_loss,
                moe_stats=moe_stats,
            )

        # 14) 构造一行 log_string, 完全仿照 Megatron 原本格式
        #     参考 train_utils.py:1142-1194 (Bridge 官方 training_log)
        #     顺序: iteration / consumed samples / elapsed time per iteration (ms) /
        #           learning rate / global batch size / [lm loss] / [aux_losses] / [z_loss] /
        #           loss scale / grad norm / number of skipped iterations / number of nan iterations /
        #           [MoE 路由指标]
        if self._is_write_rank():
            from datetime import datetime
            log_string = f" [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
            log_string += f" iteration {iteration:8d}/{self._cached_train_iters:8d} |"
            log_string += f" consumed samples: {consumed_samples:12d} |"
            log_string += f" elapsed time per iteration (ms): {elapsed_time_per_iteration_ms:.1f} |"
            log_string += f" learning rate: {learning_rate:.6E} |"
            log_string += f" global batch size: {batch_size:5d} |"
            log_string += f" lm loss: {lm_loss:.6E} |"
            # aux_losses 与 z_loss 同位于 lm loss 与 loss scale 之间,
            # 按原版 total_loss_dict 插入顺序 (= track_names 顺序) 打印.
            if "load_balancing_loss" in temp_loss_dict:
                log_string += f" load_balancing_loss: {aux_load_balancing_loss:.6E} |"
            if "seq_load_balancing_loss" in temp_loss_dict:
                log_string += f" seq_load_balancing_loss: {aux_seq_load_balancing_loss:.6E} |"
            if "global_load_balancing_loss" in temp_loss_dict:
                log_string += f" global_load_balancing_loss: {aux_global_load_balancing_loss:.6E} |"
            if "z_loss" in temp_loss_dict:
                log_string += f" z_loss: {z_loss:.6E} |"
            log_string += f" loss scale: {loss_scale:.1f} |"
            log_string += f" grad norm: {grad_norm:.3f} |"
            log_string += f" number of skipped iterations: {self._skipped_count:3d} |"
            log_string += f" number of nan iterations: {self._nan_count:3d} |"
            # 追加 MoE 新增指标, 用 | 分隔, 顺序与 ms-swift 对齐
            for key in (
                "moe/router_entropy",
                "moe/expert_utilization",
                "moe/num_active_experts",
                "moe/top1_prob_mean",
                "moe/top1_prob_std",
                "moe/top2_prob_mean",
                "moe/per_expert_token_std",
                "moe/router_z_loss",
            ):
                if key in moe_stats:
                    short = key.split("/")[-1]
                    log_string += f" {short}: {moe_stats[key]:.4f} |"
                    max_key = f"{key}/max"
                    min_key = f"{key}/min"
                    if max_key in moe_stats and min_key in moe_stats:
                        log_string += f" {short}/max: {moe_stats[max_key]:.4f} |"
                        log_string += f" {short}/min: {moe_stats[min_key]:.4f} |"
            print(log_string)

        # 15) 重置 advanced_iter_count (下次 log 间隔重新计)
        self._advanced_iter_count = 0

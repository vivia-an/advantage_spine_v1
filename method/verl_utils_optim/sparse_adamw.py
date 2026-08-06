"""SparseAdamW: AdamW with adaptive update clipping and delayed sparse masking."""

import copy
import math
from typing import Any, Iterable, Optional

import torch
from torch import Tensor
from torch.optim import Optimizer

from .sparse_adagrad import (
    SPARSE_PHASE_DENSE,
    SPARSE_PHASE_GRAD_ADAPTIVE,
    SPARSE_PHASE_KL_ADAPTIVE,
    SPARSE_PHASE_RAMP,
    SPARSE_PHASE_STEADY,
    SPARSE_PHASE_UNIFIED,
    _diag_flags,
    sparse_schedule_phase_index,
)

# 选择规则白名单：
#   topk             原始：取 |u| top-ρ
#   threshold        原始：|u| >= mean(|u|) * τ
#   topk_rescale     毛笔"集锋"：topk + 每张量 L2 重缩放回 update 的 L2 → 不需要 lr×N
#   precond_topk     按 |exp_avg|（≈ 动量幅值；等价 |u|·√v）选 top-ρ
#   topk_ef          EF21：topk(u + e_{t-1})，残差 e_t = (u+e_{t-1}) - masked
#   topk_rescale_ema 集锋+运笔：EMA_β(|u|) 评分 + L2 rescale（B1：能量+时间一致性）
#   topk_ef_ema      留白+运笔：EMA_β(|u + e_{t-1}|) 评分 + EF 残差（B2：EF + 时间一致性）
_SUPPORTED_SPARSE_MODES = {
    "topk", "threshold", "topk_rescale", "precond_topk", "topk_ef",
    "topk_rescale_ema", "topk_ef_ema",
}


class SparseAdamW(Optimizer):
    """AdamW + adaptive update clipping + delayed sparse masking."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        adaptive_update_clip_factor: float = 0.0,
        adaptive_update_clip_beta: float = 0.95,
        sparse_start_step: int = 0,
        sparse_mode: str = "threshold",
        sparse_threshold_scale: float = 0.0,
        sparse_topk_ratio: float = 1.0,
        sparse_ramp_steps: int = 0,
        sparse_min_numel: int = 4096,
        decoupled_weight_decay: bool = True,
        sparse_ema_beta: float = 0.9,
        soar_enabled: bool = False,        # SOAR: 用 mask 结晶度/能量在线联合调 (ρ, η); recipe 是其静态特例
        soar_c0: float = 0.10,             # 门控开启的结晶度阈 (persistence 起点≈0.096)
        soar_c1: float = 0.30,             # 门控全开的结晶度 (persistence 饱和≈0.309)
        soar_e_floor: float = 0.85,        # 能量保留地板; 低于则不收紧 ρ (recipe≈0.88)
        soar_A_max: float = 12.6,          # 超额放大上限 (≈20/√0.4, 使全门控@d0 复刻 recipe ×20)
        soar_d0: float = 0.40,             # 基线 keep-ratio (= recipe topk0.4)
        soar_d_min: float = 0.20,          # 结晶最强时的最省 keep-ratio
        soar_mult_max: float = 25.0,       # η 倍率硬上限 (防发散)
        soar_budget: float = 0.0,          # HighLRRisk 预算 B; >0 才启用风险封顶, 0=关(用 gate+硬上限)
        soar_damping_beta: float = 0.0,    # SOAR 阻尼: 对 d_t/lr_mult 做 EMA 平滑(治 e_floor 二元翻转引起的振荡); 0=关
        dynamic_switch: bool = False,      # C1 动态 switch: warmup 期测 mask churn, 检出结晶自动 commit 稀疏(替代硬编码 sparse_start)
        dynamic_switch_every: int = 10,    # 每几步测一次 would-be mask churn (对齐离线 10 步分辨率)
        dynamic_switch_max: int = 120,     # 安全帽: 到此步仍未检出则强制 commit
        dynamic_rho_frac: float = 0.5,     # mask_plateau_fire 峰值减半比
        dynamic_flat_eps: float = 0.20,    # mask_plateau_fire 平台判据
        dynamic_K: int = 2,                # 连续 K 步 below&flat 触发
        dynamic_lr_mult: float = 1.0,      # commit 后 lr 放大倍率 (paperB: switch+lr×20 同时 commit; warmup 用基 lr)
        dynamic_churn_on: str = "update",  # churn 测在 'update'(原始)还是 'momentum'(|exp_avg|, 更平滑早结晶)
    ):
        if lr <= 0:
            raise ValueError(f"Invalid lr: {lr}")
        if eps <= 0:
            raise ValueError(f"Invalid eps: {eps}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid betas: {betas}")
        if adaptive_update_clip_factor < 0:
            raise ValueError("adaptive_update_clip_factor must be >= 0")
        if not 0.0 <= adaptive_update_clip_beta < 1.0:
            raise ValueError("adaptive_update_clip_beta must be in [0, 1)")
        if sparse_mode not in _SUPPORTED_SPARSE_MODES:
            raise ValueError(f"Unsupported sparse_mode: {sparse_mode}")
        if not 0.0 < sparse_topk_ratio <= 1.0:
            raise ValueError("sparse_topk_ratio must be in (0, 1]")
        if sparse_threshold_scale < 0:
            raise ValueError("sparse_threshold_scale must be >= 0")
        if sparse_start_step < 0 or sparse_ramp_steps < 0:
            raise ValueError("sparse_start_step and sparse_ramp_steps must be >= 0")

        defaults = dict(
            lr=lr,
            betas=tuple(betas),
            eps=eps,
            weight_decay=weight_decay,
            adaptive_update_clip_factor=adaptive_update_clip_factor,
            adaptive_update_clip_beta=adaptive_update_clip_beta,
            sparse_start_step=sparse_start_step,
            sparse_mode=sparse_mode,
            sparse_threshold_scale=sparse_threshold_scale,
            sparse_topk_ratio=sparse_topk_ratio,
            sparse_ramp_steps=sparse_ramp_steps,
            sparse_min_numel=sparse_min_numel,
            decoupled_weight_decay=decoupled_weight_decay,
            sparse_ema_beta=sparse_ema_beta,
        )
        super().__init__(params, defaults)
        self._global_step = 0
        self._update_rms_ema: Optional[float] = None
        self._last_metrics: dict[str, float] = {}
        self._diag_mask_cache_enabled: bool = False
        self._last_diag_masks_by_param_id: dict[int, Tensor] = {}
        self._last_diag_masked_updates_by_param_id: dict[int, Tensor] = {}
        self._last_diag_raw_updates_by_param_id: dict[int, Tensor] = {}
        # ---- SOAR (Subnetwork-Overlap-Aware Rescaling) ----
        self._soar_enabled = bool(soar_enabled)
        self._soar_c0 = float(soar_c0); self._soar_c1 = float(soar_c1)
        self._soar_e_floor = float(soar_e_floor); self._soar_A_max = float(soar_A_max)
        self._soar_d0 = float(soar_d0); self._soar_d_min = float(soar_d_min)
        self._soar_mult_max = float(soar_mult_max); self._soar_budget = float(soar_budget)
        # 上一步在线信号 (one-step lag 控制器)
        self._soar_c_prev = 0.0      # mask_persistence_ema (结晶度)
        self._soar_e_prev = 1.0      # mask_energy_kept (命中主方向)
        self._soar_u_rms_prev = 0.0  # update rms (风险预算用)
        # 本步控制量 (step 起算)
        self._soar_d_now = float(soar_d0)
        self._soar_lr_mult_now = 1.0
        self._soar_A_now = 1.0; self._soar_gate_now = 0.0; self._soar_risk_now = 0.0
        self._soar_damping_beta = float(soar_damping_beta)
        self._soar_d_prev = float(soar_d0)        # 阻尼 EMA 状态
        self._soar_mult_prev = 1.0
        # ---- C1 动态 switch ----
        self._dynamic_switch = bool(dynamic_switch)
        self._dynamic_switch_every = int(dynamic_switch_every)
        self._dynamic_switch_max = int(dynamic_switch_max)
        self._dynamic_rho_frac = float(dynamic_rho_frac)
        self._dynamic_flat_eps = float(dynamic_flat_eps)
        self._dynamic_K = int(dynamic_K)
        self._dynamic_lr_mult = float(dynamic_lr_mult)
        self._dynamic_churn_on = str(dynamic_churn_on)
        self._dyn_churn_seq: list = []        # warmup 期 would-be mask churn 序列 (喂 mask_plateau_fire)
        self._dyn_fired_step: Optional[int] = None  # 检出结晶并 commit 稀疏的步; None=尚在 warmup 观察
        # SPARSE_DIAG=energy|jaccard|persistence|all|off （能量诊断默认开；jaccard/persistence 需显式打开）
        self._diag = _diag_flags()

    def state_dict(self) -> dict[str, Any]:
        d = super().state_dict()
        d["_sparse_extra"] = {
            "global_step": int(self._global_step),
            "update_rms_ema": None if self._update_rms_ema is None else float(self._update_rms_ema),
        }
        return d

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        sd = copy.deepcopy(state_dict)
        extra = sd.pop("_sparse_extra", None)
        template_groups = copy.deepcopy(self.param_groups)
        super().load_state_dict(sd)
        for g, tmpl in zip(self.param_groups, template_groups):
            for k, v in tmpl.items():
                if k != "params":
                    g[k] = v
        if extra is not None:
            self._global_step = int(extra.get("global_step", 0))
            ema = extra.get("update_rms_ema")
            self._update_rms_ema = None if ema is None else float(ema)
        else:
            self._global_step = 0
            self._update_rms_ema = None

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._global_step += 1
        if self._soar_enabled and self.param_groups and \
                self._global_step >= self.param_groups[0]["sparse_start_step"]:
            self._compute_soar_controls()   # 用上一步 c/e 信号定本步 (ρ, η)
        else:
            self._soar_lr_mult_now = 1.0
        if self._diag_mask_cache_enabled:
            self._last_diag_masks_by_param_id.clear()
            self._last_diag_masked_updates_by_param_id.clear()
            self._last_diag_raw_updates_by_param_id.clear()
        total_sq = 0.0
        total_elems = 0

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            decoupled = group["decoupled_weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("SparseAdamW expects dense gradients.")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = torch.zeros((), dtype=torch.float32, device=p.device)
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                eff_grad = grad
                if wd != 0 and not decoupled:
                    eff_grad = grad.add(p, alpha=wd)

                state["step"] += 1
                step_t = float(state["step"].item())

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(eff_grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(eff_grad, eff_grad, value=1.0 - beta2)

                bc1 = 1.0 - beta1 ** step_t
                bc2 = 1.0 - beta2 ** step_t
                denom = (exp_avg_sq / bc2).sqrt().add_(eps)
                update = (exp_avg / bc1) / denom
                state["_update_for_step"] = update

                total_sq += update.float().pow(2).sum().item()
                total_elems += update.numel()

        update_rms_before = math.sqrt(total_sq / max(total_elems, 1)) if total_elems > 0 else 0.0
        clip_scale = self._compute_clip_scale(update_rms_before)
        update_rms_after = update_rms_before * clip_scale

        nonzero_elems = 0
        considered_elems = 0
        sparse_target = 1.0
        if self.param_groups:
            sparse_target = self._current_sparse_target(self.param_groups[0])

        # 诊断累加器（仅在被 mask 张量上累计；energy_kept 含义对 rescale 模式恒为 1）
        diag = self._diag
        diag_any = diag["energy"] or diag["jaccard"] or diag["persistence"]
        upd_sq_sum = 0.0
        kept_sq_sum = 0.0
        masked_elems_sparse = 0
        mask_inter_sum = 0
        mask_union_sum = 0
        persist_sum = 0.0
        persist_count = 0
        self._dyn_inter_acc = 0     # C1 动态 switch: 本步 would-be mask 全局交/并 (算 churn)
        self._dyn_union_acc = 0

        for group in self.param_groups:
            lr = group["lr"]
            if self._soar_enabled and self._global_step >= group["sparse_start_step"]:
                lr = lr * self._soar_lr_mult_now
            if self._dynamic_switch and self._dyn_fired_step is not None and self._global_step >= self._dyn_fired_step:
                lr = lr * self._dynamic_lr_mult   # commit 后才放大 (warmup 用基 lr)   # η_t = α₀ · (1/√d_t) · A_t
            wd = group["weight_decay"]
            decoupled = group["decoupled_weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                update = state.pop("_update_for_step")
                update.mul_(clip_scale)

                masked_update = update
                sparse_mask_used: Optional[Tensor] = None
                was_masked = False
                if self._global_step >= self._effective_sparse_start(group) and p.numel() >= group["sparse_min_numel"]:
                    masked_update, sparse_mask_used = self._apply_sparse_mask_with_mask(update, group, state)
                    was_masked = sparse_mask_used is not None
                # C1 动态 switch: warmup 期(尚未 fire)测 would-be mask churn
                if (self._dynamic_switch and self._dyn_fired_step is None
                        and self._global_step % self._dynamic_switch_every == 0
                        and p.numel() >= group["sparse_min_numel"]):
                    _cscore = state["exp_avg"].abs() if self._dynamic_churn_on == "momentum" else update.abs()
                    _wb = self._topk_mask_from_score(_cscore, group["sparse_topk_ratio"], update.shape)
                    _prev = state.get("dyn_prev_mask")
                    if _prev is not None and _prev.shape == _wb.shape:
                        self._dyn_inter_acc += int((_wb & _prev).sum().item())
                        self._dyn_union_acc += int((_wb | _prev).sum().item())
                    state["dyn_prev_mask"] = _wb.clone()

                if self._diag_mask_cache_enabled:
                    pid = id(p)
                    raw_update = update.detach().float().clone()
                    if p.numel() < group["sparse_min_numel"]:
                        diag_m = torch.ones_like(update, dtype=torch.bool)
                    elif sparse_mask_used is None:
                        diag_m = torch.ones_like(update, dtype=torch.bool)
                    else:
                        diag_m = sparse_mask_used.detach()
                    self._last_diag_masks_by_param_id[pid] = diag_m.clone()
                    self._last_diag_masked_updates_by_param_id[pid] = masked_update.detach().float().clone()
                    self._last_diag_raw_updates_by_param_id[pid] = raw_update

                if wd != 0 and decoupled:
                    p.mul_(1.0 - lr * wd)
                p.add_(masked_update, alpha=-lr)

                considered_elems += masked_update.numel()
                nonzero_elems += masked_update.count_nonzero().item()

                # 诊断指标：仅在 was_masked 张量上聚合
                if was_masked and diag_any:
                    if diag["energy"]:
                        upd_sq_sum += update.float().pow(2).sum().item()
                        kept_sq_sum += masked_update.float().pow(2).sum().item()
                        masked_elems_sparse += masked_update.numel()
                    if diag["jaccard"] or diag["persistence"]:
                        curr_mask = sparse_mask_used if sparse_mask_used is not None else (masked_update != 0)
                        if diag["jaccard"]:
                            prev = state.get("prev_mask")
                            if prev is not None and prev.shape == curr_mask.shape:
                                mask_inter_sum += (curr_mask & prev).sum().item()
                                mask_union_sum += (curr_mask | prev).sum().item()
                            state["prev_mask"] = curr_mask.clone()
                        if diag["persistence"]:
                            persist = state.get("persist_count")
                            if persist is None or persist.shape != curr_mask.shape:
                                persist = torch.zeros_like(curr_mask, dtype=torch.int16)
                            persist.add_(curr_mask.to(torch.int16))
                            persist.mul_(curr_mask.to(torch.int16))
                            state["persist_count"] = persist
                            persist_sum += persist.float().sum().item()
                            persist_count += curr_mask.numel()

        # C1 动态 switch: 本步若做了 churn 测量, 计 churn → 喂检测器 → 触发则 commit 稀疏
        if (self._dynamic_switch and self._dyn_fired_step is None
                and self._global_step % self._dynamic_switch_every == 0):
            if self._dyn_union_acc > 0:
                _J = self._dyn_inter_acc / self._dyn_union_acc
                self._dyn_churn_seq.append(2.0 * (1.0 - _J) / (1.0 + _J))   # EB 归一化 Hamming
            from verl.utils.optim.sparse_adagrad import mask_plateau_fire
            _fi = mask_plateau_fire(self._dyn_churn_seq, self._dynamic_rho_frac,
                                    self._dynamic_flat_eps, self._dynamic_K)
            if _fi is not None:
                self._dyn_fired_step = self._global_step          # 检出结晶, 从本步 commit 稀疏
            elif self._global_step >= self._dynamic_switch_max:
                self._dyn_fired_step = self._global_step          # 安全帽: 强制 commit

        phase_idx = 0
        if self.param_groups:
            g0 = self.param_groups[0]
            phase_idx = sparse_schedule_phase_index(
                self._global_step,
                int(self._effective_sparse_start(g0)),
                int(g0["sparse_ramp_steps"]),
            )

        self._last_metrics = {
            "update_rms_before_clip": float(update_rms_before),
            "update_rms_after_clip": float(update_rms_after),
            "update_clip_scale": float(clip_scale),
            "update_clip_triggered": float(clip_scale < 0.999999),
            "update_nonzero_ratio": float(nonzero_elems / max(considered_elems, 1)),
            "update_masked_ratio": 1.0 - float(nonzero_elems / max(considered_elems, 1)),
            "update_sparse_target": float(sparse_target),
            "sparse_phase": float(phase_idx),
        }
        if diag["energy"] and masked_elems_sparse > 0 and upd_sq_sum > 0.0:
            self._last_metrics["mask_energy_kept"] = float(kept_sq_sum / upd_sq_sum)
        if diag["jaccard"] and mask_union_sum > 0:
            self._last_metrics["mask_jaccard_prev"] = float(mask_inter_sum / mask_union_sum)
        if diag["persistence"] and persist_count > 0:
            self._last_metrics["mask_persistence_ema"] = float(persist_sum / persist_count)
        if self._soar_enabled:
            self._last_metrics["soar_d_t"] = float(self._soar_d_now)
            self._last_metrics["soar_lr_mult"] = float(self._soar_lr_mult_now)
            self._last_metrics["soar_A_t"] = float(self._soar_A_now)
            self._last_metrics["soar_gate"] = float(self._soar_gate_now)
            self._last_metrics["soar_eta"] = float(self.param_groups[0]["lr"] * self._soar_lr_mult_now)
            self._last_metrics["soar_highlr_risk"] = float(self._soar_risk_now)
            self._last_metrics["soar_c_signal"] = float(self._soar_c_prev)   # 本步用的(上步)信号
            self._last_metrics["soar_e_signal"] = float(self._soar_e_prev)
            # 回写信号供下一步 (one-step lag): 结晶度/能量需 persistence+energy diag 开启
            self._soar_c_prev = float(self._last_metrics.get("mask_persistence_ema", self._soar_c_prev))
            self._soar_e_prev = float(self._last_metrics.get("mask_energy_kept", self._soar_e_prev))
            self._soar_u_rms_prev = float(update_rms_after)
        if self._dynamic_switch:
            self._last_metrics["dyn_switch_enabled"] = 1.0
            self._last_metrics["dyn_fired"] = 1.0 if self._dyn_fired_step is not None else 0.0
            self._last_metrics["dyn_fired_step"] = float(self._dyn_fired_step or 0)
            self._last_metrics["dyn_eff_sparse_start"] = float(self._effective_sparse_start(self.param_groups[0]))
            if self._dyn_churn_seq:
                self._last_metrics["dyn_churn_last"] = float(self._dyn_churn_seq[-1])
                self._last_metrics["dyn_churn_peak"] = float(max(self._dyn_churn_seq))
        return loss

    def get_last_metrics(self) -> dict[str, float]:
        return dict(self._last_metrics)

    def set_diagnostic_mask_cache_enabled(self, enabled: bool) -> None:
        self._diag_mask_cache_enabled = bool(enabled)
        if not self._diag_mask_cache_enabled:
            self._last_diag_masks_by_param_id.clear()
            self._last_diag_masked_updates_by_param_id.clear()
            self._last_diag_raw_updates_by_param_id.clear()

    def get_last_diag_mask_cache(self) -> tuple[dict[int, Tensor], dict[int, Tensor], dict[int, Tensor]]:
        """For post-step replay: shard-local keep masks plus raw/masked pre-lr updates."""
        masks = dict(self._last_diag_masks_by_param_id)
        upd = dict(self._last_diag_masked_updates_by_param_id)
        raw = dict(self._last_diag_raw_updates_by_param_id)
        return masks, upd, raw

    def _compute_clip_scale(self, update_rms_before: float) -> float:
        if update_rms_before <= 0:
            return 1.0
        group = self.param_groups[0]
        factor = group["adaptive_update_clip_factor"]
        beta = group["adaptive_update_clip_beta"]
        if factor <= 0.0:
            if self._update_rms_ema is None:
                self._update_rms_ema = update_rms_before
            else:
                self._update_rms_ema = beta * self._update_rms_ema + (1.0 - beta) * update_rms_before
            return 1.0
        if self._update_rms_ema is None:
            self._update_rms_ema = update_rms_before
            return 1.0
        threshold = max(self._update_rms_ema * factor, 1e-12)
        clip_scale = min(1.0, threshold / update_rms_before)
        self._update_rms_ema = beta * self._update_rms_ema + (1.0 - beta) * update_rms_before
        return clip_scale

    def _compute_soar_controls(self) -> None:
        """SOAR (Subnetwork-Overlap-Aware Rescaling): 用上一步 mask 结晶度 c(persistence)与能量保留 e
        (energy_kept)作为'子网络可信度'在线代理, 在 SMET 密度补偿(1/√d)基线上做门控超额放大 + 自适应收紧
        稀疏度。直觉: mask 还在漂移(c低)→退回密度补偿不敢炸; 结晶且命中高→才推向 ×20; 丢弃能量大→预算封顶。
        recipe(ρ=0.4,×20,静态)是本控制律的一个特例。信号全部已埋点, 无需 dense baseline。"""
        c = max(0.0, min(1.0, self._soar_c_prev))   # 结晶度
        e = max(0.0, min(1.0, self._soar_e_prev))    # 命中主方向
        d0, dmin = self._soar_d0, self._soar_d_min
        d_t = d0 - (d0 - dmin) * c                    # 结晶越强越省
        if e < self._soar_e_floor:                   # 能量保留低于地板→不收紧, 退回 d0
            d_t = d0
        d_t = max(dmin, min(d0, d_t))
        if self._soar_damping_beta > 0.0:                            # 阻尼: EMA 平滑 d_t (治二元翻转)
            d_t = self._soar_damping_beta * self._soar_d_prev + (1.0 - self._soar_damping_beta) * d_t
        self._soar_d_prev = d_t
        self._soar_d_now = d_t
        base_scale = 1.0 / math.sqrt(max(d_t, 1e-6))                 # SMET 1/√d 密度补偿
        gate = max(0.0, min(1.0, (c - self._soar_c0) / max(self._soar_c1 - self._soar_c0, 1e-6))) * e
        A = 1.0 + (self._soar_A_max - 1.0) * gate                    # 门控超额放大
        r = max(0.0, 1.0 - e)                                       # 丢弃能量比 = MaskError
        alpha0 = self.param_groups[0]["lr"]
        if self._soar_budget > 0.0:                                 # HighLRRisk ≤ 预算 B 封顶
            eta_base = alpha0 * base_scale
            risk_unit = (eta_base ** 2) * r * (self._soar_u_rms_prev ** 2) + 1e-12
            A = min(A, max(1.0, math.sqrt(self._soar_budget / risk_unit)))
        mult = min(base_scale * A, self._soar_mult_max)             # 硬上限防发散
        if self._soar_damping_beta > 0.0:                            # 阻尼: EMA 平滑最终倍率
            mult = self._soar_damping_beta * self._soar_mult_prev + (1.0 - self._soar_damping_beta) * mult
        self._soar_mult_prev = mult
        self._soar_lr_mult_now = mult
        self._soar_A_now = A
        self._soar_gate_now = gate
        eta = alpha0 * mult
        self._soar_risk_now = (eta ** 2) * r * (self._soar_u_rms_prev ** 2)

    def _effective_sparse_start(self, group: dict) -> int:
        """C1 动态 switch: 启用时, 稀疏起步 = 检出结晶的步(_dyn_fired_step); 未检出前=∞(保持 dense)。
        未启用则用硬编码 sparse_start_step。"""
        if not self._dynamic_switch:
            return int(group["sparse_start_step"])
        return int(self._dyn_fired_step) if self._dyn_fired_step is not None else 10 ** 9

    def _current_sparse_target(self, group: dict) -> float:
        if self._global_step < self._effective_sparse_start(group):
            return 1.0
        if self._soar_enabled:                 # SOAR: 自适应 keep-ratio (覆盖静态 topk_ratio)
            return self._soar_d_now
        # topk 家族共享 ramp 计划
        if group["sparse_mode"] in {
            "topk", "topk_rescale", "precond_topk", "topk_ef",
            "topk_rescale_ema", "topk_ef_ema",
        }:
            final_ratio = group["sparse_topk_ratio"]
            if group["sparse_ramp_steps"] <= 0:
                return final_ratio
            progress = min(
                1.0,
                float(self._global_step - group["sparse_start_step"]) / float(group["sparse_ramp_steps"]),
            )
            return 1.0 - progress * (1.0 - final_ratio)
        if group["sparse_threshold_scale"] <= 0.0:
            return 1.0
        if group["sparse_ramp_steps"] <= 0:
            return 0.0
        progress = min(
            1.0,
            float(self._global_step - group["sparse_start_step"]) / float(group["sparse_ramp_steps"]),
        )
        return 1.0 - progress

    @staticmethod
    def _topk_mask_from_score(score_abs: Tensor, keep_ratio: float, shape) -> Tensor:
        flat = score_abs.reshape(-1)
        k = max(1, int(flat.numel() * keep_ratio))
        _, indices = torch.topk(flat, k=k, sorted=False)
        mask_flat = torch.zeros_like(flat, dtype=torch.bool)
        mask_flat.scatter_(0, indices, True)
        return mask_flat.reshape(shape)

    def _apply_sparse_mask_with_mask(
        self, update: Tensor, group: dict, state: Optional[dict] = None
    ) -> tuple[Tensor, Optional[Tensor]]:
        """Return masked update and dense bool mask same shape as update (None iff no masking).

        sparse_mode 支持：
          - topk           按 |u| top-ρ
          - threshold      |u| >= mean(|u|)·τ
          - topk_rescale   topk + 每张量 L2 重缩放回 update 的 L2（毛笔集锋；自动等效 lr 放大 1/√energy_kept）
          - precond_topk   按 |exp_avg| top-ρ（≈ Fisher-加权动量幅值）
          - topk_ef        EF21：topk(u + e_{t-1})；残差 e_t = (u+e_{t-1}) − masked
        """
        mode = group["sparse_mode"]

        if mode == "topk":
            keep_ratio = self._current_sparse_target(group)
            if keep_ratio >= 1.0:
                return update, None
            mask = self._topk_mask_from_score(update.abs(), keep_ratio, update.shape)
            return update * mask, mask

        if mode == "topk_rescale":
            keep_ratio = self._current_sparse_target(group)
            if keep_ratio >= 1.0:
                return update, None
            mask = self._topk_mask_from_score(update.abs(), keep_ratio, update.shape)
            masked = update * mask
            # 能量守恒（毛笔集锋）：把 masked 的 L2 拉回 update 的 L2 → 保留张量 step 等效幅值
            u_l2 = update.float().pow(2).sum().sqrt()
            m_l2 = masked.float().pow(2).sum().sqrt()
            if m_l2.item() > 0:
                masked = masked * (u_l2 / m_l2)
            return masked, mask

        if mode == "precond_topk":
            keep_ratio = self._current_sparse_target(group)
            if keep_ratio >= 1.0:
                return update, None
            # 用 |exp_avg|（≈ momentum 幅值，等价 |u|·√v）作为评分
            score = update.abs()
            if state is not None and "exp_avg" in state:
                score = state["exp_avg"].abs()
            mask = self._topk_mask_from_score(score, keep_ratio, update.shape)
            return update * mask, mask

        if mode == "topk_ef":
            keep_ratio = self._current_sparse_target(group)
            if keep_ratio >= 1.0:
                return update, None
            assert state is not None, "topk_ef requires per-param state"
            if "ef_residual" not in state or state["ef_residual"].shape != update.shape:
                state["ef_residual"] = torch.zeros_like(update)
            ef = state["ef_residual"]
            score_tensor = update + ef
            mask = self._topk_mask_from_score(score_tensor.abs(), keep_ratio, update.shape)
            masked = score_tensor * mask
            # 残差 = 总信号 − 已应用部分（被丢的能量进下一步）
            state["ef_residual"] = score_tensor - masked
            return masked, mask

        if mode == "topk_rescale_ema":
            # B1：集锋（rescale）+ 运笔（EMA(|u|) 评分）
            keep_ratio = self._current_sparse_target(group)
            if keep_ratio >= 1.0:
                return update, None
            assert state is not None, "topk_rescale_ema requires per-param state"
            beta = float(group.get("sparse_ema_beta", 0.9))
            if "ema_abs" not in state or state["ema_abs"].shape != update.shape:
                state["ema_abs"] = update.abs().clone()
            ema = state["ema_abs"]
            ema.mul_(beta).add_(update.abs(), alpha=1.0 - beta)
            mask = self._topk_mask_from_score(ema, keep_ratio, update.shape)
            masked = update * mask
            u_l2 = update.float().pow(2).sum().sqrt()
            m_l2 = masked.float().pow(2).sum().sqrt()
            if m_l2.item() > 0:
                masked = masked * (u_l2 / m_l2)
            return masked, mask

        if mode == "topk_ef_ema":
            # B2：留白（EF）+ 运笔（EMA(|score|) 评分）
            keep_ratio = self._current_sparse_target(group)
            if keep_ratio >= 1.0:
                return update, None
            assert state is not None, "topk_ef_ema requires per-param state"
            beta = float(group.get("sparse_ema_beta", 0.9))
            if "ef_residual" not in state or state["ef_residual"].shape != update.shape:
                state["ef_residual"] = torch.zeros_like(update)
            ef = state["ef_residual"]
            score_tensor = update + ef
            if "ema_abs" not in state or state["ema_abs"].shape != update.shape:
                state["ema_abs"] = score_tensor.abs().clone()
            ema = state["ema_abs"]
            ema.mul_(beta).add_(score_tensor.abs(), alpha=1.0 - beta)
            mask = self._topk_mask_from_score(ema, keep_ratio, update.shape)
            masked = score_tensor * mask
            state["ef_residual"] = score_tensor - masked
            return masked, mask

        # threshold 模式
        target_scale = group["sparse_threshold_scale"]
        if target_scale <= 0.0:
            return update, None
        if group["sparse_ramp_steps"] > 0:
            progress = min(
                1.0,
                float(self._global_step - group["sparse_start_step"]) / float(group["sparse_ramp_steps"]),
            )
            target_scale = target_scale * progress
        threshold = update.abs().mean() * target_scale
        mask = update.abs() >= threshold
        return update * mask, mask


def topk_keep_mask_from_tensor(update: Tensor, keep_ratio: float) -> tuple[Tensor, Tensor]:
    """Counterfactual top-k on |update|; returns masked_update bool mask (same dtype as update path)."""
    if keep_ratio >= 1.0:
        m = torch.ones_like(update, dtype=torch.bool)
        return update, m
    flat = update.detach().float().abs().reshape(-1)
    if flat.numel() == 0:
        z = torch.zeros_like(update)
        return z, torch.zeros_like(update, dtype=torch.bool)
    k = max(1, int(flat.numel() * keep_ratio))
    _, indices = torch.topk(flat, k=k, sorted=False)
    mask_flat = torch.zeros_like(flat, dtype=torch.bool)
    mask_flat.scatter_(0, indices, True)
    mask = mask_flat.reshape_as(update)
    uf = update.float()
    return uf * mask, mask


def _adam_pre_lr_update_tensor(param: Tensor, state: dict, group: dict) -> Tensor:
    """Same preconditioned direction as SparseAdamW before sparse mask and clip (caller applies clip externally)."""
    beta1, beta2 = group["betas"]
    eps = group["eps"]
    step_t = float(state["step"].item())
    exp_avg = state["exp_avg"]
    exp_avg_sq = state["exp_avg_sq"]
    bc1 = 1.0 - beta1 ** step_t
    bc2 = 1.0 - beta2 ** step_t
    denom = (exp_avg_sq.float() / bc2).sqrt().add_(eps)
    return ((exp_avg.float() / bc1) / denom)


def counterfactual_topk_masks_from_optimizer(
    optimizer: Optimizer, keep_ratio: float, min_numel: int, apply_clip_scale: float = 1.0
) -> tuple[dict[int, Tensor], dict[int, Tensor], dict[int, Tensor]]:
    """Build masks from Adam-like state **after optimizer.step**. Used for dense AdamW / ablations."""
    masks: dict[int, Tensor] = {}
    masked_upd: dict[int, Tensor] = {}
    raw_upd: dict[int, Tensor] = {}
    @torch.no_grad()
    def _go() -> None:
        for group in optimizer.param_groups:
            if "betas" not in group or "eps" not in group:
                continue
            for p in group["params"]:
                st = optimizer.state.get(p)
                if st is None or "exp_avg" not in st:
                    continue
                if p.numel() < min_numel:
                    continue
                u = _adam_pre_lr_update_tensor(p, st, group)
                u = u * float(apply_clip_scale)
                mu, m = topk_keep_mask_from_tensor(u, keep_ratio=float(keep_ratio))
                masks[id(p)] = m.detach().clone()
                masked_upd[id(p)] = mu.detach().clone()
                raw_upd[id(p)] = u.detach().clone()
    _go()
    return masks, masked_upd, raw_upd


__all__ = [
    "SparseAdamW",
    "topk_keep_mask_from_tensor",
    "counterfactual_topk_masks_from_optimizer",
    "SPARSE_PHASE_DENSE",
    "SPARSE_PHASE_RAMP",
    "SPARSE_PHASE_STEADY",
    "SPARSE_PHASE_KL_ADAPTIVE",
    "SPARSE_PHASE_GRAD_ADAPTIVE",
    "SPARSE_PHASE_UNIFIED",
    "sparse_schedule_phase_index",
]

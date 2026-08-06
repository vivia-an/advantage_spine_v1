import copy
import math
import os
from typing import Any, Iterable, Optional

import torch
from torch import Tensor
from torch.optim import Optimizer

# ns_whiten (neg_mode='ns_whiten') 的 Newton-Schulz 参数, env 可调:
#   默认 promo=5/supp=0 = 标准 Muon 五步正交化(σ→1, 稳健); 设 PASP_NS_PROMO=0 PASP_NS_SUPP=5 切 pion 精确高通.
_PASP_NS_PROMO = int(os.environ.get("PASP_NS_PROMO", "5"))
_PASP_NS_SUPP = int(os.environ.get("PASP_NS_SUPP", "0"))
# v3: pos/neg 通道动量(Muon/pion 式), 白化前对 g_pos/g_neg 各做 EMA 平滑. 0=关(保持瞬时, 现有行为).
_PASP_MOMENTUM = float(os.environ.get("PASP_MOMENTUM", "0.0"))
_PASP_NESTEROV = os.environ.get("PASP_NESTEROV", "1") == "1"
# polar_sparse_whiten (psw, 2026-06-10 用户设计: 符号极性→坐标sparse→白化→rms放大):
_PASP_PSW_TOPK = float(os.environ.get("PASP_PSW_TOPK", "0.1"))    # ② 坐标 top-k 保留比例; >=1 关
_PASP_PSW_GAMMA = float(os.environ.get("PASP_PSW_GAMMA", "1.0"))  # ① 冲突软权指数 λ=min(1,|gp|/|gn|)^γ
_PASP_PSW_FLOOR = float(os.environ.get("PASP_PSW_FLOOR", "0.0"))  # ① 冲突 λ 下限
# PASP 默认使用当前正向梯度的 stable rank，并在每个 optimizer step 重算正向子空间 Q。
# 显式设置 PASP_AUTORANK=0 或 PASP_REFRESH=N>1 仅用于固定秩/缓存消融。
_PASP_AUTORANK = os.environ.get("PASP_AUTORANK", "1") == "1"
_PASP_RMIN = int(os.environ.get("PASP_RMIN", "8"))               # 自适应秩下限
_PASP_RMAX = int(os.environ.get("PASP_RMAX", "64"))              # 自适应秩上限
_PASP_REFRESH_ENV = int(os.environ.get("PASP_REFRESH", "1"))     # 默认1=每步刷新；N>1 仅作缓存消融
# 0626 ratio 消融 (ns_whiten_v2_orthrank): ortho 保留秩 = ratio × pos 秩. 1.0=与 pos 同秩(现状, bit级不变);
#   5.0=留 5 倍 ortho 探索方向, 0.5=留半. clamp 到 [1, min(m,n)]. 仅影响负支 ortho 通道, 单变量.
_PASP_ORTHO_RANK_RATIO = float(os.environ.get("PASP_ORTHO_RANK_RATIO", "1.0"))
# 0626 orthopion (ns_whiten_v2_orthopion): pos子空间走Muon式NS(promo/supp 用上方 _PASP_NS_*=5/0),
#   ortho(负⊥正)走 Pion 式高通(默认promo0/supp5)→ 高通自动留强压弱尾=谱域自适应选秩, 去掉 top-r 固定比例.
#   ⚠️瞬时梯度纯pion suppression 可能塌0, 带防塌; 塌了可 env 切 PASP_ORTHO_PROMO=2 PASP_ORTHO_SUPP=3 混合.
_PASP_ORTHO_PROMO = int(os.environ.get("PASP_ORTHO_PROMO", "0"))
_PASP_ORTHO_SUPP = int(os.environ.get("PASP_ORTHO_SUPP", "5"))


def _stable_rank(M: Tensor, iters: int = 3) -> int:
    """stable rank r_s = ‖M‖_F² / σ_max²  (momentum-free, 无 SVD; σ_max 用 power iteration 估).
    满秩但能量集中→r_s≈有效秩(eff-rank); 低秩→r_s 小. 用作自适应 rangefinder 的目标秩."""
    Mf = M.detach().float()
    fro2 = float((Mf * Mf).sum())
    if fro2 <= 1e-20:
        return 1
    n = Mf.shape[1]
    v = torch.randn(n, 1, device=Mf.device, dtype=Mf.dtype)
    for _ in range(max(1, iters)):
        u = Mf @ v
        un = float(u.norm())
        if un < 1e-20:
            return 1
        u = u / un
        v = Mf.t() @ u
        vn = float(v.norm())
        if vn < 1e-20:
            return 1
        v = v / vn
    sigma_max2 = float((Mf @ v).pow(2).sum())   # ‖M v‖² ≈ σ_max² (v≈右奇异向量)
    if sigma_max2 <= 1e-20:
        return 1
    return int(fro2 / sigma_max2 + 0.5)          # 四舍五入

try:
    from torch.distributed.tensor import DTensor, distribute_tensor
    _HAS_DTENSOR = True
except Exception:  # pragma: no cover
    try:
        from torch.distributed._tensor import DTensor, distribute_tensor
        _HAS_DTENSOR = True
    except Exception:
        DTensor = None
        _HAS_DTENSOR = False


def _remove_neg_subspace_2d(G2d: Tensor, gneg2d: Tensor, r: int) -> Tensor:
    """从合并梯度 G 切除 g_neg 的 top-r 左奇异子空间分量 (二者均为完整 2D)。已单测 (scripts/test_neg_subspace_proj.py)。"""
    m, n = gneg2d.shape
    rq = max(1, min(r, m, n))
    Gf = G2d.float(); gnf = gneg2d.float()
    if float(gnf.abs().max()) < 1e-12:        # g_neg≈0 退化: 不动
        return G2d
    q = min(rq + 4, m, n)
    try:
        U, S, _ = torch.svd_lowrank(gnf, q=q)
    except Exception:
        U, S, _ = torch.linalg.svd(gnf, full_matrices=False)
    Ur = U[:, :rq]                            # [m, rq] 列正交
    Gc = Gf - Ur @ (Ur.T @ Gf)                # 投影到 neg 子空间正交补
    return Gc.to(G2d.dtype)


def _neg_proj_grad(grad: Tensor, g_neg: Tensor, r: int) -> Tensor:
    """对完整或 DTensor 2D 梯度做 neg 谱手术; DTensor 先 full_tensor 再 distribute 回 (仿 muon)。"""
    is_dt = _HAS_DTENSOR and isinstance(grad, DTensor)
    if is_dt:
        mesh, plc = grad.device_mesh, grad.placements
        Gf = grad.full_tensor()
        gnf = g_neg.full_tensor() if isinstance(g_neg, DTensor) else g_neg
    else:
        Gf, gnf = grad, g_neg
    if Gf.dim() != 2:                          # 仅 2D 权重做谱手术
        return grad
    Gc = _remove_neg_subspace_2d(Gf, gnf, r)
    if is_dt:
        return distribute_tensor(Gc, device_mesh=mesh, placements=plc)
    return Gc


def _ns_band_filter(M: Tensor, steps: int) -> Tensor:
    """NS 多项式: 把对称 PSD 矩阵特征值推向 0/1 → 近似主子空间「软投影」(免 SVD, quintic-style, 仿 muon)。
    steps→∞ 退化为精确投影(F²=F); 有限步=可调锐度的软投影。"""
    X = M / (torch.linalg.matrix_norm(M, 2) + 1e-9)
    for _ in range(int(steps)):
        X = 1.5 * X - 0.5 * (X @ X @ X)
    return X


def _soft_neg_filter_2d(G2d: Tensor, gneg2d: Tensor, ns_steps: int, alpha: float):
    """软谱滤波: Gc = G - alpha·F(M)·G, M = 归一化 neg-Gram (gn gnᵀ, 特征值∈[0,1])。
    F(M)≈neg 左奇异主子空间的软投影 → 削掉合并梯度落在 neg 强方向上的分量 (免 SVD)。
    返回 (Gc, removed_frac)。"""
    Gf = G2d.float(); gnf = gneg2d.float()
    if float(gnf.abs().max()) < 1e-12:        # g_neg≈0 退化: 不动
        return G2d, 0.0
    s = gnf.norm() + 1e-7
    M = (gnf @ gnf.t()) / (s * s)             # [m,m] 归一化 neg Gram
    FM = _ns_band_filter(M, ns_steps)
    rm = float(alpha) * (FM @ Gf)
    Gc = Gf - rm
    removed = float(rm.pow(2).sum()) / (float(Gf.pow(2).sum()) + 1e-12)
    return Gc.to(G2d.dtype), removed


def _neg_soft_filter_grad(grad: Tensor, g_neg: Tensor, ns_steps: int, alpha: float):
    """对完整或 DTensor 2D 梯度做 neg 软谱滤波; DTensor 先 full_tensor 再 distribute 回 (仿 _neg_proj_grad)。
    返回 (eff_grad, removed_frac)。"""
    is_dt = _HAS_DTENSOR and isinstance(grad, DTensor)
    if is_dt:
        mesh, plc = grad.device_mesh, grad.placements
        Gf = grad.full_tensor()
        gnf = g_neg.full_tensor() if isinstance(g_neg, DTensor) else g_neg
    else:
        Gf, gnf = grad, g_neg
    if Gf.dim() != 2:                          # 仅 2D 权重
        return grad, 0.0
    Gc, removed = _soft_neg_filter_2d(Gf, gnf, ns_steps, alpha)
    if is_dt:
        return distribute_tensor(Gc, device_mesh=mesh, placements=plc), removed
    return Gc, removed


def _pasp_rangefinder_2d(g_pos2d: Tensor, g_neg2d: Tensor, rank: int, power_iters: int,
                         lambda_align: float, target_ratio: float,
                         lam_min: float, lam_max: float, Q_cached: Optional[Tensor] = None,
                         e_in: Optional[Tensor] = None, ef: bool = False, suppress: bool = True,
                         neg_mode: str = "shrink", neg_topr_coef: float = 1.0):
    """PASP (Polarity-Aware Shrinkage Projection) via randomized range finder。
      G = g_pos + λa·(P_pos g_neg) + λn·((I-P_pos) g_neg)
    重设计(2026-06-04): 子空间不再用 [m,m] Gram + NS 立方 (O(m³), 7900s/步元凶; 还要 vocab OOM 补丁),
    改用随机化 range finder 取 g_pos 的 r 维正交基 Q∈[m,r] (GaLore 式, O(mnr)):
      Y = g_pos·Ω(Ω∈[n,r]); (power iter: Y←g_pos·g_posᵀ·Y); Q,_ = qr(Y)。
      P_pos x = Q(Qᵀ x)。 aligned = Q(Qᵀ g_neg); ortho = g_neg − aligned。
    Q 可由上游周期复用(Q_cached, 中间步不重算 → 进一步摊薄成本; "动态"刷新有真实含义)。
    λn 每步重算并放开区间(去硬编码 0.1 牵引, 可证伪):
      λn = clamp(target_ratio·‖g_pos‖/‖ortho‖, [lam_min,lam_max])。
    vocab/embed/lm_head [151936,n] 在 range finder 下同为 O(mnr), 不再需要 >32768 标量回退。
    返回 (G_new, λn_used, Q)。Q=None 表示走了标量回退(g_pos≈0), 不计入 λn 分布。"""
    gp = g_pos2d.float(); gn = g_neg2d.float()
    m, n = gp.shape
    if neg_mode == "ns_whiten_nopolar":         # 控制臂: 白化合并梯度(g_pos+g_neg), 无极性拆分 = 此 footing 的纯 pion(无动量)
        from verl.utils.optim.pion import high_pass_ns
        Gfull = gp + gn                          # 标准梯度(pion 的输入), 不分极性
        Gw = high_pass_ns(Gfull, promotion_steps=_PASP_NS_PROMO, suppression_steps=_PASP_NS_SUPP)
        if float(Gw.norm()) <= 1e-6 * float(Gfull.norm() + 1e-9): Gw = Gfull   # 防塌回退
        return Gw.to(g_pos2d.dtype), 1.0, None, None    # Q=None: 无子空间; v2 与之只差“极性拆分+λn配权”
    if neg_mode == "polar_sparse_whiten":       # psw (2026-06-10): ①符号极性→②坐标sparse→③白化 (④rms 在 step())
        from verl.utils.optim.pion import high_pass_ns
        # ① 极性(坐标基, 免QR/SVD): 同号全保; 反号软权 λ=min(1,|gp|/|gn|)^γ — neg强压向0, pos强保≈1
        conflict = (gp * gn) < 0
        ratio = (gp.abs() / (gn.abs() + 1e-12)).clamp(max=1.0) ** _PASP_PSW_GAMMA
        lam = torch.where(conflict, ratio.clamp(min=_PASP_PSW_FLOOR), torch.ones_like(gp))
        Gc = gp + lam * gn
        # ② sparse(留白, 坐标基): top-k 截幅度尾 → ③ 只均衡幸存骨架; 与①同基, 等效一次融合 mask
        if 0.0 < _PASP_PSW_TOPK < 1.0:
            k = max(1, int(Gc.numel() * _PASP_PSW_TOPK))
            thr = torch.kthvalue(Gc.abs().flatten().float(), Gc.numel() - k + 1).values
            Gs = Gc * (Gc.abs() >= thr)
        else:
            Gs = Gc
        # ③ 白化(谱基): 截断NS 自带谱高通(σ≥0.3→≈1, 小σ→1.5^t·σ≪1); T_k 剪掉的方向 σ=0 且 f(0)=0 不复活
        Gw = high_pass_ns(Gs, promotion_steps=_PASP_NS_PROMO, suppression_steps=_PASP_NS_SUPP)
        if float(Gw.norm()) <= 1e-6 * float(Gs.norm() + 1e-9): Gw = Gs   # 防塌回退
        return Gw.to(g_pos2d.dtype), float(lam.mean()), None, None   # λ均值借 lam_n 通道观测
    sp = gp.norm()
    if float(sp) < 1e-9:                        # g_pos≈0: 无可信子空间, 标量回退
        return (gp + float(target_ratio) * gn).to(g_pos2d.dtype), float(target_ratio), None, None
    if Q_cached is not None and Q_cached.dim() == 2 and Q_cached.shape[0] == m:
        Q = Q_cached.to(device=gp.device, dtype=gp.dtype)        # 周期复用上次基
    else:
        if _PASP_AUTORANK:                          # B: 每步按 stable rank 自适应秩 (momentum-free)
            r = int(max(_PASP_RMIN, min(_PASP_RMAX, _stable_rank(gp), m, n)))
        else:
            r = int(max(1, min(rank, m, n)))
        Omega = torch.randn(n, r, device=gp.device, dtype=gp.dtype)
        Y = gp @ Omega                                            # [m,r]
        for _ in range(int(power_iters)):                         # power iteration 提升子空间质量
            Y = gp @ (gp.t() @ Y)
        Q, _ = torch.linalg.qr(Y, mode="reduced")                 # [m,r] 正交基, O(m r²)
    aligned = Q @ (Q.t() @ gn)                  # P_pos g_neg (与正信号一致, 少压); O(mnr)
    ortho = gn - aligned                        # (I-P_pos) g_neg (弥散, 强压)
    if ef and e_in is not None:                 # EF21-on-ortho: 把上步收缩丢弃的弥散 neg 累加回来
        ortho = ortho + e_in.to(device=gp.device, dtype=gp.dtype)
    if neg_mode == "orth_lowrank":              # V1: 去 aligned(pos⊥neg) + ortho 只留 top-r 奇异方向(免SVD, 复用 range finder)
        rq = int(Q.shape[1])                    # neg 同 pos 秩
        Om2 = torch.randn(n, rq, device=gp.device, dtype=gp.dtype)
        Y2 = ortho @ Om2
        for _ in range(int(power_iters)):       # power iter 提升 top-r 子空间质量
            Y2 = ortho @ (ortho.t() @ Y2)
        Q2, _ = torch.linalg.qr(Y2, mode="reduced")     # ortho top-r 左奇异基 [m,rq]
        ortho_topr = Q2 @ (Q2.t() @ ortho)              # 投到自身 top-r → 只留最强奇异方向
        G = gp + float(neg_topr_coef) * ortho_topr      # 纯正交 + neg 同秩低秩, 去 aligned
        return G.to(g_pos2d.dtype), float(neg_topr_coef), Q, None
    on = ortho.norm()
    if not suppress:                            # auto-switch: pos 子空间未稳→不抑制, 全保 neg (λn=1=plain dual)
        lam_n = 1.0
    else:
        lam_n = float(target_ratio) * float(sp) / (float(on) + 1e-7)
        lam_n = max(float(lam_min), min(float(lam_max), lam_n))
    if neg_mode == "ns_whiten_v2":              # P2-fix: whiten-THEN-weight, 极性透过白化保留(不被抹平)
        from verl.utils.optim.pion import high_pass_ns
        # 两极性通道各自白化到单位尺度(pos⊥ortho), 再按 λn 配权:
        #   pos 每方向=1, 弥散每方向=λn → 极性差(pos>弥散)留在“白化之后”, 不被均一化吃掉.
        pos_chan = gp + float(lambda_align) * aligned                    # 信号通道(pos 子空间)
        Pw = high_pass_ns(pos_chan, promotion_steps=_PASP_NS_PROMO, suppression_steps=_PASP_NS_SUPP)
        Ow = high_pass_ns(ortho,    promotion_steps=_PASP_NS_PROMO, suppression_steps=_PASP_NS_SUPP)
        if float(Pw.norm()) <= 1e-6 * float(pos_chan.norm() + 1e-9): Pw = pos_chan   # 防塌回退
        if float(Ow.norm()) <= 1e-6 * float(on + 1e-9): Ow = ortho
        G = Pw.float() + lam_n * Ow.float()
        e_out = ((1.0 - lam_n) * ortho).to(g_pos2d.dtype) if ef else None
        return G.to(g_pos2d.dtype), lam_n, Q, e_out
    if neg_mode == "ns_whiten_v2_orthrank":     # 0617用户: 不scalar压ortho到λn, 而是保留ortho top-r(r=pos秩)满权, 只丢ortho尾巴(探索信号假说)
        from verl.utils.optim.pion import high_pass_ns
        # 与 ns_whiten_v2 唯一差别: ortho 通道不再 λn 标量压, 而是先抽自身 top-r 子空间(数量=pos秩), 满权保留, 尾巴丢弃.
        rq_pos = int(Q.shape[1])                                          # pos 秩
        # 0626 ratio 消融: ortho 保留秩 = ratio × pos 秩 (默认1.0=与pos同秩, 现状 bit级不变); clamp[1,min(m,n)]
        rq = max(1, min(int(round(_PASP_ORTHO_RANK_RATIO * rq_pos)), min(gp.shape[0], gp.shape[1])))
        Om2 = torch.randn(n, rq, device=gp.device, dtype=gp.dtype)
        Y2 = ortho @ Om2
        for _ in range(int(power_iters)):                                # power iter 提升 top-r 子空间质量
            Y2 = ortho @ (ortho.t() @ Y2)
        Q2, _ = torch.linalg.qr(Y2, mode="reduced")                      # ortho top-r 左奇异基 [m,rq]
        ortho_topr = Q2 @ (Q2.t() @ ortho)                               # ortho 投到自身 top-r → 保最强 rq 个探索方向
        pos_chan = gp + float(lambda_align) * aligned
        Pw = high_pass_ns(pos_chan,   promotion_steps=_PASP_NS_PROMO, suppression_steps=_PASP_NS_SUPP)
        Ow = high_pass_ns(ortho_topr, promotion_steps=_PASP_NS_PROMO, suppression_steps=_PASP_NS_SUPP)
        if float(Pw.norm()) <= 1e-6 * float(pos_chan.norm() + 1e-9): Pw = pos_chan
        if float(Ow.norm()) <= 1e-6 * float(ortho_topr.norm() + 1e-9): Ow = ortho_topr
        G = Pw.float() + Ow.float()                                      # ortho top-r 满权(无λn压制)
        e_out = (ortho - ortho_topr).to(g_pos2d.dtype) if ef else None   # 残差 = 丢掉的 ortho 尾巴
        return G.to(g_pos2d.dtype), 1.0, Q, e_out
    if neg_mode == "ns_whiten_v2_orthauto":     # 0626用户: 不用Pion, ortho留"自身stable_rank自适应top-r"(去固定比例+永不塌+可解释)
        from verl.utils.optim.pion import high_pass_ns
        # 与 orthrank 唯一差别: ortho 保留秩 = ortho 自身 stable_rank(自测有效秩, clamp[8,64]), 而非 pos 秩(去掉1:1固定耦合).
        #   关键=相对阈值自适应消尾(Pion NS 只是 matrix-free 实现); 此处用 stable_rank+rangefinder 显式实现, 永不塌(总留 r≥8).
        r_auto = _stable_rank(ortho)
        # 0627: ratio 可乘在自适应秩上(保守变体, PASP_ORTHO_RANK_RATIO<1 留更少ortho方向压grad_norm/稳定性); 默认1.0=纯自适应不变现状
        rq = max(_PASP_RMIN, min(int(round(_PASP_ORTHO_RANK_RATIO * r_auto)), _PASP_RMAX, min(gp.shape[0], gp.shape[1])))
        Om2 = torch.randn(n, rq, device=gp.device, dtype=gp.dtype)
        Y2 = ortho @ Om2
        for _ in range(int(power_iters)):                                # power iter 提升 top-r 子空间质量
            Y2 = ortho @ (ortho.t() @ Y2)
        Q2, _ = torch.linalg.qr(Y2, mode="reduced")                      # ortho top-r 左奇异基 [m,rq]
        ortho_topr = Q2 @ (Q2.t() @ ortho)                               # ortho 投到自身 top-r(r=自身有效秩)→ 留强方向丢尾
        pos_chan = gp + float(lambda_align) * aligned
        Pw = high_pass_ns(pos_chan,   promotion_steps=_PASP_NS_PROMO, suppression_steps=_PASP_NS_SUPP)
        Ow = high_pass_ns(ortho_topr, promotion_steps=_PASP_NS_PROMO, suppression_steps=_PASP_NS_SUPP)
        if float(Pw.norm()) <= 1e-6 * float(pos_chan.norm() + 1e-9): Pw = pos_chan
        if float(Ow.norm()) <= 1e-6 * float(ortho_topr.norm() + 1e-9): Ow = ortho_topr
        G = Pw.float() + Ow.float()                                      # ortho 自适应top-r 满权
        e_out = (ortho - ortho_topr).to(g_pos2d.dtype) if ef else None   # 残差 = 丢掉的 ortho 尾巴
        return G.to(g_pos2d.dtype), 1.0, Q, e_out
    if neg_mode == "ns_whiten_v2_orthopion":    # 0626用户: pos子空间Muon式NS(σ→1), ortho(负⊥正)用Pion式高通自动选秩→去掉top-r 1:1固定比例
        from verl.utils.optim.pion import high_pass_ns
        # pos 通道: Muon 式全正交化 (promo5/supp0, 同 winner); ortho 通道: Pion 式高通 (默认 promo0/supp5).
        #   高通本身"锚定强奇异值、压弱尾→0"=谱域自适应选秩, 天然替代手工 top-r 截断, 无需 rank 比例.
        pos_chan = gp + float(lambda_align) * aligned
        Pw = high_pass_ns(pos_chan, promotion_steps=_PASP_NS_PROMO, suppression_steps=_PASP_NS_SUPP)       # Muon式 pos
        Ow = high_pass_ns(ortho,    promotion_steps=_PASP_ORTHO_PROMO, suppression_steps=_PASP_ORTHO_SUPP)  # Pion式高通 ortho
        if float(Pw.norm()) <= 1e-6 * float(pos_chan.norm() + 1e-9): Pw = pos_chan
        if float(Ow.norm()) <= 1e-6 * float(ortho.norm() + 1e-9): Ow = ortho   # 防塌: 高通塌0→回退原 ortho
        G = Pw.float() + Ow.float()
        e_out = (ortho - Ow).to(g_pos2d.dtype) if ef else None   # EF残差 = 高通滤掉的 ortho 尾巴
        return G.to(g_pos2d.dtype), 1.0, Q, e_out
    if neg_mode == "ns_whiten_orthokeep":       # 0617用户: 压掉aligned(与pos同向的neg), 只留正交ortho满权, pos+ortho合并后单次pion高通(探索信号假说·反极性)
        from verl.utils.optim.pion import high_pass_ns
        G_asm = gp + ortho                                               # 丢 aligned, 留 pos + 全 ortho(完全正交的neg不压)
        Gw = high_pass_ns(G_asm, promotion_steps=_PASP_NS_PROMO, suppression_steps=_PASP_NS_SUPP)
        if float(Gw.norm()) > 1e-6 * float(G_asm.norm() + 1e-9):         # 防塌
            G_asm = Gw
        e_out = aligned.to(g_pos2d.dtype) if ef else None                # 残差 = 丢掉的 aligned
        return G_asm.to(g_pos2d.dtype), 0.0, Q, e_out
    if neg_mode == "ns_whiten_keeprank":        # 0618用户: 在orthokeep(留全ortho)基础上, 把aligned的主方向(top-r=pos秩)再加回来 → 留aligned主方向 + 留全ortho
        from verl.utils.optim.pion import high_pass_ns
        # 与 orthokeep 唯一差别: 不丢 aligned, 而是抽 aligned 自身 top-r(数量=pos秩)主方向加回.
        # 注: aligned=Q(Qᵀgn) 已在 span(Q)(秩=rq), 故 aligned_topr≈aligned → 本式≈最大保留单通道白化 high_pass(gp+gn).
        rq = int(Q.shape[1])                                             # aligned 保留秩 = pos 秩
        Om3 = torch.randn(n, rq, device=gp.device, dtype=gp.dtype)
        Y3 = aligned @ Om3
        for _ in range(int(power_iters)):                                # power iter 提升 top-r 子空间质量
            Y3 = aligned @ (aligned.t() @ Y3)
        Q3, _ = torch.linalg.qr(Y3, mode="reduced")                      # aligned top-r 左奇异基 [m,rq]
        aligned_topr = Q3 @ (Q3.t() @ aligned)                           # aligned 投到自身 top-r → 保最强 rq 个主方向
        G_asm = gp + aligned_topr + ortho                                # pos + aligned主方向 + 全 ortho
        Gw = high_pass_ns(G_asm, promotion_steps=_PASP_NS_PROMO, suppression_steps=_PASP_NS_SUPP)
        if float(Gw.norm()) > 1e-6 * float(G_asm.norm() + 1e-9):         # 防塌
            G_asm = Gw
        e_out = (aligned - aligned_topr).to(g_pos2d.dtype) if ef else None  # 残差 = 丢掉的 aligned 尾巴
        return G_asm.to(g_pos2d.dtype), 0.0, Q, e_out
    if neg_mode == "ns_whiten_allneg":          # 全压(2026-06-15 用户): pos保留, 全部neg(aligned+ortho)统一λn压制
        from verl.utils.optim.pion import high_pass_ns
        # 与 ns_whiten_v2 唯一差别: aligned 不再 λa=1 并入 pos 通道, 而是和 ortho 一起算进 g_neg 整体压 λn.
        #   v2:   G = whiten(g_pos + 1.0·aligned) + λn·whiten(ortho)        # aligned 全保
        #   allneg: G = whiten(g_pos)             + λn·whiten(g_neg)        # 所有 neg 一视同仁压
        # λn 仍沿用 :218 的式子(基于 ‖ortho‖), 与 v2 同口径 → 单变量=aligned 是否被压.
        Pw = high_pass_ns(gp, promotion_steps=_PASP_NS_PROMO, suppression_steps=_PASP_NS_SUPP)
        Nw = high_pass_ns(gn, promotion_steps=_PASP_NS_PROMO, suppression_steps=_PASP_NS_SUPP)
        if float(Pw.norm()) <= 1e-6 * float(sp + 1e-9): Pw = gp
        if float(Nw.norm()) <= 1e-6 * float(gn.norm() + 1e-9): Nw = gn
        G = Pw.float() + lam_n * Nw.float()
        e_out = ((1.0 - lam_n) * gn).to(g_pos2d.dtype) if ef else None
        return G.to(g_pos2d.dtype), lam_n, Q, e_out
    G = gp + float(lambda_align) * aligned + lam_n * ortho
    if neg_mode == "ns_whiten":                 # P2: 极性组合(shrink)后, 对 G 做 NS 正交化(像 pion)
        from verl.utils.optim.pion import high_pass_ns   # 复用 pion 实现, 单变量
        # promo=5/supp=0: 标准 Muon 五步 NS, 奇异值→≈1(满秩稳健, PASP 瞬时梯度无动量, 不可用 pion 纯suppression 否则塌0).
        #   (pion 默认 promo=0/supp=5 高通是给带动量低秩梯度的; 经 NS_PROMO/NS_SUPP env 可切到 pion 精确模式)
        Gw = high_pass_ns(G, promotion_steps=_PASP_NS_PROMO, suppression_steps=_PASP_NS_SUPP)
        if float(Gw.norm()) > 1e-6 * float(G.norm() + 1e-9):   # 防塌: 白化退化成≈0 则回退原 G
            G = Gw.to(g_pos2d.dtype)
        # 白化已是预条件(像 pion 无二阶矩); step() 对该 mode 绕过 adagrad 除法 + 配 rms_scale 复原步长
    # EF 残差 = 本步未施加的弥散 neg = (1-λn)·ortho_eff (lam_min>0 保证收缩, e 有界)
    e_out = ((1.0 - lam_n) * ortho).to(g_pos2d.dtype) if ef else None
    return G.to(g_pos2d.dtype), lam_n, Q, e_out


def _pasp_grad(g_pos: Tensor, g_neg: Tensor, rank: int, power_iters: int,
               lambda_align: float, target_ratio: float, lam_min: float, lam_max: float,
               Q_cached: Optional[Tensor] = None, e_in: Optional[Tensor] = None, ef: bool = False,
               suppress: bool = True, neg_mode: str = "shrink", neg_topr_coef: float = 1.0):
    """DTensor-aware PASP (range finder; mirror _neg_proj_grad)。非 2D 权重回退标量收缩 g_pos+target·g_neg。
    返回 (eff_grad, λn_used, Q, e_out)。Q∈[m,r] 完整基供周期缓存; e_out 为 EF 残差(sharded 回上游存)。"""
    is_dt = _HAS_DTENSOR and isinstance(g_pos, DTensor)
    if is_dt:
        mesh, plc = g_pos.device_mesh, g_pos.placements
        gpf = g_pos.full_tensor()
        gnf = g_neg.full_tensor() if isinstance(g_neg, DTensor) else g_neg
        e_full = e_in.full_tensor() if (ef and isinstance(e_in, DTensor)) else (e_in if ef else None)
    else:
        gpf, gnf = g_pos, g_neg
        e_full = e_in if ef else None
    if gpf.dim() != 2:                          # 1D 权重: 无子空间, 标量收缩
        return g_pos + float(target_ratio) * g_neg, float(target_ratio), None, None
    Gc, lam_n, Q, e_out = _pasp_rangefinder_2d(gpf, gnf, rank, power_iters, lambda_align,
                                               target_ratio, lam_min, lam_max, Q_cached, e_full, ef, suppress,
                                               neg_mode, neg_topr_coef)
    if is_dt:
        Gd = distribute_tensor(Gc, device_mesh=mesh, placements=plc)
        ed = distribute_tensor(e_out, device_mesh=mesh, placements=plc) if (ef and e_out is not None) else None
        return Gd, lam_n, Q, ed
    return Gc, lam_n, Q, e_out


def _subspace_overlap(Qa: Tensor, Qb: Tensor) -> float:
    """两正交基 [m,r] 的子空间重合度 = ‖Qaᵀ Qb‖_F² / r ∈ [0,1] (1=同一子空间)。
    PASP auto-switch 判稳: pos 子空间在相邻刷新间不再旋转(重合高)= pos 秩稳定。"""
    r = int(min(Qa.shape[1], Qb.shape[1]))
    if r <= 0:
        return 0.0
    M = Qa[:, :r].float().t() @ Qb[:, :r].float()
    return float(M.pow(2).sum().item() / r)


def mask_plateau_fire(churn_seq, rho_frac: float = 0.5, flat_eps: float = 0.20, K: int = 2):
    """尺度不变 peak-relative + flatness mask 平台检测器 (旗舰 C1).

    输入 churn_seq: list[float], 每个=相邻窗口 mask 的 Early-Bird 归一化 Hamming 距 d_t.
    判据 (修复 naive Early-Bird 固定阈值在高稀疏下的双阈值失败):
      below_t = d_t <= rho_frac * peak   (churn 从历史峰值减半)
      flat_t  = |d_t - d_{t-1}| / max(d_t,eps) < flat_eps   (已进平台)
      连续 K 步 below&flat → 触发, 返回触发处的序列 index (0-based, 相对 churn_seq).
    返回 None 表示未触发. 这是尺度不变量: 不依赖绝对阈值, 故跨 mask 阈值(topk 0.1/0.2/0.4)迁移.
    """
    peak = 0.0
    consec = 0
    d_prev = None
    for i, d in enumerate(churn_seq):
        if d is None:
            d_prev = None
            continue
        peak = max(peak, d)
        below = d <= rho_frac * peak
        flat = (d_prev is not None) and (abs(d - d_prev) / max(d, 1e-9) < flat_eps)
        consec = consec + 1 if (below and flat) else 0
        if consec >= K:
            return i
        d_prev = d
    return None


def _diag_flags() -> dict[str, bool]:
    raw = os.environ.get("SPARSE_DIAG", "energy").lower()
    if raw in {"0", "off", "none", "false"}:
        return {"energy": False, "jaccard": False, "persistence": False}
    if raw in {"1", "all", "full", "true"}:
        return {"energy": True, "jaccard": True, "persistence": True}
    tokens = {t.strip() for t in raw.split(",") if t.strip()}
    return {
        "energy": "energy" in tokens,
        "jaccard": "jaccard" in tokens,
        "persistence": "persistence" in tokens,
    }

SPARSE_PHASE_DENSE = 0
SPARSE_PHASE_RAMP = 1
SPARSE_PHASE_STEADY = 2
SPARSE_PHASE_KL_ADAPTIVE = 3
SPARSE_PHASE_GRAD_ADAPTIVE = 4
SPARSE_PHASE_UNIFIED = 5


def sparse_schedule_phase_index(global_step: int, sparse_start_step: int, sparse_ramp_steps: int) -> int:
    """与 ``SparseAdagrad`` 内 ``_global_step``（每次 ``step()`` 先 +1 后的值）对齐的三段式阶段。

    0 = dense 预热（步数尚未到达 ``sparse_start_step``，调度上保持满目标）
    1 = ramp（``sparse_ramp_steps > 0`` 且 ``sparse_start_step <= step < sparse_start_step + R``）
    2 = steady（ramp 结束或 ``R==0`` 时进入的稳定稀疏档）

    注意：G-01 等 ``sparse_threshold_scale=0`` 时仍可能不掩码，但阶段仍按步数边界划分，便于日志对齐。
    """
    if global_step < sparse_start_step:
        return SPARSE_PHASE_DENSE
    if sparse_ramp_steps > 0 and global_step < sparse_start_step + sparse_ramp_steps:
        return SPARSE_PHASE_RAMP
    return SPARSE_PHASE_STEADY


class SparseAdagrad(Optimizer):
    """AdaGrad with update clipping and delayed sparse masking.

    This optimizer is designed for dense transformer training. "Sparse" here
    refers to masking part of the preconditioned update after a dense warmup
    period, rather than sparse tensor gradients.

    RL 注意：与 AdamW 公平对比时需使用**同量级**的 ``lr``（例如 5e-6）与调度；``lr`` 大两个数量级会单独导致策略崩溃，
    与稀疏/裁剪逻辑无关（见 run_dapo_am_sparseadagrad_16k 与 adam_16k 对齐项）。
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-2,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        adaptive_update_clip_factor: float = 0.0,
        adaptive_update_clip_beta: float = 0.95,
        sparse_start_step: int = 0,
        sparse_mode: str = "threshold",
        sparse_threshold_scale: float = 0.0,
        sparse_topk_ratio: float = 1.0,
        sparse_ramp_steps: int = 0,
        sparse_min_numel: int = 4096,
        warmup_momentum_beta: float = 0.0,
        warmup_momentum_steps: int = 0,
        decoupled_weight_decay: bool = False,
        sparse_ema_beta: float = 0.9,  # 兼容 SparseAdamW 接口(EMA 评分模式专用); Adagrad 暂未用 ema 评分,接受并忽略
        kl_adaptive: bool = False,
        kl_target_low: float = 0.001,
        kl_target_high: float = 0.002,
        kl_ema_beta: float = 0.95,
        rho_delta: float = 0.05,
        rho_min: float = 0.3,
        rho_max: float = 1.0,
        rho_floor_scale: float = 0.0,
        acsa_alpha: float = 0.0,
        grad_adaptive: bool = False,
        grad_ema_beta: float = 0.95,
        grad_warmup_steps: int = 5,
        # --- dual-channel pos/neg second-moment accumulation (paper extension) ---
        dual_mode: str = "off",                   # off | approx | true
        dual_score: str = "magnitude",            # magnitude | neg_energy (mask 评分模式)
        dual_patience_down: int = 5,              # K_down: all_correct 连降几步切 neg_suppress
        dual_patience_up: int = 3,                # K_up:   all_correct 回升几步切 magnitude
        dual_rescale: bool = False,               # 能量守恒 rescale 开关
        dual_neg_coef: float = 1.0,               # 融合 g = g_pos + dual_neg_coef * g_neg
        neg_proj_rank: int = 0,                   # >0: 硬投影切除 g_neg top-r 左奇异子空间; -2: NS 软谱滤波(免SVD); (需 dual_mode=true + FSDP2 DTensor 2D)
        neg_soft_ns_steps: int = 5,               # neg_proj_rank=-2/-3 时 NS 步数(软投影锐度)
        neg_soft_alpha: float = 1.0,              # -2 软滤波强度: G - alpha·F(neg-Gram)·G
        pasp_lambda_align: float = 1.0,           # -3 PASP: 保留 pos-aligned neg 的系数 (1=全保)
        pasp_target_ratio: float = 0.1,           # -3 PASP: 弥散 neg 相对 g_pos 的目标干扰比 (自适应 λn 锚, 去硬编码 0.1)
        pasp_rank: int = 16,                      # -3 PASP: PASP_AUTORANK=0 时使用的固定秩消融值
        pasp_power_iters: int = 1,                # -3 PASP: range finder power iteration 次数 (子空间质量)
        pasp_refresh_steps: int = 1,              # -3 PASP: 正向子空间 Q 默认每步刷新；>1 仅作缓存消融
        pasp_lam_min: float = 0.0,                # -3 PASP: λn 下界 (放开, 去掉 0.1 牵引 → 可证伪)
        pasp_lam_max: float = 1.0,                # -3 PASP: λn 上界
        pasp_neg_mode: str = "shrink",            # -3 PASP neg 处理: 'shrink'(原:λa保aligned+λn压ortho) | 'orth_lowrank'(V1:去aligned让pos⊥neg+ortho只留top-r奇异)
        pasp_neg_topr_coef: float = 1.0,          # V1: top-r 正交 neg 的系数 (1=全强度; 可扫 0.5 等)
        pasp_error_feedback: bool = False,        # -3 PASP: EF21-on-ortho 补偿被收缩丢弃的弥散 neg (需 lam_min>0 保收缩)
        pasp_auto_switch: bool = False,           # -3 PASP: 自动切换——pos 子空间稳定后才开抑制(之前 λn=1=plain dual)
        pasp_stable_thresh: float = 0.60,         # auto-switch: 相邻刷新 Q 子空间重合 ≥此 算稳 (rank16 嵌 eff-rank≈11 子空间, 稳态重合上限≈11/16≈0.69, 故取0.6 隔开漂移~0.02)
        pasp_stable_patience: int = 2,            # auto-switch: 连续稳定刷新次数→开启抑制
        rms_scale: float = 0.0,                    # >0: Muon 式 adjusted_lr = lr × rms_scale × √(max(A,B)) (2D权重, 追 Muon 有效步长); 0=关
    ):
        if lr <= 0:
            raise ValueError(f"Invalid lr: {lr}")
        if eps <= 0:
            raise ValueError(f"Invalid eps: {eps}")
        if adaptive_update_clip_factor < 0:
            raise ValueError("adaptive_update_clip_factor must be >= 0")
        if not 0.0 <= adaptive_update_clip_beta < 1.0:
            raise ValueError("adaptive_update_clip_beta must be in [0, 1)")
        if sparse_mode not in {"threshold", "topk"}:
            raise ValueError(f"Unsupported sparse_mode: {sparse_mode}")
        if not 0.0 < sparse_topk_ratio <= 1.0:
            raise ValueError("sparse_topk_ratio must be in (0, 1]")
        if sparse_threshold_scale < 0:
            raise ValueError("sparse_threshold_scale must be >= 0")
        if sparse_start_step < 0 or sparse_ramp_steps < 0:
            raise ValueError("sparse_start_step and sparse_ramp_steps must be >= 0")
        if not 0.0 <= warmup_momentum_beta < 1.0:
            raise ValueError("warmup_momentum_beta must be in [0, 1)")
        if warmup_momentum_steps < 0:
            raise ValueError("warmup_momentum_steps must be >= 0")

        self._kl_adaptive = bool(kl_adaptive)
        self._kl_target_low = float(kl_target_low)
        self._kl_target_high = float(kl_target_high)
        self._kl_ema_beta = float(kl_ema_beta)
        self._rho_delta = float(rho_delta)
        self._rho_min = float(rho_min)
        self._rho_max = float(rho_max)
        self._kl_ema: Optional[float] = None
        self._adaptive_rho: float = rho_max
        self._grad_adaptive = bool(grad_adaptive)
        self._grad_ema_beta = float(grad_ema_beta)
        self._grad_warmup_steps = int(grad_warmup_steps)
        self._grad_norm_ema: Optional[float] = None
        self._grad_baseline: Optional[float] = None
        # Layer 3: dynamic rho floor driven by batch pos_ratio (Hydra 可传 rho_floor_scale)
        self._rho_floor_scale: float = float(rho_floor_scale)
        # Layer 2: advantage-aware coordinate boost (ACSA-L2) — scale keep-ratio by (1 + alpha * p+)
        self._acsa_alpha: float = float(acsa_alpha)
        self._current_pos_ratio: float = 0.5  # default: 50% positive
        # unified grad_norm trigger from dp_actor
        self._unified_rho: float = 1.0
        self._unified_layer: int = 0
        self._unified_active: bool = False
        # --- dual-channel state (see __init__ for semantic) ---
        # Hydra may coerce string "true" -> bool True; accept both.
        if isinstance(dual_mode, bool):
            dual_mode = "true" if dual_mode else "off"
        # off:       no dual; single-channel Adagrad (baseline)
        # stat_only: collect g_pos/g_neg -> sum_pos_sq/sum_neg_sq, but preconditioner走 (g_pos+g_neg)² 单通道路径 (训练等价 single)
        # approx:    cheap split using batch pos_ratio (no extra backward), dual preconditioner
        # true:      real two-backward split, dual preconditioner
        if dual_mode not in {"off", "approx", "true", "stat_only"}:
            raise ValueError(f"dual_mode must be off/approx/true/stat_only, got {dual_mode}")
        if dual_score not in {"magnitude", "neg_energy"}:
            raise ValueError(f"dual_score must be magnitude/neg_energy, got {dual_score}")
        self._dual_mode: str = dual_mode
        self._dual_score: str = dual_score
        self._dual_patience_down: int = int(dual_patience_down)
        self._dual_patience_up: int = int(dual_patience_up)
        self._dual_rescale: bool = bool(dual_rescale)
        self._dual_neg_coef: float = float(dual_neg_coef)
        self._neg_proj_rank: int = int(neg_proj_rank)
        self._neg_soft_ns_steps: int = int(neg_soft_ns_steps)
        self._neg_soft_alpha: float = float(neg_soft_alpha)
        self._pasp_lambda_align: float = float(pasp_lambda_align)
        self._pasp_target_ratio: float = float(pasp_target_ratio)
        self._pasp_rank: int = int(pasp_rank)
        self._pasp_power_iters: int = int(pasp_power_iters)
        self._pasp_refresh_steps: int = int(pasp_refresh_steps)
        self._pasp_lam_min: float = float(pasp_lam_min)
        self._pasp_lam_max: float = float(pasp_lam_max)
        self._pasp_neg_mode: str = str(pasp_neg_mode)         # 'shrink'(原PASP) | 'orth_lowrank'(V1: pos⊥neg + neg 同秩 top-r)
        self._pasp_neg_topr_coef: float = float(pasp_neg_topr_coef)
        self._pasp_error_feedback: bool = bool(pasp_error_feedback)
        self._pasp_ef_norm_list: list = []        # 每步重置: 各层 EF 残差范数 → 监控收敛/发散
        self._pasp_auto_switch: bool = bool(pasp_auto_switch)
        self._pasp_stable_thresh: float = float(pasp_stable_thresh)
        self._pasp_stable_patience: int = int(pasp_stable_patience)
        self._pasp_stable: dict = {}              # id(p) -> [stable_count, active] (抑制是否已开)
        self._pasp_overlap_list: list = []        # 每步重置: 刷新步的子空间重合度
        self._pasp_rank_eff_list: list = []       # 每步重置: 各层自适应秩 (B/_PASP_AUTORANK) → 跨层 mean/分位
        self._pasp_Q_cache: dict = {}             # id(p) -> 缓存基 Q∈[m,r] (周期刷新, 不入 ckpt state)
        self._pasp_lam_n_list: list = []          # 每步重置: 各 2D 层 λn → 跨层 mean/分位 (可证伪)
        if self._neg_proj_rank != 0 and dual_mode != "true":
            raise ValueError("neg_proj_rank!=0 (硬投影>0 或 软滤波=-2) 需要 dual_mode='true' (要 g_neg)")
        self._rms_scale: float = float(rms_scale)  # Muon 式 adjusted_lr per-2D 缩放; 0=关
        # rms_scale 排除 embed/lm_head (vocab矩阵, √max 会爆): max fan 上界 + 维度比上界
        self._rms_max_fan: int = 32768          # max(A,B)>此 → 视为 vocab 矩阵, 不放大
        self._rms_exclude_ratio: float = 8.0    # max/min>此 (如 embed 37×) → vocab 矩阵, 不放大
        # patience counters + current mask mode (two-state switch driven by all_correct)
        self._dual_mask_mode: str = "magnitude"   # default healthy
        self._dual_p_down: int = 0
        self._dual_p_up: int = 0
        self._dual_ac_ema: Optional[float] = None
        # external dual-gradient buffer (set by trainer before .step() when dual_mode='true')
        self._dual_grad_pos: dict = {}            # id(param) -> g_pos tensor
        self._dual_grad_neg: dict = {}
        # diagnostics
        self._last_dual_metrics: dict[str, float] = {}
        # v9: pos_energy_share 时间序列追踪 (EMA + 历史最大 + 反转量)
        # 用于自动检测 "pos 主导 → neg 主导" 的训练阶段切换
        self._dual_pos_share_ema: Optional[float] = None
        self._dual_pos_share_max: float = 0.0

        defaults = dict(
            lr=lr,
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
            warmup_momentum_beta=warmup_momentum_beta,
            warmup_momentum_steps=warmup_momentum_steps,
            decoupled_weight_decay=decoupled_weight_decay,
        )
        super().__init__(params, defaults)
        self._global_step = 0
        self._update_rms_ema: Optional[float] = None
        self._last_metrics: dict[str, float] = {}
        self._diag = _diag_flags()
        if not hasattr(self, '_kl_adaptive'):
            self._kl_adaptive = False
            self._kl_ema = None
            self._adaptive_rho = 1.0

    def state_dict(self) -> dict[str, Any]:
        d = super().state_dict()
        d["_sparse_extra"] = {
            "global_step": int(self._global_step),
            "update_rms_ema": None if self._update_rms_ema is None else float(self._update_rms_ema),
            "kl_ema": None if self._kl_ema is None else float(self._kl_ema),
            "adaptive_rho": float(self._adaptive_rho),
            "grad_norm_ema": None if self._grad_norm_ema is None else float(self._grad_norm_ema),
            "grad_baseline": None if self._grad_baseline is None else float(self._grad_baseline),
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
            kl_ema = extra.get("kl_ema")
            self._kl_ema = None if kl_ema is None else float(kl_ema)
            self._adaptive_rho = float(extra.get("adaptive_rho", getattr(self, '_rho_max', 1.0)))
            gn_ema = extra.get("grad_norm_ema")
            self._grad_norm_ema = None if gn_ema is None else float(gn_ema)
            gb = extra.get("grad_baseline")
            self._grad_baseline = None if gb is None else float(gb)
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
        total_sq = 0.0
        total_elems = 0
        # --- dual-channel diagnostic accumulators (paper extension) ---
        d_pos_total = 0.0; d_neg_total = 0.0
        d_gpos_sq = 0.0; d_gneg_sq = 0.0; d_cross = 0.0
        d_precond_sgl_sq = 0.0; d_precond_dual_sq = 0.0
        d_pos_share_samples = []
        d_per_layer_sparsity = []
        d_nan_count = 0; d_inf_count = 0
        d_precond_min = float('inf'); d_precond_max = 0.0
        d_dual_param_count = 0
        # ---- 坐标级 pos/neg 重合分析 (per-coordinate boolean active 交集, 聚合版) ----
        # pos_active_i = |g_pos_i| > tau_pos; neg_active_i = |g_neg_i| > tau_neg (tau=95 分位, 抽样估)
        # overlap = pos_active & neg_active. 统计三分占比 + 重合区同号/反号 + 重合区谁能量大.
        d_ov_total = 0          # 总坐标数
        d_ov_pos_active = 0     # pos 活跃坐标数
        d_ov_neg_active = 0     # neg 活跃坐标数
        d_ov_overlap = 0        # 重合 (pos & neg 都活跃)
        d_ov_overlap_samedir = 0   # 重合且同号 (sign(g_pos)==sign(g_neg))
        d_ov_overlap_pos_gt_neg = 0  # 重合区 |g_pos|>|g_neg| 的坐标数
        self._dual_mask_diag = {
            "pos_overlap_num": 0, "pos_overlap_den": 0,
            "update_pos_dom_sq": 0.0, "update_neg_dom_sq": 0.0,
            "rescale_factors": [], "E_drop_sum": 0.0, "E_keep_sum": 0.0,
        }

        # Pre-compute dual-channel split (per-param g_pos / g_neg) if enabled.
        # 'true'   = trainer fed g_pos/g_neg explicitly via set_dual_grads()
        # 'approx' = derive from current p.grad using batch-level pos_ratio scalar
        # 'off'    = use single channel (existing path)
        dual_on = (self._dual_mode != "off")
        self._pasp_lam_n_list = []   # PASP(-3): 本步各 2D 层 λn (在第1个loop里append, 末尾聚合); 必须在第1个loop前重置
        self._pasp_ef_norm_list = [] # PASP(-3)+EF: 本步各层残差范数, 末尾聚合监控收敛
        self._pasp_overlap_list = [] # PASP(-3)+auto-switch: 本步刷新的子空间重合度
        self._pasp_rank_eff_list = [] # PASP(-3)+autorank: 本步各层自适应秩, 末尾聚合

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("SparseAdagrad expects dense gradients.")

                state = self.state[p]
                if len(state) == 0:
                    state["sum_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["ema_grad"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                # v9-resume fix: 如果是从 dual_mode=off 保存的 ckpt 恢复后启用 dual_mode='true',
                # state 已存在但缺 sum_pos_sq / sum_neg_sq, 需要懒加载初始化
                if dual_on and "sum_pos_sq" not in state:
                    state["sum_pos_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["sum_neg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                # dual-channel split: get g_pos / g_neg
                if dual_on:
                    pid = id(p)
                    if self._dual_mode in ("true", "stat_only") and pid in self._dual_grad_pos:
                        g_pos = self._dual_grad_pos[pid]
                        g_neg = self._dual_grad_neg.get(pid, torch.zeros_like(p))
                    else:
                        # approx (or true 模式下未喂梯度时 fallback): split using batch pos_ratio
                        pr = max(0.0, min(1.0, self._current_pos_ratio))
                        g_pos = grad * pr
                        g_neg = grad * (1.0 - pr)
                    # accumulate separate second moments (always, in any dual mode)
                    state["sum_pos_sq"].addcmul_(g_pos, g_pos, value=1.0)
                    state["sum_neg_sq"].addcmul_(g_neg, g_neg, value=1.0)
                    # ---- gradient-level dual stats (per-step, before any precond/mask) ----
                    with torch.no_grad():
                        d_gpos_sq += g_pos.float().pow(2).sum().item()
                        d_gneg_sq += g_neg.float().pow(2).sum().item()
                        d_cross   += (g_pos.float() * g_neg.float()).sum().item()
                        # 坐标级 pos/neg 重合分析 (聚合版布尔交集); 仅大张量 (≥4096) 算, 控成本
                        if p.numel() >= 4096:
                            # DTensor 安全: 诊断统计在本地分片上算 (索引 DTensor 会 mixed-tensor 报错)
                            _gpl = g_pos.to_local() if (_HAS_DTENSOR and isinstance(g_pos, DTensor)) else g_pos
                            _gnl = g_neg.to_local() if (_HAS_DTENSOR and isinstance(g_neg, DTensor)) else g_neg
                            ap = _gpl.float().abs().reshape(-1)
                            an = _gnl.float().abs().reshape(-1)
                            n_el = ap.numel()
                            # 抽样估 95 分位作 tau (避 torch.quantile 16M 上限 + 提速)
                            if n_el > 200000:
                                idx = torch.randperm(n_el, device=ap.device)[:200000]
                                tau_pos = torch.quantile(ap[idx], 0.95)
                                tau_neg = torch.quantile(an[idx], 0.95)
                            else:
                                tau_pos = torch.quantile(ap, 0.95)
                                tau_neg = torch.quantile(an, 0.95)
                            pos_act = ap > tau_pos
                            neg_act = an > tau_neg
                            ov = pos_act & neg_act
                            n_ov = int(ov.sum().item())
                            d_ov_total += n_el
                            d_ov_pos_active += int(pos_act.sum().item())
                            d_ov_neg_active += int(neg_act.sum().item())
                            d_ov_overlap += n_ov
                            if n_ov > 0:
                                gp_f = _gpl.float().reshape(-1)
                                gn_f = _gnl.float().reshape(-1)
                                same = (torch.sign(gp_f[ov]) == torch.sign(gn_f[ov]))
                                d_ov_overlap_samedir += int(same.sum().item())
                                d_ov_overlap_pos_gt_neg += int((ap[ov] > an[ov]).sum().item())
                    # stat_only: 训练侧用单通道行为, eff/precond 走原 Adagrad 路径
                    if self._dual_mode == "stat_only":
                        eff = self._prepare_step_grad(p, grad, group)
                    elif self._neg_proj_rank > 0:
                        # neg 谱手术(硬投影): 保留全 neg (coef=1), 但切除 neg 的 top-r 强奇异子空间
                        merged = g_pos + g_neg
                        eff = _neg_proj_grad(merged, g_neg, self._neg_proj_rank)
                    elif self._neg_proj_rank == -2:
                        # neg 软谱滤波(NS 免SVD): merged - alpha·F(neg-Gram)·merged; ns_steps 调锐度
                        merged = g_pos + g_neg
                        eff, _neg_rm = _neg_soft_filter_grad(
                            merged, g_neg, self._neg_soft_ns_steps, self._neg_soft_alpha)
                        # 暂存到瞬态属性; _last_metrics 在 step 末整体重建, 故末尾再写入 (见 neg_proj_rank 旁)
                        self._neg_filter_rm_last = float(_neg_rm)
                    elif self._neg_proj_rank == -3 and not (
                        p.dim() == 2
                        and max(p.shape[0], p.shape[1]) <= self._rms_max_fan
                        and (max(p.shape[0], p.shape[1]) / max(min(p.shape[0], p.shape[1]), 1)) <= self._rms_exclude_ratio
                    ):
                        # vocab(embed/lm_head) 或 1D: 不白化, dual 合并梯度 → 走正常 adagrad 预条件 (对齐 pion 给它们 AdamW 的路由)
                        state["_pasp_whitened"] = False
                        eff = g_pos + self._dual_neg_coef * g_neg
                    elif self._neg_proj_rank == -3:
                        state["_pasp_whitened"] = True
                        # PASP(range finder): g_pos + λa·P_pos g_neg + λn(自适应)·(I-P_pos)g_neg
                        # 默认从当前 g_pos 每步重估 stable rank 并重算 Q；仅显式缓存消融才跨步复用。
                        if _PASP_MOMENTUM > 0.0:        # v3: pos/neg 各自动量平滑后再喂白化(像 pion)
                            _mp = state.get("mom_pos");  _mn = state.get("mom_neg")
                            if _mp is None: _mp = torch.zeros_like(g_pos); state["mom_pos"] = _mp
                            if _mn is None: _mn = torch.zeros_like(g_neg); state["mom_neg"] = _mn
                            _mp.mul_(_PASP_MOMENTUM).add_(g_pos)    # buf = β·buf + grad
                            _mn.mul_(_PASP_MOMENTUM).add_(g_neg)
                            if _PASP_NESTEROV:
                                g_pos = g_pos + _PASP_MOMENTUM * _mp
                                g_neg = g_neg + _PASP_MOMENTUM * _mn
                            else:
                                g_pos = _mp.clone(); g_neg = _mn.clone()
                        _rsteps = _PASP_REFRESH_ENV if _PASP_REFRESH_ENV > 0 else self._pasp_refresh_steps
                        _refresh = (self._global_step % max(_rsteps, 1) == 0)
                        _old_Q = self._pasp_Q_cache.get(id(p))
                        _q_cache = None if _refresh else _old_Q
                        _e_in = state.get("pasp_ef_residual") if self._pasp_error_feedback else None
                        # auto-switch: pos 子空间未稳前不抑制 (suppress=False → λn=1=plain dual)
                        if self._pasp_auto_switch:
                            _suppress = bool(self._pasp_stable.get(id(p), [0, False])[1])
                        else:
                            _suppress = True
                        eff, _lam_n, _Q, _e_out = _pasp_grad(
                            g_pos, g_neg, self._pasp_rank, self._pasp_power_iters,
                            self._pasp_lambda_align, self._pasp_target_ratio,
                            self._pasp_lam_min, self._pasp_lam_max, Q_cached=_q_cache,
                            e_in=_e_in, ef=self._pasp_error_feedback, suppress=_suppress,
                            neg_mode=self._pasp_neg_mode, neg_topr_coef=self._pasp_neg_topr_coef)
                        # auto-switch 判稳: 刷新步比较 new Q vs old Q 子空间重合
                        if self._pasp_auto_switch and _refresh and _old_Q is not None and _Q is not None:
                            _ov = _subspace_overlap(_old_Q, _Q)
                            _cnt, _active = self._pasp_stable.get(id(p), [0, False])
                            if _ov >= self._pasp_stable_thresh:
                                _cnt += 1
                                if _cnt >= self._pasp_stable_patience:
                                    _active = True
                            else:
                                _cnt = 0
                            self._pasp_stable[id(p)] = [_cnt, _active]
                            self._pasp_overlap_list.append(_ov)
                        if _Q is not None:                       # 真子空间路径 (排除 1D / g_pos≈0 标量回退)
                            self._pasp_Q_cache[id(p)] = _Q
                            self._pasp_lam_n_list.append(float(_lam_n))
                            if _PASP_AUTORANK:                   # B: 记录每步自适应秩, 末尾聚合监控
                                self._pasp_rank_eff_list.append(int(_Q.shape[1]))
                        elif self._pasp_neg_mode == "polar_sparse_whiten":
                            self._pasp_lam_n_list.append(float(_lam_n))   # psw 无 Q: λ=坐标级均值, 仍记录分布
                        if self._pasp_error_feedback and _e_out is not None:
                            state["pasp_ef_residual"] = _e_out   # EF 残差存 state, 下步累加; lam_min>0 保证有界
                            self._pasp_ef_norm_list.append(float(_e_out.norm() if not (_HAS_DTENSOR and isinstance(_e_out, DTensor)) else _e_out.to_local().norm()))
                    else:
                        eff = g_pos + self._dual_neg_coef * g_neg
                else:
                    eff = self._prepare_step_grad(p, grad, group)
                if group["weight_decay"] != 0 and not group["decoupled_weight_decay"]:
                    eff = eff.add(p, alpha=group["weight_decay"])
                state["_eff_for_step"] = eff

                _ns_whiten = (self._neg_proj_rank == -3 and self._pasp_neg_mode in ("ns_whiten", "ns_whiten_v2", "ns_whiten_v2_orthrank", "ns_whiten_v2_orthopion", "ns_whiten_v2_orthauto", "ns_whiten_orthokeep", "ns_whiten_keeprank", "ns_whiten_nopolar", "polar_sparse_whiten") and state.get("_pasp_whitened", False))
                if dual_on and self._dual_mode != "stat_only":
                    # dual-preconditioner: combine both channels' second moments
                    precond_sq = state["sum_pos_sq"] + (self._dual_neg_coef ** 2) * state["sum_neg_sq"]
                    # mirror to sum_sq for downstream code that reads it (diagnostics)
                    state["sum_sq"].copy_(precond_sq)
                    if _ns_whiten:
                        update = eff                       # 白化更新即预条件(pion 式), 不再除二阶矩
                    else:
                        update = eff / (precond_sq.sqrt() + group["eps"])
                else:
                    # single-channel preconditioner (off OR stat_only): standard Adagrad
                    state["sum_sq"].addcmul_(eff, eff, value=1.0)
                    update = eff / (state["sum_sq"].sqrt() + group["eps"])

                # --- dual-channel diagnostics (state level) ---
                if dual_on:
                    with torch.no_grad():
                        sp = state["sum_pos_sq"]; sn = state["sum_neg_sq"]
                        d_pos_total += sp.float().sum().item()
                        d_neg_total += sn.float().sum().item()
                        # precond comparison (dual vs single equivalent)
                        precond_dual = sp + (self._dual_neg_coef ** 2) * sn
                        precond_sgl  = state["sum_sq"]   # 单通道路径下 = (g_full)² 累积
                        d_precond_dual_sq += precond_dual.float().sum().item()
                        d_precond_sgl_sq  += precond_sgl.float().sum().item()
                        # min/max for numerical sanity
                        pd_min = precond_dual.min().item(); pd_max = precond_dual.max().item()
                        if pd_min < d_precond_min: d_precond_min = pd_min
                        if pd_max > d_precond_max: d_precond_max = pd_max
                        # NaN/Inf count
                        d_nan_count += int(torch.isnan(update).sum().item())
                        d_inf_count += int(torch.isinf(update).sum().item())
                        # sample pos_share distribution from this layer (1k coords max)
                        if p.numel() >= 1024:
                            _spl = sp.to_local() if (_HAS_DTENSOR and isinstance(sp, DTensor)) else sp
                            _snl = sn.to_local() if (_HAS_DTENSOR and isinstance(sn, DTensor)) else sn
                            flat_p = _spl.flatten()[:1024]; flat_n = _snl.flatten()[:1024]
                            ratio = (flat_p / (flat_p + flat_n + 1e-12)).cpu().tolist()
                            d_pos_share_samples.extend(ratio)
                        d_dual_param_count += 1

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

        # diagnostic accumulators (kept on host as floats; updated per tensor)
        diag = self._diag
        upd_sq_sum = 0.0
        kept_sq_sum = 0.0
        masked_elems_sparse = 0  # numel counted only on tensors that were actually masked
        mask_inter_sum = 0
        mask_union_sum = 0
        persist_sum = 0.0
        persist_count = 0
        # (dual diagnostic accumulators already initialized at step entry; _pasp_lam_n_list 在第1个loop前已重置)

        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                eff = state.pop("_eff_for_step")
                if self._neg_proj_rank == -3 and self._pasp_neg_mode in ("ns_whiten", "ns_whiten_v2", "ns_whiten_v2_orthrank", "ns_whiten_v2_orthopion", "ns_whiten_v2_orthauto", "ns_whiten_orthokeep", "ns_whiten_keeprank", "ns_whiten_nopolar", "polar_sparse_whiten") and state.get("_pasp_whitened", False):
                    update = eff                           # ns_whiten: 仅 hidden 2D 白化即预条件, 绕过 adagrad 除法
                else:
                    update = eff / (state["sum_sq"].sqrt() + group["eps"])
                update.mul_(clip_scale)

                masked_update = update
                was_masked = False
                if self._unified_active:
                    if self._unified_rho < 1.0 and p.numel() >= group["sparse_min_numel"]:
                        masked_update = self._apply_kl_adaptive_mask(update)
                        was_masked = True
                elif self._kl_adaptive or self._grad_adaptive:
                    if self._adaptive_rho < 1.0 and p.numel() >= group["sparse_min_numel"]:
                        masked_update = self._apply_kl_adaptive_mask(update)
                        was_masked = True
                elif self._global_step >= group["sparse_start_step"] and p.numel() >= group["sparse_min_numel"]:
                    masked_update = self._apply_sparse_mask(update, group, param=p)
                    was_masked = masked_update is not update

                # Muon/Moonlight 式 adjusted_lr: 2D hidden 权重按 rms_scale×√(max(A,B)) 放大有效步长.
                # ⚠️ 排除 embed/lm_head: 其一维=vocab(如151936)≫hidden, √max 会给 ~78× (MLP 仅~22×) → 爆.
                # 用维度比启发式: max/min > _RMS_EXCLUDE_RATIO 视为 embed/lm_head (vocab矩阵), 走 base lr.
                eff_lr = lr
                if self._rms_scale > 0.0 and p.dim() == 2:
                    A, B = int(p.shape[0]), int(p.shape[1])
                    if max(A, B) <= self._rms_max_fan and (max(A, B) / max(min(A, B), 1)) <= self._rms_exclude_ratio:
                        eff_lr = lr * self._rms_scale * math.sqrt(max(A, B))
                if group["weight_decay"] != 0 and group["decoupled_weight_decay"]:
                    p.mul_(1.0 - lr * group["weight_decay"])   # wd 用基础 lr (同 Muon)
                p.add_(masked_update, alpha=-eff_lr)

                # DTensor 安全: count_nonzero 无 sharding 策略 → 在本地分片算 (considered 也用本地, 保持比值一致)
                _mu = masked_update.to_local() if (_HAS_DTENSOR and isinstance(masked_update, DTensor)) else masked_update
                considered_elems += _mu.numel()
                nonzero_elems += _mu.count_nonzero().item()
                # per-layer sparsity (dual mode only, to bound overhead)
                if dual_on and was_masked and p.numel() >= 4096:
                    layer_sparsity = 1.0 - _mu.count_nonzero().item() / max(_mu.numel(), 1)
                    d_per_layer_sparsity.append(layer_sparsity)

                # diagnostics: only meaningful on masked tensors
                if was_masked and (diag["energy"] or diag["jaccard"] or diag["persistence"]):
                    if diag["energy"]:
                        upd_sq_sum += update.float().pow(2).sum().item()
                        kept_sq_sum += masked_update.float().pow(2).sum().item()
                        masked_elems_sparse += masked_update.numel()
                    if diag["jaccard"] or diag["persistence"]:
                        curr_mask = masked_update != 0
                        if diag["jaccard"]:
                            prev = state.get("prev_mask")
                            if prev is not None and prev.shape == curr_mask.shape:
                                # resume fix (2026-05-21): ckpt 中 prev_mask 可能被 FSDP 保存为 BFloat16,
                                # bitwise_and/or 不支持 BF16 → 显式 .bool() 转换
                                prev_b = prev.bool() if prev.dtype != torch.bool else prev
                                mask_inter_sum += (curr_mask & prev_b).sum().item()
                                mask_union_sum += (curr_mask | prev_b).sum().item()
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

        if self._unified_active:
            phase_idx = SPARSE_PHASE_UNIFIED
            sparse_target = self._unified_rho
        elif self._kl_adaptive:
            phase_idx = SPARSE_PHASE_KL_ADAPTIVE
            sparse_target = self._adaptive_rho
        elif self._grad_adaptive:
            phase_idx = SPARSE_PHASE_GRAD_ADAPTIVE
            sparse_target = self._adaptive_rho
        else:
            phase_idx = 0
            if self.param_groups:
                g0 = self.param_groups[0]
                phase_idx = sparse_schedule_phase_index(
                    self._global_step,
                    int(g0["sparse_start_step"]),
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
        if self._kl_adaptive:
            self._last_metrics["kl_ema"] = float(self._kl_ema or 0.0)
            self._last_metrics["adaptive_rho"] = float(self._adaptive_rho)
        if self._grad_adaptive:
            self._last_metrics["adaptive_rho"] = float(self._adaptive_rho)
            self._last_metrics["grad_norm_ema"] = float(self._grad_norm_ema or 0.0)
            self._last_metrics["grad_baseline"] = float(self._grad_baseline or 0.0)
        if self._rho_floor_scale > 0.0:
            rho_floor = max(self._rho_min, self._current_pos_ratio * self._rho_floor_scale)
            self._last_metrics["rho_floor"] = float(rho_floor)
            self._last_metrics["pos_ratio_for_floor"] = float(self._current_pos_ratio)
        if self._unified_active:
            self._last_metrics["unified_rho"] = float(self._unified_rho)
            self._last_metrics["unified_layer"] = float(self._unified_layer)

        # --- dual-channel metrics (paper extension) ---
        # always write dual_mode_flag for sanity check (even off mode)
        _mode_flag = {"off":0,"approx":1,"stat_only":2,"true":3}.get(self._dual_mode, 0)
        self._last_metrics["dual_mode_flag"] = float(_mode_flag)
        self._last_metrics["neg_proj_rank"] = float(self._neg_proj_rank)
        if self._neg_proj_rank == -2:          # 软滤波: 削掉的 neg 能量占比 (末层值, 监控用)
            self._last_metrics["dual_neg_filter_energy"] = float(getattr(self, "_neg_filter_rm_last", 0.0))
            self._last_metrics["neg_soft_ns_steps"] = float(self._neg_soft_ns_steps)
        elif self._neg_proj_rank == -3:        # PASP: 跨层 λ_noise 分布 (可证伪去硬编码; 末层值已废弃)
            _ll = list(getattr(self, "_pasp_lam_n_list", []))
            if _ll:
                _lls = sorted(_ll)
                _q = lambda f: _lls[min(len(_lls) - 1, int(f * (len(_lls) - 1)))]
                _mean = sum(_ll) / len(_ll)
                self._last_metrics["dual_pasp_lambda_noise"] = float(_mean)        # 向后兼容: =跨层均值
                self._last_metrics["dual_pasp_lambda_noise_mean"] = float(_mean)
                self._last_metrics["dual_pasp_lambda_noise_p10"] = float(_q(0.10))
                self._last_metrics["dual_pasp_lambda_noise_p50"] = float(_q(0.50))
                self._last_metrics["dual_pasp_lambda_noise_p90"] = float(_q(0.90))
                self._last_metrics["dual_pasp_lambda_noise_min"] = float(_lls[0])
                self._last_metrics["dual_pasp_lambda_noise_max"] = float(_lls[-1])
                self._last_metrics["dual_pasp_nlayers"] = float(len(_ll))
            _ef = list(getattr(self, "_pasp_ef_norm_list", []))
            if self._pasp_error_feedback:
                self._last_metrics["dual_pasp_ef_enabled"] = 1.0
                if _ef:
                    self._last_metrics["dual_pasp_ef_residual_norm_mean"] = float(sum(_ef) / len(_ef))
                    self._last_metrics["dual_pasp_ef_residual_norm_max"] = float(max(_ef))
            if self._pasp_auto_switch:
                self._last_metrics["dual_pasp_auto_switch"] = 1.0
                _st = list(self._pasp_stable.values())
                if _st:
                    self._last_metrics["dual_pasp_suppress_active_frac"] = float(
                        sum(1 for v in _st if v[1]) / len(_st))   # 已开抑制的层占比
                _ov = list(getattr(self, "_pasp_overlap_list", []))
                if _ov:
                    self._last_metrics["dual_pasp_subspace_overlap_mean"] = float(sum(_ov) / len(_ov))
            self._last_metrics["dual_pasp_lambda_align"] = float(self._pasp_lambda_align)
            self._last_metrics["dual_pasp_rank"] = float(self._pasp_rank)
            self._last_metrics["dual_pasp_refresh_steps"] = float(_PASP_REFRESH_ENV if _PASP_REFRESH_ENV > 0 else self._pasp_refresh_steps)
            _re = list(getattr(self, "_pasp_rank_eff_list", []))
            if _re:                                  # B: 自适应秩跨层统计 (验证 r 落在合理区间、随训练演化)
                _re.sort()
                self._last_metrics["dual_pasp_rank_eff_mean"] = float(sum(_re) / len(_re))
                self._last_metrics["dual_pasp_rank_eff_min"] = float(_re[0])
                self._last_metrics["dual_pasp_rank_eff_max"] = float(_re[-1])
                self._last_metrics["dual_pasp_rank_eff_p50"] = float(_re[len(_re) // 2])
        if self._dual_mode != "off" and d_dual_param_count > 0:
            import statistics as _st
            total_energy = d_pos_total + d_neg_total + 1e-12
            self._last_metrics["dual_sum_pos_sq_total"] = float(d_pos_total)
            self._last_metrics["dual_sum_neg_sq_total"] = float(d_neg_total)
            _pes = float(d_pos_total / total_energy)
            self._last_metrics["dual_pos_energy_share"] = _pes
            # v9: EMA 跟踪 pos_energy_share 演化 + 反转量
            if self._dual_pos_share_ema is None:
                self._dual_pos_share_ema = _pes
            else:
                self._dual_pos_share_ema = 0.9 * self._dual_pos_share_ema + 0.1 * _pes
            self._dual_pos_share_max = max(self._dual_pos_share_max, self._dual_pos_share_ema)
            self._last_metrics["dual_pos_share_ema"] = self._dual_pos_share_ema
            self._last_metrics["dual_pos_share_max_so_far"] = self._dual_pos_share_max
            self._last_metrics["dual_pos_share_decline"] = max(0.0, self._dual_pos_share_max - self._dual_pos_share_ema)
            # pos_share distribution
            if d_pos_share_samples:
                ss = sorted(d_pos_share_samples)
                n = len(ss)
                def _q(p): return ss[min(n-1, int(n*p))]
                self._last_metrics["dual_pos_share_mean"] = float(sum(ss)/n)
                self._last_metrics["dual_pos_share_p25"] = float(_q(0.25))
                self._last_metrics["dual_pos_share_p50"] = float(_q(0.50))
                self._last_metrics["dual_pos_share_p75"] = float(_q(0.75))
                self._last_metrics["dual_pos_share_p90"] = float(_q(0.90))
            # cross-term: 单通道 sum_sq 累加的额外部分
            grad_total = d_gpos_sq + d_gneg_sq + 1e-12
            self._last_metrics["dual_cross_term_energy"] = float(d_cross)
            self._last_metrics["dual_cross_term_ratio"] = float(d_cross / grad_total)
            # g_pos/g_neg norm + cosine
            self._last_metrics["dual_g_pos_norm"] = float(d_gpos_sq ** 0.5)
            self._last_metrics["dual_g_neg_norm"] = float(d_gneg_sq ** 0.5)
            _denom = (d_gpos_sq * d_gneg_sq) ** 0.5 + 1e-12
            self._last_metrics["dual_g_cos_sim"] = float(d_cross / _denom)
            # precond ratio (dual vs single equivalent)
            if d_precond_sgl_sq > 1e-12:
                self._last_metrics["dual_precond_ratio"] = float((d_precond_dual_sq / d_precond_sgl_sq) ** 0.5)
            # numerical sanity
            self._last_metrics["dual_nan_count"] = float(d_nan_count)
            self._last_metrics["dual_inf_count"] = float(d_inf_count)
            self._last_metrics["dual_precond_min"] = float(d_precond_min)
            self._last_metrics["dual_precond_max"] = float(d_precond_max)
            # per-layer sparsity std
            if len(d_per_layer_sparsity) >= 2:
                self._last_metrics["dual_per_layer_sparsity_std"] = float(_st.pstdev(d_per_layer_sparsity))
                self._last_metrics["dual_per_layer_sparsity_mean"] = float(sum(d_per_layer_sparsity)/len(d_per_layer_sparsity))
            # mask diagnostics
            md = self._dual_mask_diag
            if md["pos_overlap_den"] > 0:
                self._last_metrics["dual_mask_pos_overlap"] = float(md["pos_overlap_num"] / md["pos_overlap_den"])
            u_pd = md["update_pos_dom_sq"]; u_nd = md["update_neg_dom_sq"]
            if (u_pd + u_nd) > 1e-12:
                self._last_metrics["dual_update_pos_dom_norm"] = float(u_pd ** 0.5)
                self._last_metrics["dual_update_neg_dom_norm"] = float(u_nd ** 0.5)
                self._last_metrics["dual_update_pos_dom_share"] = float(u_pd / (u_pd + u_nd))
            if md["rescale_factors"]:
                rf = md["rescale_factors"]
                self._last_metrics["dual_rescale_factor_mean"] = float(sum(rf)/len(rf))
                self._last_metrics["dual_rescale_factor_max"] = float(max(rf))
                self._last_metrics["dual_E_drop_total"] = float(md["E_drop_sum"])
                self._last_metrics["dual_E_keep_total"] = float(md["E_keep_sum"])
            # patience & mask mode
            self._last_metrics["dual_mask_mode_int"] = float(1 if self._dual_mask_mode == "neg_energy" else 0)
            self._last_metrics["dual_p_down"] = float(self._dual_p_down)
            self._last_metrics["dual_p_up"] = float(self._dual_p_up)
            self._last_metrics["dual_ac_ema"] = float(self._dual_ac_ema or 0.0)
            # 坐标级 pos/neg 重合分析 (聚合版布尔交集, tau=95 分位)
            if d_ov_total > 0:
                self._last_metrics["dual_ov_pos_active_frac"] = float(d_ov_pos_active / d_ov_total)
                self._last_metrics["dual_ov_neg_active_frac"] = float(d_ov_neg_active / d_ov_total)
                self._last_metrics["dual_ov_overlap_frac"] = float(d_ov_overlap / d_ov_total)
                # 重合占 pos 活跃 / neg 活跃的比例 (Jaccard-like, 重合密度)
                if d_ov_pos_active > 0:
                    self._last_metrics["dual_ov_overlap_over_pos"] = float(d_ov_overlap / d_ov_pos_active)
                if d_ov_neg_active > 0:
                    self._last_metrics["dual_ov_overlap_over_neg"] = float(d_ov_overlap / d_ov_neg_active)
                if d_ov_overlap > 0:
                    # 重合区: 同号占比 (pos/neg 同向) + |g_pos|>|g_neg| 占比 (重合区谁能量大)
                    self._last_metrics["dual_ov_overlap_samedir_frac"] = float(d_ov_overlap_samedir / d_ov_overlap)
                    self._last_metrics["dual_ov_overlap_pos_gt_neg_frac"] = float(d_ov_overlap_pos_gt_neg / d_ov_overlap)
        # cleanup side-channel
        self._dual_mask_diag = None

        if diag["energy"] and masked_elems_sparse > 0 and upd_sq_sum > 0.0:
            self._last_metrics["mask_energy_kept"] = float(kept_sq_sum / upd_sq_sum)
        if diag["jaccard"] and mask_union_sum > 0:
            self._last_metrics["mask_jaccard_prev"] = float(mask_inter_sum / mask_union_sum)
        if diag["persistence"] and persist_count > 0:
            self._last_metrics["mask_persistence_ema"] = float(persist_sum / persist_count)
        return loss

    def get_last_metrics(self) -> dict[str, float]:
        return dict(self._last_metrics)

    def update_pos_ratio(self, pos_ratio: float) -> None:
        """Feed current batch positive-sample ratio for dynamic rho floor (Layer 3).
        Call before step(). No-op if rho_floor_scale == 0.
        """
        self._current_pos_ratio = float(pos_ratio)

    def set_unified_rho(self, rho: float, layer: int) -> None:
        """Receive target ρ and layer from the unified grad_norm trigger in dp_actor."""
        self._unified_rho = max(self._rho_min, min(1.0, float(rho)))
        self._unified_layer = int(layer)
        self._unified_active = True

    # ------------------------------------------------------------------
    # Dual-channel hooks (paper extension): feed g_pos/g_neg from trainer,
    # patience-switch mask mode by all_correct signal.
    # ------------------------------------------------------------------
    def set_dual_grads(self, grad_pos_dict: dict, grad_neg_dict: dict) -> None:
        """Trainer calls this between backward() and step() when dual_mode='true'.

        grad_pos_dict / grad_neg_dict: id(param) -> Tensor (same shape as param.grad)
        """
        self._dual_grad_pos = grad_pos_dict
        self._dual_grad_neg = grad_neg_dict

    def on_all_correct(self, ac: float, ema_beta: float = 0.9, delta: float = 0.005) -> None:
        """Patience-driven mask-mode switch. Trainer calls each step.

        ac: current all_correct fraction (in [0,1])
        Logic: ac < ema-δ K_down times → switch to 'neg_energy';
               ac > ema+δ K_up times → switch back to 'magnitude'.
        """
        if self._dual_score != "neg_energy":
            return  # only meaningful when score=neg_energy
        if self._dual_ac_ema is None:
            self._dual_ac_ema = ac
            return
        ref = self._dual_ac_ema
        if ac < ref - delta:
            self._dual_p_down += 1
            self._dual_p_up = 0
        elif ac > ref + delta:
            self._dual_p_up += 1
            self._dual_p_down = 0
        else:
            # within dead-zone: no change
            pass
        if self._dual_p_down >= self._dual_patience_down:
            self._dual_mask_mode = "neg_energy"
        if self._dual_p_up >= self._dual_patience_up:
            self._dual_mask_mode = "magnitude"
            self._dual_p_down = 0
        # update slow EMA last (uses prev ema for switch logic)
        self._dual_ac_ema = ema_beta * self._dual_ac_ema + (1.0 - ema_beta) * ac

    def update_grad_norm_signal(self, grad_norm: float) -> None:
        """Feed grad_norm from previous step for ρ*-driven mode. Call before step()."""
        if not self._grad_adaptive:
            return
        beta = self._grad_ema_beta
        if self._grad_norm_ema is None:
            self._grad_norm_ema = grad_norm
        else:
            self._grad_norm_ema = beta * self._grad_norm_ema + (1.0 - beta) * grad_norm
        if self._global_step < self._grad_warmup_steps:
            self._grad_baseline = self._grad_norm_ema
            self._adaptive_rho = self._rho_max
        elif self._grad_baseline is not None and self._grad_norm_ema > 0:
            raw_rho = self._grad_baseline / self._grad_norm_ema
            self._adaptive_rho = max(self._rho_min, min(self._rho_max, raw_rho))

    def update_kl_signal(self, kl_value: float) -> None:
        """Feed current KL divergence for KASO mode. Call before step()."""
        if not self._kl_adaptive:
            return
        if self._kl_ema is None:
            self._kl_ema = kl_value
        else:
            self._kl_ema = self._kl_ema_beta * self._kl_ema + (1.0 - self._kl_ema_beta) * kl_value
        if self._kl_ema < self._kl_target_low:
            self._adaptive_rho = min(self._adaptive_rho + self._rho_delta, self._rho_max)
        elif self._kl_ema > self._kl_target_high:
            self._adaptive_rho = max(self._adaptive_rho - self._rho_delta, self._rho_min)

    def _prepare_step_grad(self, p: Tensor, grad: Tensor, group: dict) -> Tensor:
        beta = group["warmup_momentum_beta"]
        warmup_steps = group["warmup_momentum_steps"]
        state = self.state[p]
        if beta <= 0.0 or warmup_steps <= 0 or self._global_step > warmup_steps:
            state["_prepared_grad"] = grad
            return grad

        ema_grad = state["ema_grad"]
        ema_grad.mul_(beta).add_(grad, alpha=1.0 - beta)
        state["_prepared_grad"] = ema_grad.clone()
        return state["_prepared_grad"]

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

    def _current_sparse_target(self, group: dict) -> float:
        if self._global_step < group["sparse_start_step"]:
            return 1.0

        if group["sparse_mode"] == "topk":
            final_ratio = group["sparse_topk_ratio"]
            if group["sparse_ramp_steps"] <= 0:
                target = final_ratio
            else:
                progress = min(
                    1.0,
                    float(self._global_step - group["sparse_start_step"]) / float(group["sparse_ramp_steps"]),
                )
                target = 1.0 - progress * (1.0 - final_ratio)
        elif group["sparse_threshold_scale"] <= 0.0:
            return 1.0
        elif group["sparse_ramp_steps"] <= 0:
            target = 0.0
        else:
            progress = min(
                1.0,
                float(self._global_step - group["sparse_start_step"]) / float(group["sparse_ramp_steps"]),
            )
            target = 1.0 - progress

        # Layer 2: advantage-aware boost — when p+ is high, keep more coordinates
        # s_j = |Δθ_j| * (1 + alpha * p+)  ≡  effective rho *= (1 + alpha * p+)
        if self._acsa_alpha > 0.0:
            boost = 1.0 + self._acsa_alpha * self._current_pos_ratio
            target = min(1.0, target * boost)

        # Layer 3: dynamic rho floor — when signal quality is high (pos_ratio > 0),
        # enforce a minimum keep-ratio to prevent over-masking good gradient coordinates
        if self._rho_floor_scale > 0.0:
            rho_floor = max(self._rho_min, self._current_pos_ratio * self._rho_floor_scale)
            target = max(target, rho_floor)

        return target

    def _apply_kl_adaptive_mask(self, update: Tensor) -> Tensor:
        """Top-k mask driven by KL-adaptive ρ or unified ρ."""
        keep_ratio = self._unified_rho if self._unified_active else self._adaptive_rho
        if keep_ratio >= 1.0:
            return update
        flat = update.abs().reshape(-1)
        k = max(1, int(flat.numel() * keep_ratio))
        _, indices = torch.topk(flat, k=k, sorted=False)
        mask = torch.zeros_like(flat, dtype=torch.bool)
        mask.scatter_(0, indices, True)
        return update * mask.reshape_as(update)

    def _apply_sparse_mask(self, update: Tensor, group: dict, param: Optional[Tensor] = None) -> Tensor:
        # DTensor 安全 (FSDP2): topk/scatter_ 在 DTensor 上无 sharding 策略 → min() empty 崩溃。
        # FSDP1(原配方)对每 rank 本地展平分片做 topk(per-shard); 故 FSDP2 下也在本地分片上算 mask,
        # 与原配方语义一致, 再 from_local 包回 DTensor (p.add_ 需 masked_update 仍是 DTensor)。
        if _HAS_DTENSOR and isinstance(update, DTensor):
            mesh, plc = update.device_mesh, update.placements
            out_local = self._apply_sparse_mask_dense(update.to_local(), group, param=param)
            return DTensor.from_local(out_local, mesh, plc, run_check=False)
        return self._apply_sparse_mask_dense(update, group, param=param)

    def _apply_sparse_mask_dense(self, update: Tensor, group: dict, param: Optional[Tensor] = None) -> Tensor:
        mode = group["sparse_mode"]
        if mode == "topk":
            keep_ratio = self._current_sparse_target(group)
            if keep_ratio >= 1.0:
                return update
            # ----- choose score function -----
            # default: magnitude of update (existing behavior)
            # dual + neg_energy: when mask mode 切到 neg_energy, score = sum_pos² / (sum_pos² + sum_neg²)
            #                    i.e. keep pos-dominant coordinates, drop neg-dominant
            use_neg_energy = (
                self._dual_mode != "off"
                and self._dual_score == "neg_energy"
                and self._dual_mask_mode == "neg_energy"
                and param is not None
            )
            if use_neg_energy:
                st = self.state.get(param, {})
                sp = st.get("sum_pos_sq"); sn = st.get("sum_neg_sq")
                if _HAS_DTENSOR and isinstance(sp, DTensor): sp = sp.to_local()
                if _HAS_DTENSOR and isinstance(sn, DTensor): sn = sn.to_local()
                if sp is not None and sn is not None:
                    score_flat = (sp / (sp + sn + 1e-12)).reshape(-1)
                else:
                    score_flat = update.abs().reshape(-1)
            else:
                score_flat = update.abs().reshape(-1)
            k = max(1, int(score_flat.numel() * keep_ratio))
            _, indices = torch.topk(score_flat, k=k, sorted=False)
            mask = torch.zeros_like(score_flat, dtype=torch.bool)
            mask.scatter_(0, indices, True)
            mask = mask.reshape_as(update)
            sparse_update = update * mask
            # ----- per-mask dual diagnostics (side-channel) -----
            if self._dual_mode != "off" and param is not None:
                st_mask_diag = getattr(self, "_dual_mask_diag", None)
                if st_mask_diag is not None:
                    st = self.state.get(param, {})
                    sp = st.get("sum_pos_sq"); sn = st.get("sum_neg_sq")
                    if _HAS_DTENSOR and isinstance(sp, DTensor): sp = sp.to_local()
                    if _HAS_DTENSOR and isinstance(sn, DTensor): sn = sn.to_local()
                    if sp is not None and sn is not None:
                        with torch.no_grad():
                            ps_flat = (sp / (sp + sn + 1e-12)).reshape(-1)
                            mask_flat = mask.reshape(-1).bool()
                            pos_dom_flat = (ps_flat > 0.5)
                            # mask ∩ pos-dominated coordinates / |mask|
                            inter = int((mask_flat & pos_dom_flat).sum().item())
                            denom = int(mask_flat.sum().item())
                            st_mask_diag["pos_overlap_num"] += inter
                            st_mask_diag["pos_overlap_den"] += denom
                            # update split by pos-dominated vs neg-dominated coords
                            u_pos_sq = (update.reshape(-1)[pos_dom_flat]).float().pow(2).sum().item()
                            u_neg_sq = (update.reshape(-1)[~pos_dom_flat]).float().pow(2).sum().item()
                            st_mask_diag["update_pos_dom_sq"] += u_pos_sq
                            st_mask_diag["update_neg_dom_sq"] += u_neg_sq
            # ----- optional energy-preserving rescale -----
            if self._dual_rescale and self._dual_mode != "off":
                u_sq = update.float().pow(2).sum()
                k_sq = sparse_update.float().pow(2).sum().clamp_min(1e-12)
                scale = (u_sq / k_sq).sqrt()
                sparse_update = sparse_update * scale
                # rescale diagnostics
                if self._dual_mode != "off":
                    st_mask_diag = getattr(self, "_dual_mask_diag", None)
                    if st_mask_diag is not None:
                        st_mask_diag["rescale_factors"].append(float(scale.item()))
                        st_mask_diag["E_drop_sum"] += float((u_sq - k_sq).item())
                        st_mask_diag["E_keep_sum"] += float(k_sq.item())
            return sparse_update

        target_scale = group["sparse_threshold_scale"]
        if target_scale <= 0.0:
            return update
        if group["sparse_ramp_steps"] > 0:
            progress = min(
                1.0,
                float(self._global_step - group["sparse_start_step"]) / float(group["sparse_ramp_steps"]),
            )
            target_scale = target_scale * progress
        threshold = update.abs().mean() * target_scale
        return update * (update.abs() >= threshold)

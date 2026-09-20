"""接收端**同步（resync）**：几何攻击后把密钥频点位置找回来。

> ⚠️ **本机制已被实测否定（ITERATION_LOG §22），默认关闭、请勿启用。**
> `scripts/diag_mag_sync.py` 的验证结果：定位命中率 0/6（连 clean 都定位不到
> 恒等变换），并且开轮廓后 clean 从 1.000 掉到 0.562、PSNR 还要多付 0.77 dB。
> 原因（定量）：本载波在 30 dB 工作点上的**单频点相干 SNR 只有 ≈0.25**
> （由"每比特 32 个观测才凑到 1.44σ 偏转"反推），整环 320 个频点相干叠加也只有
> ≈4.5σ，而 3087 个假设的盲搜索噪声底就有 ≈4.0σ —— 信噪比根本不够支撑盲搜索；
> 而**幅度**轮廓更没有相干增益，直接淹没在封面自身的频谱起伏里。
> 保留此文件仅为记录与将来复现（例如换成"相干模板 + 缩小搜索空间"的思路），
> 不要在正式实验里打开 `mag_profile` / `sync`。
>
> 若将来要重试，正确的方向是：把搜索空间压到 1 维（只用封面无关的量，
> 例如从 VAE 潜变量的自相关估计缩放），或者用**独立且能量远高于数据载波**的
> 同步图案（代价是 PSNR），而不是复用数据载波的幅度轮廓。

原始设计说明（已证伪，仅留档）
--------------------------------------------------------------------------------
为什么当初认为需要（ITERATION_LOG §20.4/§21 的实测）：
v3 鲁棒性全表里唯一的短板是重采样类几何变换
（translate2px 0.434 / rotate15 0.490 / cropresize0.8 0.539 / zoom1.1 0.707），
而 JPEG/噪声/模糊/裁剪/光度类都在 0.96-0.99。原因是载波用**密钥相位**做相干叠加，
图像一旦被旋转/平移/缩放，频谱被"重排 + 加相位斜坡"，接收端仍在原频点读 → 相干叠加失配。

三种几何变换在**频谱域是解析可逆**的（不需要重过 VAE/UNet）：
  - 平移 d  ：F(k) -> F(k)·e^{-j2π k·d/N}   ⇒ 给已取到的复数值乘反向相位斜坡即可；
  - 旋转 θ  ：F_rot(k) = F_orig(R_{-θ}k)     ⇒ 在旋转后的坐标上重新取点；
  - 缩放 σ  ：频点半径 r -> r/σ              ⇒ 在 r/σ 处取点。
（这一部分数学仍然正确，问题出在"盲搜索缺参考信号"，见上。）
"""

from __future__ import annotations

import math

import torch

# 默认搜索网格：覆盖评测里用到的几何强度（rotate ±15°、translate ±3px、zoom 0.85-1.15）
DEFAULT_THETAS = (-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0)
DEFAULT_SIGMAS = (0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15)
DEFAULT_SHIFTS = (-3, -2, -1, 0, 1, 2, 3)


def _gather(F_: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
    """F_ (C,H,W) -> (C,K)，越界填 0。"""
    C, H, W = F_.shape
    valid = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
    out = F_[:, rows.clamp(0, H - 1), cols.clamp(0, W - 1)]
    return out * valid[None, :].to(out.dtype)


def warp_bins(bins: torch.Tensor, res: int, theta_deg: float, sigma: float
              ) -> tuple[torch.Tensor, torch.Tensor]:
    """把原始频点坐标变换到"图像被旋转 θ、缩放 σ 之后"应在的位置。

    fftshift 坐标（中心 res//2 - 0.5）。旋转用 R_{+θ}（与图像被旋转同向）。
    """
    c = (res - 1) / 2.0
    k = bins.float() - c
    t = math.radians(theta_deg)
    ct, st = math.cos(t), math.sin(t)
    x = (ct * k[:, 1] - st * k[:, 0]) / max(sigma, 1e-6)
    y = (st * k[:, 1] + ct * k[:, 0]) / max(sigma, 1e-6)
    return torch.round(y + c).long(), torch.round(x + c).long()


def shift_ramp(rows: torch.Tensor, cols: torch.Tensor, res: int,
               dx: int, dy: int, device, dtype=torch.complex64) -> torch.Tensor:
    """补偿图像平移 (dx,dy) 像素所需的相位斜坡 e^{+j2π(k·d)/N}。"""
    c = (res - 1) / 2.0
    ph = 2.0 * math.pi * ((rows.float() - c) * dx + (cols.float() - c) * dy) / res
    return torch.exp(1j * ph.to(device)).to(dtype)


def _profile_scores(F_: torch.Tensor, params: dict, res: int,
                    bins: torch.Tensor, mag_ref: torch.Tensor,
                    thetas, sigmas, shifts, device) -> tuple[float, tuple]:
    """用幅度轮廓相关做盲定位。返回 (best_score, (theta,sigma,dx,dy))。

    评分 = corr(|F_rec at 假设位置|, m_k)。相位不参与 → 对平移天然不敏感，
    所以平移只需在**最后**用相位斜坡精修一次（这里对 dx=dy=0 评分即可）。
    """
    best_score, best = -1e9, (0.0, 1.0, 0, 0)
    mag_ref = mag_ref / mag_ref.mean().clamp(min=1e-8)
    for th in thetas:
        for sg in sigmas:
            rows, cols = warp_bins(bins, res, th, sg)
            mag = _gather(F_, rows, cols).abs().mean(0)          # 通道平均
            if float(mag.max()) <= 0:
                continue
            m = mag / mag.mean().clamp(min=1e-8)
            sc = float((m * mag_ref).mean())
            if sc > best_score:
                best_score, best = sc, (th, sg, 0, 0)
    return best_score, best


def refine_shift(F_: torch.Tensor, params: dict, res: int, bins: torch.Tensor,
                 theta: float, sigma: float, shifts, device) -> tuple[int, int, float]:
    """固定 (θ,σ) 后，用**校验位偏转**（接收端已知的 m 个比特）精修平移。

    幅度相关对平移不敏感，而相位斜坡只影响同相分量；16 个校验位足够在
    |shifts| ≤ 3 的 7×7 小网格里定出平移（噪声底只有 √(2 ln 49) ≈ 2.8σ）。
    """
    rows, cols = warp_bins(bins, res, theta, sigma)
    vals = _gather(F_, rows, cols) * torch.exp(
        -1j * params["phases"].to(device))[None, :]
    best, best_sc = (0, 0), -1e9
    for dx in shifts:
        for dy in shifts:
            ramp = shift_ramp(rows, cols, res, -dx, -dy, device, vals.dtype)
            re = (vals * ramp[None, :]).real.mean(0)
            sc = float(re.abs().mean())
            if sc > best_sc:
                best_sc, best = sc, (dx, dy)
    return best[0], best[1], best_sc


def estimate_alignment(F_: torch.Tensor, params: dict, res: int,
                       thetas=DEFAULT_THETAS, sigmas=DEFAULT_SIGMAS,
                       shifts=DEFAULT_SHIFTS, device=None,
                       refine_shift_search: bool = True) -> dict:
    """盲定位：返回 {theta, sigma, dx, dy, score}。F_ 为 (C,H,W) 的 fftshift 频谱。"""
    device = device or F_.device
    bins = params["bins"].to(device)
    _, best = _profile_scores(F_, params, res, bins, params["base_mag"].to(device),
                              thetas, sigmas, shifts, device)
    theta, sigma, _, _ = best
    score = float(params["base_mag"].mean())  # 占位，真正分数见 _profile_scores
    dx = dy = 0
    if refine_shift_search:
        dx, dy, _ = refine_shift(F_, params, res, bins, theta, sigma, shifts, device)
    return {"theta": float(theta), "sigma": float(sigma), "dx": int(dx),
            "dy": int(dy), "search_space": len(thetas) * len(sigmas) * len(shifts) ** 2}


def gather_aligned(F_: torch.Tensor, params: dict, res: int, est: dict,
                   device=None) -> torch.Tensor:
    """按估计的对齐取复数频点并去旋转（返回 (K,) 的同相分量，供 mf/解码器使用）。"""
    device = device or F_.device
    bins = params["bins"].to(device)
    rows, cols = warp_bins(bins, res, est["theta"], est["sigma"])
    vals = _gather(F_, rows, cols) * torch.exp(
        -1j * params["phases"].to(device))[None, :]
    ramp = shift_ramp(rows, cols, res, -int(est["dx"]), -int(est["dy"]),
                      device, vals.dtype)
    vals = vals * ramp[None, :]
    return vals.real.mean(0), vals.imag.mean(0), rows, cols

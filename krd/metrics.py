"""指标：PSNR / SSIM / 比特准确率。输入均为 [-1,1] 张量。"""

import math

import torch
import torch.nn.functional as F


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = ((a - b) ** 2).mean().item()
    if mse <= 1e-12:
        return float("inf")
    # [-1,1] 动态范围 2，等价于 [0,255] 的 255 峰值计算
    return 10.0 * math.log10(4.0 / mse)


_GAUSS_WIN = 11
_GAUSS_SIGMA = 1.5


def _gaussian_window(ch: int, device) -> torch.Tensor:
    coords = torch.arange(_GAUSS_WIN, dtype=torch.float32, device=device) - _GAUSS_WIN // 2
    g = torch.exp(-coords ** 2 / (2 * _GAUSS_SIGMA ** 2))
    g = (g / g.sum())
    w = g[:, None] @ g[None, :]
    return w.expand(ch, 1, _GAUSS_WIN, _GAUSS_WIN).contiguous()


def ssim(a: torch.Tensor, b: torch.Tensor) -> float:
    """逐通道 SSIM 后取均值。动态范围 L=2（[-1,1]）。"""
    ch = a.shape[1]
    win = _gaussian_window(ch, a.device)
    pad = _GAUSS_WIN // 2
    mu_a = F.conv2d(a, win, padding=pad, groups=ch)
    mu_b = F.conv2d(b, win, padding=pad, groups=ch)
    var_a = F.conv2d(a * a, win, padding=pad, groups=ch) - mu_a ** 2
    var_b = F.conv2d(b * b, win, padding=pad, groups=ch) - mu_b ** 2
    cov = F.conv2d(a * b, win, padding=pad, groups=ch) - mu_a * mu_b
    c1, c2 = (0.01 * 2.0) ** 2, (0.03 * 2.0) ** 2
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))
    return s.mean().item()


def bit_accuracy(logits: torch.Tensor, bits: torch.Tensor) -> float:
    """logits (B,L), bits (B,L)∈{0,1}。"""
    pred = (logits > 0.0).float()
    return (pred == (bits > 0.5).float()).float().mean().item()

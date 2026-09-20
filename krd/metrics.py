"""指标：PSNR / SSIM / 比特准确率。输入均为 [-1,1] 张量。"""

import math

import torch
import torch.nn.functional as F


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    """**批内全局 MSE** 定义的 PSNR（历史口径，保留以保证旧结果可复现）。

    ⚠️ 口径警告：全局 MSE 是"先对所有图像求均方误差、再取 log"，
    而 MSE 对重尾误差极其敏感——一张坏图就能吃掉好几 dB，且**结果随批大小 n 变化**
    （n 越大，坏图被摊薄，PSNR 越高）。实测同一配置 n=4 与 n=24 可差 1.3 dB。
    因此跨运行比较 PSNR 必须固定 n；论文报数建议改用 `psnr_mean`（逐图 PSNR 再平均，
    这是文献通行口径，通常比全局口径高 0.5-1.5 dB）。
    """
    mse = ((a - b) ** 2).mean().item()
    if mse <= 1e-12:
        return float("inf")
    # [-1,1] 动态范围 2，等价于 [0,255] 的 255 峰值计算
    return 10.0 * math.log10(4.0 / mse)


def psnr_mean(a: torch.Tensor, b: torch.Tensor) -> float:
    """逐图 PSNR 再取平均（文献通行口径，与 n 的关联弱得多）。

    a, b: (B,C,H,W)。B=1 时与 `psnr` 相同。
    """
    if a.dim() == 3:
        return psnr(a, b)
    per = []
    for i in range(a.shape[0]):
        mse = ((a[i] - b[i]) ** 2).mean().item()
        per.append(float("inf") if mse <= 1e-12 else 10.0 * math.log10(4.0 / mse))
    finite = [p for p in per if p != float("inf")]
    return sum(finite) / len(finite) if finite else float("inf")


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

"""感知指标：LPIPS（可选依赖）+ 简单的局部对比度度量。

设计原则：
  - LPIPS 是"不可见性"的标准指标，但 `lpips` / `torchmetrics` 是可选依赖，
    没装时不能让整条评测流水线崩掉；
  - 因此 LPIPS 走惰性加载：首次使用时尝试构建（优先 `lpips` 包，
    退回 `torchmetrics.image.lpip`），失败则返回 None 并在结果里标注 unavailable；
  - 输入统一为 [-1,1] 的 (B,C,H,W) 张量，与 krd.metrics 保持一致。
"""

import torch
import torch.nn.functional as F

_LPIPS_FN = None
_LPIPS_TRIED = False
_LPIPS_BACKEND = None


def _build_lpips(device):
    """尝试构建 LPIPS 度量，返回 (fn, backend)；不可用返回 (None, None)。"""
    try:  # 首选官方 lpips 包（AlexNet 特征，与主流论文一致）
        import lpips
        net = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        for p in net.parameters():
            p.requires_grad_(False)

        def _fn(a, b):
            return net(a, b, normalize=True).mean().item()

        return _fn, "lpips(alex)"
    except Exception:
        pass
    try:  # 退回 torchmetrics
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)
        metric.eval()

        def _fn(a, b):
            return metric(a, b).item()

        return _fn, "torchmetrics(lpips-alex)"
    except Exception:
        return None, None


def lpips_available() -> bool:
    global _LPIPS_FN, _LPIPS_TRIED, _LPIPS_BACKEND
    if not _LPIPS_TRIED:
        _LPIPS_FN, _LPIPS_BACKEND = _build_lpips(torch.device("cpu"))
        _LPIPS_TRIED = True
    return _LPIPS_FN is not None


def lpips(a: torch.Tensor, b: torch.Tensor) -> float | None:
    """LPIPS(a, b)，输入 [-1,1] (B,C,H,W)；依赖缺失时返回 None。"""
    global _LPIPS_FN, _LPIPS_TRIED, _LPIPS_BACKEND
    if not _LPIPS_TRIED:
        _LPIPS_FN, _LPIPS_BACKEND = _build_lpips(a.device)
        _LPIPS_TRIED = True
    if _LPIPS_FN is None:
        return None
    with torch.no_grad():
        return float(_LPIPS_FN(a.float(), b.float()))


def lpips_backend() -> str | None:
    return _LPIPS_BACKEND


def local_contrast(x: torch.Tensor, win: int = 3) -> torch.Tensor:
    """局部标准差（逐通道），用于判断注入是否造成异常的高频纹理。"""
    c = x.shape[1]
    k = torch.ones(c, 1, win, win, device=x.device, dtype=x.dtype) / (win * win)
    mu = F.conv2d(x, k, padding=win // 2, groups=c)
    ex2 = F.conv2d(x * x, k, padding=win // 2, groups=c)
    return (ex2 - mu * mu).clamp(min=0).sqrt()

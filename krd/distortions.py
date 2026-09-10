"""失真仿真：可微 JPEG（直通估计器）+ 噪声/模糊/缩放/亮度/几何攻击。

训练复原解码器时以随机失真模拟信道，是鲁棒性的来源；
评测时按 (name, param) 复现标准攻击（STANDARD_ATTACKS）。

几何攻击说明：`crop` 是**真裁剪**（裁边 + 边缘回填），不是 torch.roll 循环平移；
另有 cropresize / rotate / translate / zoom。早期版本用循环平移冒充裁剪，
会系统性高估几何鲁棒性，已在本文档对应位置标注。
"""

import math

import torch
import torch.nn.functional as F

# 标准 JPEG 量化表（亮度 / 色度，之字形前的 8x8 排列）
_LUM = [
    [16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99],
]
_CHR = [
    [17, 18, 24, 47, 99, 99, 99, 99], [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99], [47, 66, 99, 99, 99, 99, 99, 99],
] + [[99] * 8] * 4


def _quant_table(base, quality: int, device) -> torch.Tensor:
    base = torch.tensor(base, dtype=torch.float32, device=device)
    q = min(max(int(quality), 1), 100)
    scale = 5000.0 / q if q < 50 else 200.0 - 2.0 * q
    return ((base * scale + 50.0) / 100.0).floor().clamp(1.0, 255.0)


def _dct_basis(n: int = 8, device="cpu") -> torch.Tensor:
    i = torch.arange(n, dtype=torch.float32, device=device)[:, None]  # 空间(行)
    k = torch.arange(n, dtype=torch.float32, device=device)[None, :]  # 频率(列)
    D = torch.cos(math.pi * (2 * i + 1) * k / (2 * n))
    c = torch.full((n,), math.sqrt(2.0 / n), device=device)
    c[0] = math.sqrt(1.0 / n)
    D = D * c[None, :]  # 归一化系数随频率 k（列）缩放
    return D  # 正变换 X = D^T @ block @ D；逆变换 block = D @ X @ D^T


def _round_st(x: torch.Tensor) -> torch.Tensor:
    """直通估计器的取整：前向取整，梯度直通。"""
    return x + (torch.round(x) - x).detach()


def _blockify(x: torch.Tensor, s: int = 8) -> torch.Tensor:
    b, c, h, w = x.shape
    return x.view(b * c, h // s, s, w // s, s).permute(0, 1, 3, 2, 4).reshape(-1, s, s)


def _unblockify(x: torch.Tensor, b: int, c: int, h: int, w: int, s: int = 8) -> torch.Tensor:
    x = x.view(b * c, h // s, w // s, s, s).permute(0, 1, 3, 2, 4).reshape(b * c, h, w)
    return x.view(b, c, h, w)


def rgb_to_ycbcr(x255: torch.Tensor):
    r, g, b = x255[:, 0], x255[:, 1], x255[:, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128.0 + (-0.168736 * r - 0.331264 * g + 0.5 * b)
    cr = 128.0 + (0.5 * r - 0.418688 * g - 0.081312 * b)
    return y, cb, cr


def ycbcr_to_rgb(y, cb, cr):
    cb, cr = cb - 128.0, cr - 128.0
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    return torch.stack([r, g, b], dim=1)


def diff_jpeg(x: torch.Tensor, quality: int = 50) -> torch.Tensor:
    """[-1,1] RGB -> JPEG 压缩模拟 -> [-1,1]。可微（取整直通），不采样色度。"""
    device = x.device
    D = _dct_basis(8, device)
    x255 = (x + 1.0) * 127.5
    y, cb, cr = rgb_to_ycbcr(x255)
    b, _, h, w = x.shape
    qt_y = _quant_table(_LUM, quality, device)
    qt_c = _quant_table(_CHR, quality, device)

    def _proc(ch: torch.Tensor, qt: torch.Tensor) -> torch.Tensor:
        blk = _blockify(ch.unsqueeze(1), 8)          # (N, 8, 8)
        freq = D.t() @ blk @ D
        quant = _round_st(freq / qt) * qt
        return D @ quant @ D.t()

    y2, cb2, cr2 = _proc(y, qt_y), _proc(cb, qt_c), _proc(cr, qt_c)
    y2 = _unblockify(y2, b, 1, h, w)[:, 0]
    cb2 = _unblockify(cb2, b, 1, h, w)[:, 0]
    cr2 = _unblockify(cr2, b, 1, h, w)[:, 0]
    out = ycbcr_to_rgb(y2, cb2, cr2).clamp(0.0, 255.0)
    return out / 127.5 - 1.0


# ---------------- 经典空域/几何失真 ----------------

def gauss_noise(x: torch.Tensor, sigma: float = 0.05) -> torch.Tensor:
    return (x + sigma * torch.randn_like(x)).clamp(-1, 1)


def gauss_blur(x: torch.Tensor, ksize: int = 3, sigma: float = 1.0) -> torch.Tensor:
    coords = torch.arange(ksize, dtype=torch.float32, device=x.device) - ksize // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g = (g / g.sum()).view(1, 1, -1)
    c = x.shape[1]
    kx = g.reshape(1, 1, 1, ksize).expand(c, 1, 1, ksize)
    ky = g.reshape(1, 1, ksize, 1).expand(c, 1, ksize, 1)
    x = F.conv2d(F.pad(x, (ksize // 2, ksize // 2, 0, 0), mode="replicate"), kx, groups=c)
    x = F.conv2d(F.pad(x, (0, 0, ksize // 2, ksize // 2), mode="replicate"), ky, groups=c)
    return x.clamp(-1, 1)


def resize_cycle(x: torch.Tensor, scale: float = 0.5) -> torch.Tensor:
    b, c, h, w = x.shape
    small = F.interpolate(x, scale_factor=scale, mode="bilinear", align_corners=False)
    return F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False).clamp(-1, 1)


def brightness_contrast(x: torch.Tensor, brightness: float = 0.0, contrast: float = 1.0) -> torch.Tensor:
    mean = x.mean(dim=(2, 3), keepdim=True)
    return ((x - mean) * contrast + mean + brightness).clamp(-1, 1)


def _affine(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """对 (B,C,H,W) 施加 2x3 仿射矩阵（归一化坐标），边界用零填充（模拟真实取景丢失）。"""
    b, c, h, w = x.shape
    grid = F.affine_grid(theta, (b, c, h, w), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


# ---------------- 几何攻击（真实裁剪/旋转/缩放/平移） ----------------
#
# 早期版本的 "crop" 用 torch.roll 实现，那是**循环平移**而不是裁剪：
# 频谱幅度完全不变、只有相位被整体旋转，因此测不出裁剪的真实破坏。
# 下面这组才是真正的几何攻击。

def crop_fill(x: torch.Tensor, margin: int = 2) -> torch.Tensor:
    """真裁剪：四边各裁掉 margin 像素，空出的区域用边缘像素回填。

    与 torch.roll 的循环平移不同，被裁掉的内容永久丢失，且回填区域引入真实的空间
    不连续性 —— 这才是"裁剪/重新取景"攻击。
    """
    margin = int(margin)
    if margin <= 0:
        return x
    padded = F.pad(x, (margin, margin, margin, margin), mode="replicate")
    h, w = x.shape[-2], x.shape[-1]
    return padded[:, :, margin:margin + h, margin:margin + w]


def center_crop_resize(x: torch.Tensor, keep: float = 0.8) -> torch.Tensor:
    """中心裁剪掉 (1-keep) 比例后再缩放回原尺寸（裁剪 + 重采样）。"""
    b, c, h, w = x.shape
    ch, cw = max(1, int(round(h * keep))), max(1, int(round(w * keep)))
    top, left = (h - ch) // 2, (w - cw) // 2
    cropped = x[:, :, top:top + ch, left:left + cw]
    return F.interpolate(cropped, size=(h, w), mode="bilinear", align_corners=False).clamp(-1, 1)


def rotate(x: torch.Tensor, degrees: float = 5.0) -> torch.Tensor:
    """旋转（角度制），边界零填充 —— 真实旋转会丢失四角内容。"""
    theta = torch.zeros(x.shape[0], 2, 3, device=x.device, dtype=x.dtype)
    rad = math.radians(float(degrees))
    cos, sin = math.cos(rad), math.sin(rad)
    theta[:, 0, 0], theta[:, 0, 1] = cos, -sin
    theta[:, 1, 0], theta[:, 1, 1] = sin, cos
    return _affine(x, theta).clamp(-1, 1)


def translate(x: torch.Tensor, shift: int = 2) -> torch.Tensor:
    """整像素平移（零填充），与循环平移明确区分。"""
    shift = int(shift)
    if shift == 0:
        return x
    b, _, h, w = x.shape
    theta = torch.zeros(b, 2, 3, device=x.device, dtype=x.dtype)
    theta[:, 0, 0] = theta[:, 1, 1] = 1.0
    theta[:, 0, 2] = 2.0 * shift / max(w - 1, 1)
    theta[:, 1, 2] = 2.0 * shift / max(h - 1, 1)
    return _affine(x, theta).clamp(-1, 1)


def scale_zoom(x: torch.Tensor, factor: float = 1.1) -> torch.Tensor:
    """以图像中心为原点缩放（factor>1 放大，边界零填充）。"""
    theta = torch.zeros(x.shape[0], 2, 3, device=x.device, dtype=x.dtype)
    theta[:, 0, 0] = 1.0 / float(factor)
    theta[:, 1, 1] = 1.0 / float(factor)
    return _affine(x, theta).clamp(-1, 1)


def schedule_noise(x: torch.Tensor, schedule, t: torch.Tensor | None = None,
                   t_lo: int = 200, t_hi: int = 600) -> torch.Tensor:
    """在扩散调度自身坐标下加噪：x_t = √ᾱ_t·x + √(1-ᾱ_t)·ε。

    扩散再生攻击（加噪到 t_reg 再去噪）的一阶效应就是这个加噪项，而去噪半程
    正是流水线假设可恢复的部分 —— 因此它是训练侧廉价且忠实的再生攻击代理。
    """
    if t is None:
        t = torch.randint(t_lo, t_hi + 1, (x.shape[0],), device=x.device)
    return schedule.add_noise(x, t.to(x.device), torch.randn_like(x)).clamp(-1, 1)


_ATTACKS = {
    "jpeg": lambda x, p: diff_jpeg(x, int(p)),
    "noise": lambda x, p: gauss_noise(x, float(p)),
    "blur": lambda x, p: gauss_blur(x, int(p), sigma=1.0),
    "resize": lambda x, p: resize_cycle(x, float(p)),
    "bright": lambda x, p: brightness_contrast(x, float(p), 1.0),
    "contrast": lambda x, p: brightness_contrast(x, 0.0, float(p)),
    # 几何攻击（真实实现）
    "crop": lambda x, p: crop_fill(x, int(p)),
    "cropresize": lambda x, p: center_crop_resize(x, float(p)),
    "rotate": lambda x, p: rotate(x, float(p)),
    "translate": lambda x, p: translate(x, int(p)),
    "zoom": lambda x, p: scale_zoom(x, float(p)),
}

# 评测用标准攻击表（名称, 参数）；供各 eval 脚本共用，避免口径漂移。
STANDARD_ATTACKS = [
    ("clean", "clean", None),
    ("jpeg30", "jpeg", 30), ("jpeg50", "jpeg", 50), ("jpeg75", "jpeg", 75),
    ("noise0.05", "noise", 0.05), ("noise0.10", "noise", 0.10),
    ("blur3x3", "blur", 3), ("blur5x5", "blur", 5),
    ("resize0.5", "resize", 0.5), ("resize0.7", "resize", 0.7),
    ("bright+0.1", "bright", 0.1), ("contrast1.2", "contrast", 1.2),
    # —— 几何攻击：crop/translate 已改为真实实现（此前 crop 是循环平移）——
    ("crop2px", "crop", 2), ("crop4px", "crop", 4),
    ("cropresize0.8", "cropresize", 0.8),
    ("rotate5", "rotate", 5.0), ("rotate15", "rotate", 15.0),
    ("translate2px", "translate", 2), ("zoom1.1", "zoom", 1.1),
]


def apply_attack(x: torch.Tensor, name: str, param) -> torch.Tensor:
    if name == "clean":
        return x
    return _ATTACKS[name](x, param)


def random_distortion(x: torch.Tensor, rng=None, schedule=None,
                      sched_noise_prob: float = 0.0, geom_prob: float = 0.25) -> torch.Tensor:
    """训练用随机失真：均匀选一种攻击 + 随机强度。

    schedule + sched_noise_prob > 0 时，以该概率改用调度坐标加噪（再生攻击代理）。
    geom_prob 为选中几何攻击（裁剪/旋转/缩放/平移）的条件概率 —— 与评测口径一致，
    避免"训练里没有几何失真、评测里突然加几何攻击"的训练/测试不一致。
    """
    import random
    rng = rng or random
    if schedule is not None and rng.random() < sched_noise_prob:
        return schedule_noise(x, schedule)
    if rng.random() < geom_prob:
        kind = rng.choice(["crop", "cropresize", "rotate", "translate", "zoom"])
        if kind == "crop":
            return crop_fill(x, rng.choice([1, 2, 3]))
        if kind == "cropresize":
            return center_crop_resize(x, rng.uniform(0.75, 0.95))
        if kind == "rotate":
            return rotate(x, rng.uniform(-10.0, 10.0))
        if kind == "translate":
            return translate(x, rng.choice([-2, -1, 1, 2]))
        return scale_zoom(x, rng.uniform(0.9, 1.2))
    kind = rng.choice(["jpeg", "noise", "blur", "resize", "bc"])
    if kind == "jpeg":
        return diff_jpeg(x, rng.choice([30, 40, 50, 60, 75, 90]))
    if kind == "noise":
        return gauss_noise(x, rng.uniform(0.02, 0.10))
    if kind == "blur":
        return gauss_blur(x, rng.choice([3, 5]), rng.uniform(0.8, 1.5))
    if kind == "resize":
        return resize_cycle(x, rng.uniform(0.5, 0.9))
    return brightness_contrast(x, rng.uniform(-0.08, 0.08), rng.uniform(0.8, 1.2))

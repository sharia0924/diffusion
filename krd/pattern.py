"""密钥门控傅里叶环图案：把 ECC 扩展后的比特写入 x_T 的频谱。

密钥 (任意字符串) -> SHA256 -> 确定性随机源，决定：
  - 频谱环带 (r_min, r_max) 与选中的半平面频点集合（含随机置换）；
  - 每个频点的相位 phi_i 与基础幅度 m_i。
比特 b -> 符号 s_i = ±1，写入 F[k_i] = m_i * e^{j phi_i} * s_i * scale，共轭写入镜像点。

错密钥 => 选到完全不同的频点/相位 => 特征与模板去相关 => 解码退化为随机 (~50%)。
全部变换用 ortho 归一化，注入是确定性的、可逆定位的。
"""

import hashlib

import numpy as np
import torch


def _halfplane_bins(res: int, r_min: int, r_max: int) -> np.ndarray:
    """fftshift 坐标下半平面（fx>0 或 fx==0 且 fy>0）内、半径在 [r_min, r_max] 的频点。

    半平面保证每个 ±k 频点对只取一次，写入时同步写共轭镜像，保持频谱 Hermitian（结果为实图）。
    """
    c = res // 2
    ys, xs = np.meshgrid(np.arange(res), np.arange(res), indexing="ij")
    fy, fx = ys - c, xs - c
    r = np.sqrt(fy ** 2 + fx ** 2)
    half = (fx > 0) | ((fx == 0) & (fy > 0))
    keep = half & (r >= r_min) & (r <= r_max)
    return np.stack([ys[keep], xs[keep]], axis=1)


def default_r_min(res: int) -> int:
    """环带内半径的默认值：随分辨率自适应。

    小分辨率（隐空间 8×8 / 16×16）不能沿用固定的 r_min=3 —— 此时 Nyquist 上限
    `res//2-1` 只有 3，可用频点会退化成个位数甚至 0。按 res 缩放；
    对 res≥24 仍返回 3，保持历史结果可复现。
    """
    return max(1, min(3, res // 8))


def ring_capacity(res: int, r_min: int | None = None) -> int:
    """给定分辨率下环带内**可用频点对**总数（单通道，不含通道维度）。

    LDM 迁移的容量核算依赖它：像素 32² 约 176 对；隐空间 8² 只有个位数，
    16² 约 30 对，64² 约 900+ 对 —— 这就是"频点预算 ∝ 分辨率²"的量化依据。
    """
    r_min = default_r_min(res) if r_min is None else int(r_min)
    r_max_cap = res // 2 - 1
    if r_max_cap < r_min:
        return 0
    return len(_halfplane_bins(res, r_min, r_max_cap))


def key_params(key: str, n_pairs: int, res: int = 32, r_min: int | None = None,
               nonce: str = "") -> dict:
    """由密钥(+nonce)确定性生成图案参数。n_pairs 为每图（单通道）写入的复数频点对数。

    nonce 用于逐图随机化图案位置：同一密钥在不同 nonce 下选择完全不同的频点/相位，
    使"多张同密钥载密图差分平均"的密钥恢复攻击失效（见 security.slot_detection_auc）。

    r_min=None 时按分辨率自适应（见 default_r_min），小分辨率下才不会退化为空环带。
    """
    assert n_pairs > 0 and res % 2 == 0
    r_min = default_r_min(res) if r_min is None else int(r_min)
    seed = int.from_bytes(
        hashlib.sha256(f"krd:{key}:{nonce}".encode("utf-8")).digest()[:8], "little"
    )
    rng = np.random.default_rng(seed)

    r_max_cap = res // 2 - 1  # 避开 Nyquist 行/列，保证镜像点唯一
    chosen = None
    for rm in range(r_min, r_max_cap + 1):
        if len(_halfplane_bins(res, r_min, rm)) >= n_pairs:
            chosen = rm
            break
    if chosen is None:
        raise ValueError(
            f"res={res} 的环带放不下 {n_pairs} 对频点（可用 {ring_capacity(res, r_min)} 对），"
            f"请减小 n_pairs 或增大分辨率"
        )

    bins = _halfplane_bins(res, r_min, chosen)
    perm = rng.permutation(len(bins))[:n_pairs]
    bins = bins[perm].astype(np.int64)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=n_pairs).astype(np.float32)
    base_mag = rng.uniform(0.5, 1.5, size=n_pairs).astype(np.float32)
    return {
        "key": key,
        "nonce": nonce,
        "res": res,
        "r": (int(r_min), int(chosen)),
        "bins": torch.from_numpy(bins),
        "phases": torch.from_numpy(phases),
        "base_mag": torch.from_numpy(base_mag),
    }


def check_bits(key: str, nonce: str, n: int = 32) -> torch.Tensor:
    """密钥校验比特：SHA256(key|nonce) 的前 n bit。

    作为"额外槽位"嵌入频谱（与消息比特同一符号调制机制），接收端可用
    matched-filter（无需训练解码器）验证密钥有效性；错密钥距离 ≈ n/2。
    """
    digest = hashlib.sha256(f"krd-check:{key}:{nonce}".encode("utf-8")).digest()
    bits = np.unpackbits(np.frombuffer(digest, dtype=np.uint8))[:n]
    return torch.from_numpy(bits.copy()).float()


def _lin_index(bins: torch.Tensor, W: int) -> torch.Tensor:
    """频点 (row, col) -> 展平后的线性下标，供 gather/scatter 使用。"""
    b = bins.to(dtype=torch.long)
    return b[:, 0] * W + b[:, 1]


def _spec_take(F_: torch.Tensor, bins: torch.Tensor, W: int) -> torch.Tensor:
    """F_ (C,H,W) 复数谱 -> 指定频点处的值 (C, n_pairs)。

    用**扁平 gather** 而不是 `F_[:, rows, cols]` 高级索引：
    后者在部分 PyTorch/CUDA 组合下会触发 IndexKernel 越界断言
    （本机 torch 2.5.0+cu118 上，隐空间 4 通道输入必现；
    像素空间 3 通道侥幸可用），且对批量输入会把三个索引解释到前三个维度上。
    gather/scatter 的语义无歧义，且 CPU/CUDA 行为一致。
    """
    C = F_.shape[0]
    idx = _lin_index(bins, W).to(F_.device).view(1, -1).expand(C, -1)
    return F_.reshape(C, -1).gather(1, idx)


def _spec_put(F_: torch.Tensor, bins: torch.Tensor, W: int, values: torch.Tensor) -> torch.Tensor:
    """把 values (C, n_pairs) 散射回 F_ (C,H,W) 的指定频点（原地修改并返回）。"""
    C = F_.shape[0]
    idx = _lin_index(bins, W).to(F_.device).view(1, -1).expand(C, -1)
    flat = F_.reshape(C, -1)
    flat.scatter_(1, idx, values)
    return F_.view(C, F_.shape[1], F_.shape[2])


def inject_pattern(x_T: torch.Tensor, bits: torch.Tensor, params: dict,
                   strength: float = 1.0, mode: str = "replace") -> torch.Tensor:
    """把 bits（长度 L_e, 取值 {0,1}）写入单张 x_T (C,H,W) 的频谱。

    L_e 组比特，每组占 g = n_pairs // L_e 个频点对；组内符号 = 2b-1。

    mode:
      - "replace"（历史默认，向后兼容）：F[k] <- template * strength * median(|F[k]|)。
        这是**破坏性**写法：它把选中频点的原始系数（含相位）整体覆盖。
        实测在隐空间上只要 strength>0 就把载密图 PSNR 从 35 dB 打到 14.6 dB，
        且 0.001~0.2 区间几乎不变 —— 说明损失来自"覆盖"而非幅度大小。
      - "add"（推荐）：F[k] <- F[k] + d * s_j * e^{jφ_j}，d = strength * median(|F[k]|)。
        保留原始系数与幅度结构，只叠加一个相对幅度为 strength 的图案，
        载密图质量随 strength 平滑可控（这才是"隐写"该有的行为）。
    """
    L_e = int(bits.numel())
    bins = params["bins"].to(x_T.device)
    n_pairs = bins.shape[0]
    assert n_pairs % L_e == 0, f"n_pairs({n_pairs}) 必须能被 L_e({L_e}) 整除"
    g = n_pairs // L_e

    sign = torch.where(bits.to(x_T.device) > 0.5,
                       torch.tensor(1.0, device=x_T.device),
                       torch.tensor(-1.0, device=x_T.device)).repeat_interleave(g)
    phases = params["phases"].to(x_T.device)
    base_mag = params["base_mag"].to(x_T.device)

    C, H, W = x_T.shape
    F_ = torch.fft.fftshift(torch.fft.fft2(x_T, norm="ortho"), dim=(-2, -1))
    ring_mag = _spec_take(F_, bins, W).abs()                           # (C, n_pairs)
    scale = strength * ring_mag.median(dim=1, keepdim=True).values     # (C, 1)

    # strength≈0 时必须**完全不写**：replace 模式下"照常替换"等于把选中频点置零，
    # 破坏反而更大（实测 strength 从 0.3 降到 0.01，PSNR 卡在 14.5 dB 不上升）。
    if float(scale.abs().max()) < 1e-12:
        return x_T

    if mode == "replace":
        template = base_mag[None, :] * torch.exp(1j * phases)[None, :] * sign[None, :]
        F_ = F_.clone()
        _spec_put(F_, bins, W, template * scale)
        mirror = torch.stack([(-bins[:, 0]) % H, (-bins[:, 1]) % W], dim=1)
        _spec_put(F_, mirror, W, torch.conj(template) * scale)
    elif mode == "add":
        delta = scale * sign * torch.exp(1j * phases)[None, :]
        F_ = F_.clone()
        _spec_put(F_, bins, W, _spec_take(F_, bins, W) + delta)
        mirror = torch.stack([(-bins[:, 0]) % H, (-bins[:, 1]) % W], dim=1)
        _spec_put(F_, mirror, W, _spec_take(F_, mirror, W) + torch.conj(delta))
    else:
        raise ValueError(f"未知注入模式: {mode}（可选 replace/add）")
    return torch.fft.ifft2(torch.fft.ifftshift(F_, dim=(-2, -1)), norm="ortho").real


@torch.no_grad()
def ring_features(x_T: torch.Tensor, params: dict) -> torch.Tensor:
    """从单张 x_T (C,H,W) 提取密钥对应频点的特征向量 (2*n_pairs,)。

    关键步骤：按密钥相位**去旋转**（derotate）—— v_j = F[k_j]·e^{-jφ_j}。
    注入的图案 F[k_j] = m_j·e^{jφ_j}·s_j·scale 去旋后，同相分量 ≈ m_j·s_j·scale
    （符号=比特、与密钥无关），正交分量≈噪声。若不去旋，"特征维→符号"的映射
    会随密钥随机旋转，密钥盲的解码器无法学习（损失停在 ln2）。
    归一化：按选中频点幅度中值（尺度不变）。
    """
    bins = params["bins"].to(x_T.device)
    phases = params["phases"].to(x_T.device)
    F_ = torch.fft.fftshift(torch.fft.fft2(x_T, norm="ortho"), dim=(-2, -1))
    vals = _spec_take(F_, bins, x_T.shape[-1])                         # (C, n_pairs) complex
    vals = vals * torch.exp(-1j * phases)[None, :]
    norm = vals.abs().median().clamp(min=1e-8)
    feats = torch.cat([vals.real.mean(0), vals.imag.mean(0)]) / norm
    return feats.float()


def template_features(bits: torch.Tensor, params: dict) -> torch.Tensor:
    """理想（无噪声）去旋转特征模板：同相 = m·s，正交 = 0。"""
    L_e = int(bits.numel())
    bins = params["bins"]
    n_pairs = bins.shape[0]
    g = n_pairs // L_e
    sign = torch.where(bits > 0.5, 1.0, -1.0).repeat_interleave(g)
    base = params["base_mag"].to(bits.device)
    return torch.cat([base * sign, torch.zeros_like(base)]).float()


def ecc_encode(bits: torch.Tensor, reps: int) -> torch.Tensor:
    """简单重复码：每个比特重复 reps 次（原型实现；正式版可换 BCH/LDPC）。"""
    assert reps >= 1
    return bits.repeat_interleave(reps, dim=-1) if bits.dim() > 1 else bits.repeat_interleave(reps)


def ecc_collapse(logits: torch.Tensor, n_bits: int, reps: int) -> torch.Tensor:
    """对 ECC 扩展维度的 logits 组内求均值，还原到 n_bits。"""
    if reps == 1:
        return logits
    return logits.view(*logits.shape[:-1], n_bits, reps).mean(dim=-1)

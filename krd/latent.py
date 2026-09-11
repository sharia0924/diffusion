"""隐空间工具：VAE 潜变量缓存与数据集包装（LDM 迁移的基础设施）。

设计要点：
  - 潜变量**离线缓存**到磁盘（`*.pt`）：VAE 编码只做一次，之后潜空间 DDPM 的训练
    读的是小 48 倍的张量（32²×3 → 4²×4），I/O 与显存都友好；
  - 缓存键包含 VAE 配置与数据集参数，配置变了自动重建，避免用到过期潜变量；
  - `LatentTensorDataset` 直接按索引返回潜变量，配 DataLoader 即可训练。
"""

import hashlib
import json
import os

import torch
from torch.utils.data import Dataset


def cache_key(vae_desc: str, split: str, size: int | None, n: int | None) -> str:
    """缓存指纹：VAE 配置 + 数据集划分 + 尺寸 + 样本数。"""
    raw = f"{vae_desc}|{split}|{size}|{n}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def cache_path(cache_dir: str, key: str) -> str:
    return os.path.join(cache_dir, f"latents_{key}.pt")


class LatentTensorDataset(Dataset):
    """把预计算的潜变量张量包成 Dataset。"""

    def __init__(self, latents: torch.Tensor, labels: torch.Tensor | None = None):
        assert latents.dim() == 4, f"expect (N,C,h,w), got {tuple(latents.shape)}"
        self.latents = latents
        self.labels = labels

    def __len__(self) -> int:
        return self.latents.shape[0]

    def __getitem__(self, i):
        z = self.latents[i]
        if self.labels is None:
            return z, torch.tensor(-1)
        return z, self.labels[i]


@torch.no_grad()
def build_latent_cache(vae, dataset, out_path: str, batch_size: int = 64,
                       device: str = "cpu", log_every: int = 20,
                       dtype: torch.dtype = torch.float32) -> dict:
    """把整个数据集编码成潜变量并落盘；返回元信息。

    使用确定性编码（VAE 的 mu，不采样），保证同一张图每次编码结果一致 ——
    这对隐写/水印的"可复现性"是必要的。
    """
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    zs, ys = [], []
    for i, (x, y) in enumerate(loader):
        z = vae.encode(x.to(device), sample=False)
        zs.append(z.to(dtype=dtype, device="cpu"))
        ys.append(y)
        if log_every and (i + 1) % log_every == 0:
            print(f"  [latent-cache] {i + 1}/{len(loader)} batch", flush=True)
    latents = torch.cat(zs)
    labels = torch.cat(ys)
    z_ch, h, w = latents.shape[1:]
    meta = {"n": int(latents.shape[0]), "z_ch": int(z_ch), "h": int(h), "w": int(w),
            "vae": vae.describe(), "mean": float(latents.mean()), "std": float(latents.std()),
            "absmax": float(latents.abs().max())}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save({"latents": latents, "labels": labels, "meta": meta}, out_path)
    with open(out_path.replace(".pt", ".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  [latent-cache] saved {tuple(latents.shape)} -> {out_path}", flush=True)
    return meta


def load_latent_cache(path: str) -> tuple[torch.Tensor, torch.Tensor, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck["latents"], ck.get("labels"), ck.get("meta", {})


@torch.no_grad()
def decode_latents(vae, z: torch.Tensor, target_hw: tuple[int, int] | None = None) -> torch.Tensor:
    """潜变量 -> 图像（[-1,1]）。"""
    return vae.decode(z, target_hw=target_hw)


def latent_capacity_report(z_shape: tuple[int, int, int], n_bits: int,
                           r_min: int | None = None) -> dict:
    """给一个空间几何，报告可用的频点预算与"每比特观测数天花板"。

    这是 LDM 迁移的核心收益量化：**可用的复数频点对数 ∝ 分辨率² × 通道数**，
    而嵌入槽位数固定时，每比特能平均的观测数就随之提升。

    口径说明：
      - `avail_pairs_all_channels` = 单通道环带可用频点对 × 通道数，即容量的**上限**；
      - `pairs_per_bit_current` = 当前实际配置（`slots × bin_pairs_per_bit`）下的
        每比特观测数 —— 这才是与像素空间对比时应引用的数字；
      - `obs_per_bit_ceiling` = avail_pairs / slots，表示"如果把所有可用频点都用上"的
        上限，用于说明某个几何**能支撑**多少观测。

    参数：
      z_shape = (C, h, w)；n_bits = 槽位数（含 ECC 展开）。
    """
    c, h, w = z_shape
    from krd.pattern import _halfplane_bins, default_r_min
    res = max(h, w)
    r_min = default_r_min(res) if r_min is None else int(r_min)
    r_max_cap = min(h, w) // 2 - 1
    avail = len(_halfplane_bins(res, r_min, r_max_cap)) if r_max_cap >= r_min else 0
    total_pairs = avail * c
    return {"z_shape": (c, h, w), "avail_bins_single_channel": int(avail),
            "avail_pairs_all_channels": int(total_pairs), "r_min": int(r_min),
            "r_max": int(max(r_max_cap, 0)), "slots": int(n_bits),
            "obs_per_bit_ceiling": (total_pairs / n_bits) if n_bits else 0.0}

"""评测脚本共用的小工具：标准 CIFAR 载入、评测输入构造、nonce 协议。

抽出来的目的有三个：
  1. **统一 nonce 协议**：nonce = H(key || counter)，自包含、不依赖 cover
     （旧写法 `derive_nonce(cover)` 需要侧信道传输 nonce，方案并非盲提取）；
  2. **修掉重复取批的旧写法**：旧代码里 `while done < n: next(iter(loader))`
     每次都重建迭代器，实际反复取的是同一个第一批；
  3. 让各 eval 脚本的取数与评测口径一致，避免报告之间"口径漂移"。
"""

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from krd.utils import derive_nonces_from_keys, token_key

CIFAR_TF = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


def cifar_loader(data_root: str, train: bool = False, batch_size: int = 16,
                 shuffle: bool = False, resize: int | None = None) -> DataLoader:
    """CIFAR-10 loader。

    resize：先把图放大到该尺寸（如 64）。**隐空间模型必须与 VAE 训练时的输入尺寸一致**
    （native VAE 用 --resize 64 训练时，这里的 cover 也必须是 64×64，
    否则编码-解码的像素尺度不一致，PSNR 无意义）。
    """
    tf = []
    if resize is not None:
        tf.append(transforms.Resize(resize, antialias=True))
    tf += [transforms.ToTensor(),
           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    ds = datasets.CIFAR10(data_root, train=train, download=True,
                          transform=transforms.Compose(tf))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def gather_covers(loader: DataLoader, n: int) -> torch.Tensor:
    """从 loader 顺序取 n 张图（真正推进迭代器，不再重复第一批）。"""
    xs, done = [], 0
    for x, _ in loader:
        xs.append(x)
        done += x.shape[0]
        if done >= n:
            break
    return torch.cat(xs)[:n]


def make_eval_inputs(loader: DataLoader, n: int, n_bits: int, device: str,
                     nonce_start: int = 0, seed_bits: bool = True):
    """构造评测用 (covers, bits, keys, nonces)。

    nonce 协议：nonce_i = H(key_i || nonce_start + i)，逐图不同、**不含 cover 信息**。
    复原端只需要 (key, counter)；counter 是公开参数，随图传输即可。
    """
    covers = gather_covers(loader, n).to(device)
    keys = [token_key() for _ in range(n)]
    if seed_bits:
        bits = torch.randint(0, 2, (n, n_bits), device=device).float()
    else:  # 确定性比特（复现实验用）
        g = torch.Generator(device="cpu").manual_seed(0)
        bits = torch.randint(0, 2, (n, n_bits), generator=g).float().to(device)
    nonces = derive_nonces_from_keys(keys, start=nonce_start)
    return covers, bits, keys, nonces


def repeat_nonces(key: str, n: int, shared: bool = False) -> list[str]:
    """攻击/对照实验用：shared=True 时所有图共用同一 nonce（复现脆弱基线）。"""
    if shared:
        return [derive_nonces_from_keys([key], start=0)[0]] * n
    return derive_nonces_from_keys([key] * n, start=0)


class StegoIO:
    """统一的"像素空间 <-> 模型空间"适配层（LDM 迁移用）。

    像素空间模型：所有方法直通，行为与以前完全一致。
    隐空间模型：cover 先编码成潜变量再嵌入；失真在**像素**空间施加后再编码回来
    （JPEG/噪声/几何攻击都定义在像素上）；PSNR/SSIM/LPIPS 等图像级指标
    在解码回像素后计算。
    """

    def __init__(self, stego, pixel_res: int | None = None, inject_at: float = 1.0):
        self.stego = stego
        self.vae = getattr(stego, "vae", None)
        self.latent = self.vae is not None
        self.pixel_res = pixel_res or getattr(stego, "pixel_res", 32)
        # 注入时刻：训练/评测必须一致（见 LDM_WORKPOINT_DIAGNOSIS.md §7）。
        # 默认 1.0 = 历史行为（末端注入）。
        self.inject_at = float(inject_at)

    # ---- 空间转换 ----
    def to_space(self, x_pix: torch.Tensor) -> torch.Tensor:
        return x_pix if not self.latent else self.vae.encode(x_pix, sample=False)

    def to_pixels(self, z: torch.Tensor) -> torch.Tensor:
        if not self.latent:
            return z
        return self.vae.decode(z, target_hw=(self.pixel_res, self.pixel_res))

    # ---- 流水线 ----
    def hide(self, covers_pix, bits, keys, hide_steps, strength, nonces=None,
             inject_at=None):
        """inject_at=None 时使用 self.inject_at（由解码器 config 决定）。"""
        ia = self.inject_at if inject_at is None else inject_at
        return self.stego.hide(self.to_space(covers_pix), bits, keys, hide_steps,
                               strength=strength, nonces=nonces, inject_at=ia)

    def recover(self, z_or_pix, keys, rec_steps, decoder, nonces=None):
        return self.stego.recover(z_or_pix, keys, rec_steps, decoder, nonces=nonces)

    def recover_features(self, z_or_pix, keys, rec_steps, nonces=None):
        return self.stego.recover_features(z_or_pix, keys, rec_steps, nonces=nonces)

    def invert(self, z_or_pix, steps):
        return self.stego.invert_latents(z_or_pix, steps)

    def attack(self, z_or_pix, fn):
        """在像素空间施加失真，再回到模型空间（隐空间模型的真实信道）。"""
        return self.to_space(fn(self.to_pixels(z_or_pix)))

    def regeneration_attack(self, z, t_reg: int, steps: int):
        """扩散再生攻击：在模型空间加噪重采样；隐空间模型额外解码到像素。"""
        return self.stego.regeneration_attack(z, t_reg=t_reg, steps=steps)

    def baseline_psnr(self, covers_pix) -> float:
        """无嵌入时的往返 PSNR —— 隐空间链路的天然上限（VAE 往返）。"""
        if not self.latent:
            return float("nan")
        from krd.metrics import psnr
        return psnr(self.to_pixels(self.to_space(covers_pix)), covers_pix)

    def capacity_report(self, n_bits: int) -> dict | None:
        zs = getattr(self.stego, "latent_shape", None)
        if zs is None:
            return None
        from krd.latent import latent_capacity_report
        return latent_capacity_report(tuple(zs), n_bits)

    def describe(self) -> str:
        if not self.latent:
            return "空间=像素（无 VAE）"
        return (f"空间=隐空间 {tuple(getattr(self.stego, 'latent_shape', []))}，"
                f"VAE={self.vae.describe()}")


def build_stego_io(stego, pixel_res: int | None = None) -> StegoIO:
    return StegoIO(stego, pixel_res=pixel_res)

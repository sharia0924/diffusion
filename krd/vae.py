"""VAE 后端：把图像编码到隐空间（LDM 迁移的基础设施）。

提供两种后端，接口一致，便于"本地小 VAE 跑通链路 / 远程 SD-VAE 出正式数字"：

  1. `native`（默认，无额外依赖）—— 本项目自带的卷积 VAE，可用
     `scripts/train_vae.py` 在 CIFAR-10 上快速训练。用于验证整条 LDM 链路
     与"隐空间提升每比特观测数"的机制，重建保真度中等。
  2. `sd`（需要 `pip install diffusers`）—— Stable Diffusion 的 VAE
     （f8 下采样，8 通道 latent，官方 scaling_factor≈0.18215）。
     重建保真度高（512² 图约 30 dB+），是论文正式版的后端。

统一约定：
  - 输入图像 [-1,1]，形状 (B,3,H,W)；
  - `encode(x)` 返回**已乘 scaling_factor** 的 latent（供扩散模型直接使用）；
  - `decode(z)` 接受同尺度 latent 并返回 [-1,1] 图像；
  - `latent_channels` / `downsample` / `scaling_factor` 描述隐空间几何，
    供 pattern/stego 计算容量与每比特观测数。
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- 原生小型 VAE

class _ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, ch)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class NativeVAE(nn.Module):
    """小型卷积 VAE：下采样 2^downsample 倍，latent 通道 z_ch。

    编码器/解码器都用残差块堆叠；KL 权重由训练脚本控制（默认很小，
    接近确定性自编码器，保证重建保真度）。
    """

    def __init__(self, in_ch: int = 3, base: int = 64, z_ch: int = 4,
                 downsample: int = 3, ch_mults=(1, 2, 4, 8)):
        super().__init__()
        # 约定：downsample 表示空间压缩倍数，必须等于 ch_mults 的级数-1，
        # 即每一级之间做一次 stride=2（含首级 in_ch -> chans[0] 的投影）。
        assert downsample == len(ch_mults) - 1, (
            f"downsample({downsample}) 必须等于 len(ch_mults)-1 ({len(ch_mults) - 1})；"
            f"例如 downsample=3 对应 ch_mults=(1,2,4,8)")
        self.in_ch = in_ch
        self.z_ch = z_ch
        self.downsample = downsample
        self.ch_mults = tuple(ch_mults)
        chans = [base * m for m in ch_mults]

        # ---- 编码器：in -> chans[0] -> chans[1] -> ... -> chans[-1] ----
        self.enc_in = nn.Conv2d(in_ch, chans[0], 3, padding=1)
        self.enc_levels = nn.ModuleList()
        for i in range(len(chans) - 1):
            self.enc_levels.append(nn.ModuleList([
                _ResBlock(chans[i]),
                nn.Conv2d(chans[i], chans[i], 3, padding=1),
                nn.Conv2d(chans[i], chans[i + 1], 3, stride=2, padding=1),
            ]))
        self.enc_mid = nn.Sequential(_ResBlock(chans[-1]), _ResBlock(chans[-1]))
        last = chans[-1]
        self.to_mu = nn.Conv2d(last, z_ch, 1)
        self.to_logvar = nn.Conv2d(last, z_ch, 1)

        # ---- 解码器：chans[-1] -> ... -> chans[0] -> in ----
        self.from_z = nn.Conv2d(z_ch, last, 1)
        self.dec_mid = nn.Sequential(_ResBlock(last), _ResBlock(last))
        self.dec_levels = nn.ModuleList()
        for i in reversed(range(len(chans) - 1)):
            self.dec_levels.append(nn.ModuleList([
                nn.Conv2d(chans[i + 1], chans[i], 3, padding=1),   # 上报 + 降通道
                _ResBlock(chans[i]),
                nn.Conv2d(chans[i], chans[i], 3, padding=1),
            ]))
        self.dec_out = nn.Sequential(nn.GroupNorm(8, chans[0]), nn.SiLU(),
                                     nn.Conv2d(chans[0], in_ch, 3, padding=1))

    def _encode_dist(self, x: torch.Tensor):
        h = self.enc_in(x)
        for res, conv, down in self.enc_levels:
            h = conv(res(h))
            h = down(F.silu(h))
        h = self.enc_mid(h)
        return self.to_mu(h), self.to_logvar(h)

    def encode(self, x: torch.Tensor, sample: bool = False) -> torch.Tensor:
        mu, logvar = self._encode_dist(x)
        if sample and self.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu

    def decode(self, z: torch.Tensor, target_hw: tuple[int, int] | None = None) -> torch.Tensor:
        h = self.dec_mid(self.from_z(z))
        levels = len(self.dec_levels)
        if target_hw is not None:
            th, tw = target_hw
            targets = [(max(1, th // (2 ** (levels - 1 - i))),
                        max(1, tw // (2 ** (levels - 1 - i)))) for i in range(levels)]
        else:
            targets = [None] * levels
        for (up_conv, res, conv), tgt in zip(self.dec_levels, targets):
            if tgt is None:
                h = F.interpolate(up_conv(h), scale_factor=2.0, mode="nearest")
            else:
                h = F.interpolate(up_conv(h), size=tgt, mode="nearest")
            h = conv(res(h))
        return self.dec_out(h)

    def forward(self, x: torch.Tensor):
        mu, logvar = self._encode_dist(x)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std) if self.training else mu
        return self.decode(z, target_hw=(x.shape[-2], x.shape[-1])), mu, logvar


# ---------------------------------------------------------------- SD VAE 后端

class SDVAE:
    """Stable Diffusion VAE 包装（diffusers 后端）。"""

    def __init__(self, model_id: str = "stabilityai/sd-vae-ft-mse", device="cpu",
                 dtype=torch.float32):
        try:
            from diffusers import AutoencoderKL
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "SD VAE 后端需要 diffusers：pip install diffusers\n"
                "（也可继续用 --vae-backend native）") from e
        self.vae = AutoencoderKL.from_pretrained(model_id).to(device=device, dtype=dtype)
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)
        self.device = device
        self.dtype = dtype
        self.latent_channels = int(self.vae.config.latent_channels)
        self.downsample = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.scaling_factor = float(self.vae.config.scaling_factor)
        self.model_id = model_id

    @torch.no_grad()
    def encode(self, x: torch.Tensor, sample: bool = False) -> torch.Tensor:
        dist = self.vae.encode(x.to(self.device, self.dtype)).latent_dist
        z = dist.sample() if (sample and self.vae.training) else dist.mode()
        return z * self.scaling_factor

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.vae.decode((z / self.scaling_factor).to(self.dtype)).sample


# ---------------------------------------------------------------- 统一工厂

class VAEWrapper:
    """统一封装：native 模型或 SD 后端，暴露一致的 encode/decode 与几何信息。"""

    def __init__(self, backend: str = "native", ckpt: str | None = None,
                 model_id: str = "stabilityai/sd-vae-ft-mse", device="cpu",
                 dtype=torch.float32, allow_download: bool = True):
        self.backend = backend
        self.device = device
        if backend == "native":
            if not ckpt or not os.path.exists(ckpt):
                raise FileNotFoundError(f"native VAE 需要已训练的 checkpoint: {ckpt}")
            ck = torch.load(ckpt, map_location=device, weights_only=False)
            a = ck.get("args", {})
            cm = a.get("ch_mults", (1, 2, 4))
            if isinstance(cm, str):          # argparse 存进来的是 "1,2,4" 字符串
                cm = tuple(int(v) for v in cm.split(","))
            cm = tuple(int(v) for v in cm)
            self.ch_mults = cm
            # downsample 必须与 ch_mults 级数一致（NativeVAE 内有断言）。
            # 之前回退值写死为 3，会把 2 级下采样的 checkpoint 误判成 4× 压缩。
            ds = int(a.get("downsample", len(cm) - 1))
            self.model = NativeVAE(in_ch=a.get("in_ch", 3), base=a.get("base", 64),
                                   z_ch=a.get("z_ch", 4),
                                   downsample=ds, ch_mults=cm).to(device)
            self.model.load_state_dict(ck["model"])
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad_(False)
            self.latent_channels = int(a.get("z_ch", 4))
            # 命名约定（避免混淆）：
            #   self.downsample_exp = 下采样级数（每级 ×2）
            #   self.downsample     = 线性压缩倍数 = 2 ** downsample_exp
            self.downsample_exp = ds
            self.downsample = 2 ** ds
            self.scaling_factor = float(a.get("scaling_factor", 1.0))
            self.model_id = ckpt
        elif backend == "sd":
            if not allow_download:
                raise RuntimeError("当前配置禁止下载 SD VAE（need --allow-download）")
            sd = SDVAE(model_id, device=os.environ.get("KRD_VAE_DEVICE", device), dtype=dtype)
            self._sd = sd
            self.model = None
            self.latent_channels = sd.latent_channels
            self.downsample = sd.downsample
            self.scaling_factor = sd.scaling_factor
            self.model_id = model_id
        else:
            raise ValueError(f"未知 VAE 后端: {backend}")

    @property
    def is_native(self) -> bool:
        return self.backend == "native"

    @torch.no_grad()
    def encode(self, x: torch.Tensor, sample: bool = False) -> torch.Tensor:
        if self.is_native:
            z = self.model.encode(x.to(self.device), sample=sample)
            return z * self.scaling_factor
        return self._sd.encode(x, sample=sample)

    @torch.no_grad()
    def decode(self, z: torch.Tensor, target_hw: tuple[int, int] | None = None) -> torch.Tensor:
        if self.is_native:
            return self.model.decode((z / self.scaling_factor).to(self.device),
                                     target_hw=target_hw)
        return self._sd.decode(z)

    def latent_shape(self, h: int, w: int) -> tuple[int, int, int]:
        return (self.latent_channels, h // self.downsample, w // self.downsample)

    def describe(self) -> str:
        exp = getattr(self, "downsample_exp", None)
        exp_s = f" (2^{exp})" if exp is not None else ""
        return (f"VAE[{self.backend}] ch={self.latent_channels} "
                f"down={self.downsample}x{exp_s} scale={self.scaling_factor:.5f} "
                f"src={self.model_id}")


def build_vae(backend: str = "native", ckpt: str | None = None, device="cpu",
              dtype=torch.float32, model_id: str = "stabilityai/sd-vae-ft-mse") -> VAEWrapper:
    return VAEWrapper(backend=backend, ckpt=ckpt, model_id=model_id,
                      device=device, dtype=dtype)

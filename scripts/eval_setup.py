"""评测脚本的统一入口辅助：按 checkpoint 自动确定像素尺寸与容量参数。

隐空间模型是在**放大后的像素图**上编码的（如 --resize 64），因此评测脚本必须：
  1. 用与 VAE 训练一致的像素尺寸取 CIFAR 图（resize）；
  2. PSNR/SSIM 在解码回该尺寸的像素后计算；
  3. 容量参数（n_bits/ecc/bpb/check/inject_mode）与训练解码器时一致。

以前各 eval 脚本硬编码 32×32、并用默认容量参数，对隐空间模型会得到无意义的指标
或直接因频点预算不足而报错。
"""

import os

import torch

from scripts.eval_common import StegoIO, cifar_loader
from scripts.train_decoder import load_stego


def pixel_res_of(ddpm_ckpt: str) -> int:
    """从 DDPM checkpoint 推断 cover 的像素尺寸（VAE 训练时的 --resize）。"""
    margs = torch.load(ddpm_ckpt, map_location="cpu", weights_only=False).get("args", {})
    return int(margs.get("resize") or 32)


def load_decoder_cfg(decoder_ckpt: str | None):
    if decoder_ckpt and os.path.exists(decoder_ckpt):
        return torch.load(decoder_ckpt, map_location="cpu", weights_only=True)["config"]
    return None


def eval_setup(ddpm_ckpt: str, data_root: str, batch_size: int, device: str,
               decoder_ckpt: str | None = None):
    """返回 (stego, io, loader, cfg, pixel_res)：自动匹配像素尺寸与容量参数。

    像素空间模型：resize=None、容量参数取默认，行为与历史完全一致（直通）。
    隐空间模型：按 VAE 的 --resize 取图、按解码器 config 配置容量。
    """
    pixel_res = pixel_res_of(ddpm_ckpt)
    cfg = load_decoder_cfg(decoder_ckpt)
    c = cfg or {}
    stego = load_stego(ddpm_ckpt, device, with_vae=True,
                       n_bits=c.get("n_bits", 16), ecc_reps=c.get("ecc", 3),
                       bins_per_bit=c.get("bpb", 2),
                       n_check_bits=c.get("n_check_bits", 32),
                       inject_mode=c.get("inject_mode", "replace"))
    io = StegoIO(stego, pixel_res=pixel_res)
    loader = cifar_loader(data_root, train=False, batch_size=batch_size,
                          resize=None if pixel_res == 32 else pixel_res)
    return stego, io, loader, cfg, pixel_res

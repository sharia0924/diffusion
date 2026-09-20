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
    """推断 cover 的像素尺寸（= VAE 训练时的 --resize）。

    取值链：DDPM 的 args.resize -> 其 vae_ckpt 的 args.resize -> 由 latent_shape
    与 VAE 下采样倍数反推 -> 32。只看 DDPM 的 resize 不够稳：早期 checkpoint
    可能没写该字段，而 VAE checkpoint 里一定有（train_vae.py 的 --resize）。
    """
    margs = torch.load(ddpm_ckpt, map_location="cpu", weights_only=False).get("args", {})
    res = margs.get("resize")
    vck = margs.get("vae_ckpt")
    vargs = {}
    if vck and os.path.exists(vck):
        try:
            vargs = torch.load(vck, map_location="cpu", weights_only=False).get("args", {})
        except Exception:
            vargs = {}
    if not res:
        res = vargs.get("resize")
    if not res and margs.get("latent_shape") and vargs.get("downsample") is not None:
        res = int(margs["latent_shape"][1]) * (2 ** int(vargs["downsample"]))
    return int(res or 32)


def load_decoder_cfg(decoder_ckpt: str | None):
    """读取解码器配置；不存在/损坏时返回 None（调用方负责提示或退化）。"""
    if not decoder_ckpt or not os.path.exists(decoder_ckpt):
        return None
    try:
        return torch.load(decoder_ckpt, map_location="cpu", weights_only=True)["config"]
    except Exception as e:
        print(f"[decoder] 读取 {decoder_ckpt} 失败（{type(e).__name__}），"
              f"按无解码器处理", flush=True)
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
    # maximize_bpb=False：严格按解码器 config 记录的 (bpb, n_pairs) 构造。
    # 若这里让它“顶满预算”，旧 checkpoint（如 16bit/bpb 2）会被顶到 bpb 4，
    # n_pairs 从 160 变 320，而解码头仍按 cfg 的 n_pairs=160 构造 →
    # 特征维度不符。verify_results.py 的 R4 就是这样崩的
    # （mat1 and mat2 shapes cannot be multiplied: 8x640 vs 320x512）。
    stego = load_stego(ddpm_ckpt, device, with_vae=True,
                       n_bits=c.get("n_bits", 16), ecc_reps=c.get("ecc", 3),
                       bins_per_bit=c.get("bpb", 2),
                       n_check_bits=c.get("n_check_bits", 32),
                       inject_mode=c.get("inject_mode", "replace"),
                       r_max=c.get("r_max"), maximize_bpb=False)
    if cfg and cfg.get("n_pairs") and int(stego.n_pairs) != int(cfg["n_pairs"]):
        raise SystemExit(
            f"[eval_setup] 容量口径不一致：解码器 config 记录 n_pairs={cfg['n_pairs']}，"
            f"但按同一 config 构造出的 stego 是 n_pairs={stego.n_pairs} "
            f"(n_bits={stego.n_bits} ecc={stego.ecc} bpb={stego.bpb} "
            f"check={stego.n_check_bits} r_max={c.get('r_max')})。"
            f"这会让解码头维度与提取到的特征维度不符，必须先修好配置再评测。")
    io = StegoIO(stego, pixel_res=pixel_res, inject_at=c.get("inject_at", 1.0),
                 n_inject=c.get("n_inject", 1))
    if c.get("inject_at") is not None:
        print(f"[inject] 从解码器回填 inject_at={io.inject_at}（训练/评测一致）", flush=True)
    loader = cifar_loader(data_root, train=False, batch_size=batch_size,
                          resize=None if pixel_res == 32 else pixel_res)
    return stego, io, loader, cfg, pixel_res

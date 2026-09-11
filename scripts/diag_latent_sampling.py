"""诊断隐空间扩散模型的采样质量：区分"没训够"与"截断设置不当"。

检查三件事：
  1. 从纯噪声 DDIM 采样 -> 解码，看生成的图是否像自然图像（不像 = 没训够）；
  2. clip_denoised=True/False 对往返 PSNR 的影响（x0 截断到 [-1,1] 可能把
     幅值约 ±6 的潜变量压坏）；
  3. 潜变量分布 vs 标准正态（扩散的先验假设 S=1000 时 x_T ≈ N(0,I)）。
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from krd.metrics import psnr, ssim
from krd.utils import seed_everything
from krd.vae import build_vae
from scripts.eval_common import cifar_loader, gather_covers
from scripts.train_decoder import load_stego
from torchvision.utils import save_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddpm-ckpt", default="checkpoints/ddpm_latent.last.pt")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--out-dir", default="results/diag_latent")
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    seed_everything(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(args.ddpm_ckpt, map_location=dev, weights_only=False)
    margs = ck.get("args", {})
    vae = build_vae(margs.get("vae_backend", "native"), ckpt=margs["vae_ckpt"], device=dev)
    stego = load_stego(args.ddpm_ckpt, dev, with_vae=False)
    print(f"vae={vae.describe()}")

    loader = cifar_loader(args.data_root, train=False, batch_size=args.n)
    covers = gather_covers(loader, args.n).to(dev)

    with torch.no_grad():
        z = vae.encode(covers, sample=False)
    print(f"潜变量统计: mean={z.mean():.3f} std={z.std():.3f} absmax={z.abs().max():.3f}")
    print(f"（扩散先验假设 x_T≈N(0,1)，scale={vae.scaling_factor}）")

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) 从纯噪声采样（检验模型是否学会数据分布）
    with torch.no_grad():
        x_T = torch.randn_like(z)
        z_gen = stego.sched.ddim_sample(stego.model, x_T, args.steps)
        x_gen = vae.decode(z_gen, target_hw=(32, 32))
    save_image(torch.cat([covers, x_gen], 0), f"{args.out_dir}/gen_vs_cover.png", nrow=args.n,
               normalize=True, value_range=(-1, 1))
    print(f"[1] 无条件采样: 潜变量 std={z_gen.std():.3f} "
          f"生成图 PSNR vs cover {psnr(x_gen, covers):.2f} dB（无关图应接近随机）")

    # 2) clip_denoised 对往返的影响
    for clip in (True, False):
        with torch.no_grad():
            z_T = stego.sched.ddim_invert(stego.model, z, args.steps, clip_denoised=clip)
            z_rec = stego.sched.ddim_sample(stego.model, z_T, args.steps, clip_denoised=clip)
            x_rec = vae.decode(z_rec, target_hw=(32, 32))
        print(f"[2] clip_denoised={clip}: 往返 PSNR {psnr(x_rec, covers):.2f} dB "
              f"SSIM {ssim(x_rec, covers):.4f} | 潜变量相对 MSE "
              f"{(z_rec - z).pow(2).mean().item() / z.pow(2).mean().item():.4f}")
        save_image(torch.cat([covers, x_rec], 0), f"{args.out_dir}/rt_clip{int(clip)}.png",
                   nrow=args.n, normalize=True, value_range=(-1, 1))
    print(f"对照图已存到 {args.out_dir}/")


if __name__ == "__main__":
    main()

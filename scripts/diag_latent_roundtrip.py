"""隐空间往返诊断：把"VAE 自身损耗"与"DDIM 反演+采样损耗"分开测量。

像素空间实测：无嵌入 DDIM 往返只有 18.81 dB —— 该项就是整条链路的上限。
隐空间路线有两级损耗，必须分开看：

  1. VAE 往返：cover --encode--> z --decode--> x     （VAE 自身的信息损失）
  2. DDIM 往返：z --invert--> z_T --sample--> z' --decode--> x'
     （扩散反演/采样的数值误差，在隐空间上叠加）

只有 (2) 接近 (1) 时，才说明扩散模型在隐空间上学得够好、可以用作载体。

用法:
  python scripts/diag_latent_roundtrip.py --vae-ckpt checkpoints/vae16.pt \
      --ddpm-ckpt checkpoints/ddpm_latent.pt --n 16
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from krd.metrics import psnr, ssim
from krd.utils import seed_everything
from krd.vae import build_vae
from scripts.eval_common import cifar_loader, gather_covers
from scripts.train_decoder import load_stego


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddpm-ckpt", default="checkpoints/ddpm_latent.pt")
    ap.add_argument("--vae-ckpt", default=None,
                    help="默认从 ddpm checkpoint 的 args 里取 vae_ckpt")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", default="20,50,100")
    ap.add_argument("--out", default="results/latent_roundtrip.json")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    seed_everything(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ddpm_ckpt, map_location=device, weights_only=False)
    margs = ck.get("args", {})
    vae_ckpt = args.vae_ckpt or margs.get("vae_ckpt")
    if not vae_ckpt or not os.path.exists(vae_ckpt):
        raise SystemExit(f"找不到 VAE checkpoint: {vae_ckpt}")

    vae = build_vae(margs.get("vae_backend", "native"), ckpt=vae_ckpt, device=device)
    stego = load_stego(args.ddpm_ckpt, device, with_vae=False)
    print(f"[vae] {vae.describe()}")
    print(f"[ddpm] in_ch={margs.get('in_ch')} latent_shape={margs.get('latent_shape')} "
          f"steps_trained≈{ck.get('epochs_done', ck.get('epoch'))} epoch", flush=True)

    loader = cifar_loader(args.data_root, train=False, batch_size=args.batch)
    covers = gather_covers(loader, args.n).to(device)

    report = {"vae_ckpt": vae_ckpt, "ddpm_ckpt": args.ddpm_ckpt, "n": args.n,
              "latent_shape": margs.get("latent_shape")}

    # 1) VAE 往返（无扩散）
    with torch.no_grad():
        z = vae.encode(covers, sample=False)
        x_vae = vae.decode(z, target_hw=(covers.shape[-2], covers.shape[-1]))
    report["vae_roundtrip_psnr"] = psnr(x_vae, covers)
    report["vae_roundtrip_ssim"] = ssim(x_vae, covers)
    print(f"[1] VAE 往返            : PSNR {report['vae_roundtrip_psnr']:.2f} dB "
          f"SSIM {report['vae_roundtrip_ssim']:.4f}", flush=True)

    # 2) DDIM 往返（在隐空间反演+采样后再解码）
    report["ddim_rows"] = []
    for st in [int(v) for v in args.steps.split(",")]:
        with torch.no_grad():
            z_T = stego.sched.ddim_invert(stego.model, z, st)
            z_rec = stego.sched.ddim_sample(stego.model, z_T, st)
            x_rec = vae.decode(z_rec, target_hw=(covers.shape[-2], covers.shape[-1]))
        latent_err = (z_rec - z).pow(2).mean().item()
        z_rms = z.pow(2).mean().item()
        row = {"steps": st, "roundtrip_psnr": psnr(x_rec, covers),
               "roundtrip_ssim": ssim(x_rec, covers),
               "latent_rel_mse": latent_err / max(z_rms, 1e-12),
               "latent_psnr": 10 * torch.log10(torch.tensor(4.0 / max(latent_err, 1e-12))).item()}
        report["ddim_rows"].append(row)
        print(f"[2] DDIM 往返 S={st:<4d}   : PSNR {row['roundtrip_psnr']:.2f} dB "
              f"SSIM {row['roundtrip_ssim']:.4f} | 潜变量相对 MSE {row['latent_rel_mse']:.4f}",
              flush=True)

    best = max(report["ddim_rows"], key=lambda r: r["roundtrip_psnr"])
    report["gap_db"] = report["vae_roundtrip_psnr"] - best["roundtrip_psnr"]
    print(f"\n[结论] VAE 往返 {report['vae_roundtrip_psnr']:.2f} dB → "
          f"加扩散后 {best['roundtrip_psnr']:.2f} dB（损失 {report['gap_db']:.2f} dB）")
    print("      若该差距 < 3 dB，说明扩散模型在隐空间上足够可用作载体。")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()

"""验证「16 dB 地板」的来源：是注入能量，还是反演/采样对 x_T 扰动的放大。

判据：
  - 若注入能量是主因，则把 bins_per_bit 从 2 降到 1（注入系数减半）应显著改善 PSNR；
  - 若反演放大是主因，则改动注入规模几乎不影响该地板值。
"""

import argparse
import sys

import torch

sys.path.insert(0, "/".join(__file__.split("/")[:-2]))
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(
    __import__("os").path.abspath(__file__))))

from krd.metrics import psnr
from krd.pattern import inject_pattern
from krd.utils import derive_nonces_from_keys, seed_everything
from scripts.eval_common import StegoIO, cifar_loader, gather_covers
from scripts.train_decoder import load_stego


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddpm-ckpt", default="checkpoints/ddpm_latent32.pt")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--strengths", default="0.005,0.05,0.5")
    ap.add_argument("--steps", default="50")
    args = ap.parse_args()
    seed_everything(0)
    dev = "cuda"

    loader = cifar_loader("./data", train=False, batch_size=args.n, resize=64)
    covers = gather_covers(loader, args.n).to(dev)

    print(f"{'bpb':>4s} {'strength':>9s} {'steps':>6s} {'PSNR':>8s} {'潜变量相对MSE':>14s}")
    print("-" * 48)
    for bpb in (2, 1):
        stego = load_stego(args.ddpm_ckpt, dev, n_bits=16, ecc_reps=3,
                           bins_per_bit=bpb, n_check_bits=32, with_vae=True,
                           inject_mode="add")
        io = StegoIO(stego, pixel_res=64)
        z = io.to_space(covers)
        keys = ["k1", "k2", "k3", "k4"][:args.n]
        nonces = derive_nonces_from_keys(keys, start=0)
        bits = torch.randint(0, 2, (args.n, stego.n_bits), device=dev).float()
        for st in [float(v) for v in args.strengths.split(",")]:
            for steps in [int(v) for v in args.steps.split(",")]:
                with torch.no_grad():
                    z_T = stego.sched.ddim_invert(stego.model, z, steps)
                    zs = []
                    for i in range(args.n):
                        b = stego.full_bits(bits[i], keys[i], nonces[i])
                        zs.append(inject_pattern(z_T[i], b, stego.params_for(keys[i], nonces[i]),
                                                 st, mode="add"))
                    z_inj = torch.stack(zs)
                    z_rec = stego.sched.ddim_sample(stego.model, z_inj, steps)
                    x_rec = io.to_pixels(z_rec)
                rel = ((z_rec - z) ** 2).mean().item() / z.pow(2).mean().item()
                print(f"{bpb:>4d} {st:>9} {steps:>6d} {psnr(x_rec, covers):>7.2f}dB {rel:>14.5f}")
    print("\n判读：若 bpb 2->1（注入系数减半）后 PSNR 几乎不变，则地板来自反演/采样放大，"
          "\n      而非注入能量 —— 此时降低 strength 无用，需改反演方案（见文档）。")


if __name__ == "__main__":
    main()

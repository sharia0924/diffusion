"""多步注入 A/B 对照（总注入能量守恒）：n_inject 对质量-可解码性的影响。

用法: python scripts/diag_multi_inject.py --n 8 --energies 0.2,0.3,0.4 --ns 1,4,8
输出: 控制台表格 + results/multi_inject_ab.json
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from krd.metrics import bit_accuracy, psnr, ssim
from krd.pattern import ecc_collapse
from krd.utils import derive_nonces_from_keys, seed_everything
from scripts.eval_common import StegoIO, cifar_loader, gather_covers
from scripts.train_decoder import load_stego


def mf_acc(s, z, keys, nonces, bits, steps):
    out = []
    for i in range(z.shape[0]):
        f = s.recover_features(z[i:i + 1], [keys[i]], steps, nonces=[nonces[i]])[0]
        g = s.n_pairs // s.total_embed_bits
        re = f[:s.n_pairs].view(s.total_embed_bits, g).mean(-1)
        lg = ecc_collapse(re[None, :s.msg_embed_bits], s.n_bits, s.ecc)
        out.append(bit_accuracy(lg, bits[i][None]))
    return sum(out) / len(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddpm-ckpt", default="checkpoints/ddpm_latent32.pt")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--inject-at", type=float, default=0.35)
    ap.add_argument("--energies", default="0.2,0.3,0.4")
    ap.add_argument("--ns", default="1,4,8")
    ap.add_argument("--out", default="results/multi_inject_ab.json")
    args = ap.parse_args()
    seed_everything(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    s = load_stego(args.ddpm_ckpt, dev, with_vae=True, inject_mode="add")
    io = StegoIO(s, pixel_res=64)
    covers = gather_covers(cifar_loader("./data", train=False, batch_size=args.n,
                                        resize=64), args.n).to(dev)
    keys = ["k%d" % i for i in range(args.n)]
    nonces = derive_nonces_from_keys(keys, start=0)
    bits = torch.randint(0, 2, (args.n, s.n_bits), device=dev).float()
    z = io.to_space(covers)

    energies = [float(v) for v in args.energies.split(",")]
    ns = [int(v) for v in args.ns.split(",")]
    print(f"配置: res={s.res} bpb={s.bpb} n_pairs={s.n_pairs} S={args.steps} "
          f"inject_at={args.inject_at} n={args.n}")
    print(f"{'总能量E':>8s} {'n':>3s} {'单次s':>9s} {'PSNR':>10s} {'SSIM':>8s} {'mf':>7s}")
    rows = []
    for E in energies:
        for n in ns:
            st = E / n
            sg = s.hide(z, bits, keys, args.steps, st, nonces=nonces,
                        inject_at=args.inject_at, n_inject=n)
            px = io.to_pixels(sg)
            p, ss, a = psnr(px, covers), ssim(px, covers), mf_acc(s, sg, keys, nonces,
                                                                  bits, args.steps)
            rows.append({"E": E, "n": n, "per_inject_strength": st, "psnr": p,
                         "ssim": ss, "mf": a})
            print(f"{E:>8.2f} {n:>3d} {st:>9.4f} {p:>9.2f}dB {ss:>8.4f} {a:>7.3f}",
                  flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()

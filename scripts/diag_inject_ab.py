"""A/B 对照：replace（覆盖系数）vs add（加性注入）在隐空间上的质量-检测权衡。

回答一个关键问题：把注入方式从"覆盖"改成"加性"，能否让载密图 PSNR
随 strength 平滑变化（而不是像 replace 那样一步掉到 14.5 dB 就卡住）。

  python scripts/diag_inject_ab.py --ddpm-ckpt checkpoints/ddpm_latent.pt --n 8
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from krd.distortions import apply_attack
from krd.metrics import bit_accuracy, psnr, ssim
from krd.pattern import ecc_collapse
from krd.utils import derive_nonces_from_keys, seed_everything, token_key
from krd.vae import build_vae
from scripts.eval_common import StegoIO, cifar_loader, gather_covers
from scripts.train_decoder import load_stego


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddpm-ckpt", default="checkpoints/ddpm_latent.pt")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--strengths", default="0,0.005,0.02,0.05,0.1,0.3")
    ap.add_argument("--modes", default="replace,add")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--out", default="results/inject_ab.md")
    ap.add_argument("--seed", type=int, default=31)
    args = ap.parse_args()
    seed_everything(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    loader = cifar_loader(args.data_root, train=False, batch_size=args.batch)
    covers = gather_covers(loader, args.n).to(dev)
    keys = [token_key() for _ in range(args.n)]
    nonces = derive_nonces_from_keys(keys, start=0)

    rows = []
    for mode in args.modes.split(","):
        stego = load_stego(args.ddpm_ckpt, dev, with_vae=True, n_bits=8, ecc_reps=1,
                           n_check_bits=8)
        stego.inject_mode = mode
        io = StegoIO(stego)
        bits = torch.randint(0, 2, (args.n, stego.n_bits),
                             generator=torch.Generator().manual_seed(0)).float().to(dev)
        base = io.baseline_psnr(covers)
        print(f"\n=== 模式 {mode}（无嵌入上限 {base:.2f} dB, n_pairs={stego.n_pairs}, "
              f"res={stego.res}）===", flush=True)
        for st in [float(v) for v in args.strengths.split(",")]:
            sg = torch.cat([
                io.hide(covers[i:i + args.batch], bits[i:i + args.batch],
                        keys[i:i + args.batch], args.steps, st,
                        nonces=nonces[i:i + args.batch])
                for i in range(0, args.n, args.batch)
            ])
            px = io.to_pixels(sg)
            # matched-filter 准确率
            accs = []
            for i in range(args.n):
                p = stego.params_for(keys[i], nonces[i])
                feats = io.recover_features(sg[i:i + 1], [keys[i]], args.steps,
                                            nonces=[nonces[i]])[0]
                gg = stego.n_pairs // stego.total_embed_bits
                re = feats[:stego.n_pairs].view(stego.total_embed_bits, gg).mean(-1)
                lg = ecc_collapse(re[None, :stego.msg_embed_bits], stego.n_bits, stego.ecc)
                accs.append(bit_accuracy(lg, bits[i][None]))
            row = {"mode": mode, "strength": st, "psnr": psnr(px, covers),
                   "ssim": ssim(px, covers), "mf": float(np.mean(accs))}
            rows.append(row)
            print(f"  strength={st:<7}: PSNR {row['psnr']:6.2f} dB "
                  f"SSIM {row['ssim']:.4f} | mf {row['mf']:.3f}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# 注入方式 A/B 对照（replace vs add）\n\n")
        f.write("| 模式 | strength | PSNR | SSIM | mf |\n|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['mode']} | {r['strength']} | {r['psnr']:.2f} | "
                    f"{r['ssim']:.4f} | {r['mf']:.3f} |\n")
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()

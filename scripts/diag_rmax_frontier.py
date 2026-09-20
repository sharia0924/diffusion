"""诊断：环带外径 r_max 的"质量-准确率"前沿（matched-filter 接收，免训练）。

动机（§11 差距分析）：目标是 PSNR≥35 dB **且** JPEG50 解码≥0.95，而当前前沿是
PSNR≈31.5 dB @ mf 0.95。JPEG50 在 64×64 图上本身的量化误差就约 30-33 dB，
所以"把载波放在 JPEG 保得住的低频段"是剩下最直接的一条路：r_max 越小，
环带越靠近 DC，JPEG 量化步长越小、系数越不容易被破坏。

但 r_max 不是单变量：`inject_pattern` 的 `strength` 是**逐系数幅度**
（scale = strength × median(|F|)），频点对数 n_pairs 越多，总注入能量越大、
PSNR 越低。所以不同 r_max 之间**必须在同一 PSNR 上比准确率**，
本脚本因此对每个 r_max 扫多个 strength，输出完整前沿。

决策规则：若某个 r_max 在 PSNR=31~34 dB 区间把 mf@jpeg50 抬到 0.95 以上，
则值得用这个 r_max 重训解码器（约 1.7h）；否则 JPEG 瓶颈不在环带位置。

用法（CPU 即可，避免与训练抢 GPU）：
  python scripts/diag_rmax_frontier.py --n 4 --r-max-list 15,13,11 \
      --strengths 0.15,0.3,0.5 --out results/rmax_frontier.json
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from krd.distortions import apply_attack
from krd.metrics import bit_accuracy, psnr, ssim
from krd.pattern import ecc_collapse, ring_capacity
from krd.utils import derive_nonces_from_keys, seed_everything
from scripts.eval_common import StegoIO, cifar_loader, gather_covers
from scripts.train_decoder import fit_capacity, load_stego


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
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--n-bits", type=int, default=8)
    ap.add_argument("--ecc-reps", type=int, default=3)
    ap.add_argument("--n-check-bits", type=int, default=32)
    ap.add_argument("--r-max-list", default="15,13,11")
    ap.add_argument("--strengths", default="0.15,0.3,0.5")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--inject-at", type=float, default=0.35)
    ap.add_argument("--n-inject", type=int, default=8)
    ap.add_argument("--pixel-res", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/rmax_frontier.json")
    args = ap.parse_args()
    seed_everything(0)

    dev = args.device
    r_max_list = [None if v in ("0", "none", "None") else int(v)
                  for v in args.r_max_list.split(",")]
    strengths = [float(v) for v in args.strengths.split(",")]

    covers = gather_covers(cifar_loader(args.data_root, train=False,
                                        batch_size=args.n, resize=args.pixel_res),
                           args.n).to(dev)
    keys = ["k%d" % i for i in range(args.n)]
    nonces = derive_nonces_from_keys(keys, start=0)

    print(f"{'r_max':>6s} {'pairs':>6s} {'bpb':>4s} {'slots':>6s} {'E':>6s} "
          f"{'PSNR':>9s} {'SSIM':>8s} {'mf':>6s} {'mf@jpeg50':>10s}", flush=True)
    rows = []
    for rm in r_max_list:
        nb, ecc, check, bpb = fit_capacity(32, args.n_bits, args.ecc_reps,
                                           args.n_check_bits, requested_bpb=2,
                                           r_max=rm, maximize_bpb=True)
        s = load_stego(args.ddpm_ckpt, dev, n_bits=nb, ecc_reps=ecc,
                       bins_per_bit=bpb, n_check_bits=check,
                       with_vae=True, inject_mode="add", r_max=rm)
        io = StegoIO(s, pixel_res=args.pixel_res)
        bits = torch.randint(0, 2, (args.n, s.n_bits), device=dev).float()
        z = io.to_space(covers)
        avail = ring_capacity(s.res, r_max=rm)
        for E in strengths:
            st = E / max(1, args.n_inject)
            sg = s.hide(z, bits, keys, args.steps, st, nonces=nonces,
                        inject_at=args.inject_at, n_inject=args.n_inject)
            px = io.to_pixels(sg)
            p, ss = psnr(px, covers), ssim(px, covers)
            a = mf_acc(s, sg, keys, nonces, bits, args.steps)
            sg_j = io.attack(sg, lambda t: apply_attack(t, "jpeg", 50))
            a_j = mf_acc(s, sg_j, keys, nonces, bits, args.steps)
            rows.append({"r_max": rm, "avail_pairs": avail, "bpb": s.bpb,
                         "n_pairs": s.n_pairs, "n_bits": s.n_bits, "ecc": s.ecc,
                         "n_check": s.n_check_bits, "strength": E,
                         "psnr": p, "ssim": ss, "mf": a, "mf_jpeg50": a_j})
            print(f"{str(rm):>6s} {avail:>6d} {s.bpb:>4d} {s.total_embed_bits:>6d} "
                  f"{E:>6.2f} {p:>8.2f}dB {ss:>8.4f} {a:>6.3f} {a_j:>10.3f}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"saved -> {args.out}", flush=True)
    print("\n[比较方式] 不要把同一 strength 的 PSNR 直接横比：strength 是逐系数幅度，"
          "n_pairs 不同则总能量不同。要看的是**同一 PSNR 下**谁 mf@jpeg50 更高。")


if __name__ == "__main__":
    main()

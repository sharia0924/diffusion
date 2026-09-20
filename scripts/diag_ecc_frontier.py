"""诊断：把"每比特观测数 N"真正顶满（ecc/bpb/check 的重新分配）。

§12.4 只考虑了"在 ecc=3、check=32 固定下把 bpb 顶满"，得到 N=18。
但 N = bpb × ecc，而频点预算约束是 (n_bits×ecc + check) × bpb ≤ avail。
把 ecc 调大、bpb 调小，反而能把 N 顶得更高（软合并的重复码在等 SNR 观测上
等价于把所有观测平均，所以**N 才是唯一决定量**）：

    n_bits=8, check=16, avail=342:
      ecc=3  -> slots=40  -> bpb=8 -> N=24
      ecc=5  -> slots=56  -> bpb=6 -> N=30
      ecc=8  -> slots=80  -> bpb=4 -> N=32
      ecc=12 -> slots=112 -> bpb=3 -> N=36
      ecc=21 -> slots=184 -> bpb=1 -> N=21   （过头了）

也就是：**当前 (ecc=3, bpb=6, N=18) 远未顶满**，理论上还能多 2.5-3 dB 的平均增益
（10log10(36/18)=3.0 dB）。同时降低 check 位数可以腾出槽位，代价是错密钥虚警率
（FAR ≈ 2^-check）。

本脚本在**同一载密图 PSNR** 下比较这些 (ecc, check, bpb) 组合的 mf 准确率，
以确认"多出来的 N"能不能兑现成实测增益。默认走 matched-filter 接收（免训练）。

用法（GPU）：
  python scripts/diag_ecc_frontier.py --n 32 --target-psnr 30 \
      --configs "3/32,3/16,5/16,8/16,12/16" --out results/ecc_frontier.json
"""

import argparse
import json
import math
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


def interp_strength(points, target):
    pts = sorted(points)
    if len(pts) == 1:
        s, p = pts[0]
        return float(s * 10 ** ((p - target) / 10.0))
    for (s1, p1), (s2, p2) in zip(pts, pts[1:]):
        if (p1 - target) * (p2 - target) <= 0 and p1 != p2:
            w = (p1 - target) / (p1 - p2)
            return float(10 ** (math.log10(s1) + w * (math.log10(s2) - math.log10(s1))))
    (s1, p1), (s2, p2) = pts[-2], pts[-1]
    slope = (p2 - p1) / (math.log10(s2) - math.log10(s1)) if s2 != s1 else -10.0
    slope = slope if slope < -0.1 else -10.0
    return float(10 ** (math.log10(s2) + (target - p2) / slope))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddpm-ckpt", default="checkpoints/ddpm_latent32.pt")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--n-bits", type=int, default=8)
    ap.add_argument("--r-max", type=int, default=None)
    ap.add_argument("--configs", default="3/32,3/16,5/16,8/16,12/16",
                    help="逗号分隔的 ecc/check 组合")
    ap.add_argument("--target-psnr", type=float, default=30.0)
    ap.add_argument("--tol", type=float, default=0.4)
    ap.add_argument("--max-iters", type=int, default=3)
    ap.add_argument("--attacks", default="clean,jpeg50,jpeg75")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--inject-at", type=float, default=0.35)
    ap.add_argument("--n-inject", type=int, default=8)
    ap.add_argument("--pixel-res", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/ecc_frontier.json")
    args = ap.parse_args()
    seed_everything(0)

    dev = args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu"
    covers = gather_covers(cifar_loader(args.data_root, train=False,
                                        batch_size=args.n, resize=args.pixel_res),
                           args.n).to(dev)
    keys = ["k%d" % i for i in range(args.n)]
    nonces = derive_nonces_from_keys(keys, start=0)

    combos = []
    for item in args.configs.split(","):
        e, c = item.split("/")
        combos.append((int(e), int(c)))

    print(f"目标 PSNR ≈ {args.target_psnr:.1f} dB（n={args.n}, S={args.steps}, "
          f"n_inject={args.n_inject}, r_max={args.r_max}, mf 接收）", flush=True)
    atk_names = [a for a in args.attacks.split(",") if a != "clean"]
    print(f"{'ecc':>4s} {'check':>5s} {'slots':>5s} {'bpb':>4s} {'pairs':>6s} {'N':>3s} "
          f"{'strength':>9s} {'PSNR':>8s} {'SSIM':>8s} {'mf':>6s} " +
          " ".join(f"{a:>9s}" for a in atk_names), flush=True)

    rows = []
    for ecc, check in combos:
        nb, ecc_f, check_f, bpb = fit_capacity(32, args.n_bits, ecc, check,
                                              requested_bpb=2, r_max=args.r_max,
                                              maximize_bpb=True)
        s = load_stego(args.ddpm_ckpt, dev, n_bits=nb, ecc_reps=ecc_f,
                       bins_per_bit=bpb, n_check_bits=check_f,
                       with_vae=True, inject_mode="add", r_max=args.r_max)
        io = StegoIO(s, pixel_res=args.pixel_res)
        bits = torch.randint(0, 2, (args.n, s.n_bits), device=dev).float()
        z = io.to_space(covers)
        pts = [(0.2, args.target_psnr + 2.0), (0.35, args.target_psnr - 1.0)]
        strength, sg, p = None, None, None
        for _ in range(args.max_iters):
            strength = interp_strength(pts, args.target_psnr)
            sg = s.hide(z, bits, keys, args.steps, strength / max(1, args.n_inject),
                        nonces=nonces, inject_at=args.inject_at, n_inject=args.n_inject)
            p = psnr(io.to_pixels(sg), covers)
            pts.append((strength, p))
            print(f"  [search] ecc={ecc} check={check} strength={strength:.4f} -> "
                  f"PSNR {p:.2f} dB", flush=True)
            if abs(p - args.target_psnr) <= args.tol:
                break
        ss = ssim(io.to_pixels(sg), covers)
        accs = {"clean": mf_acc(s, sg, keys, nonces, bits, args.steps)}
        for atk in atk_names:
            q = int(atk.replace("jpeg", ""))
            x = io.attack(sg, lambda t, q=q: apply_attack(t, "jpeg", q))
            accs[atk] = mf_acc(s, x, keys, nonces, bits, args.steps)
        row = {"ecc": s.ecc, "check": s.n_check_bits, "bpb": s.bpb,
               "slots": s.total_embed_bits, "n_pairs": s.n_pairs,
               "N": s.bpb * s.ecc, "strength": strength, "psnr": p, "ssim": ss, **accs}
        rows.append(row)
        print(f"{s.ecc:>4d} {s.n_check_bits:>5d} {s.total_embed_bits:>5d} {s.bpb:>4d} "
              f"{s.n_pairs:>6d} {s.bpb * s.ecc:>3d} {strength:>9.4f} {p:>7.2f}dB "
              f"{ss:>8.4f} {accs['clean']:>6.3f} " +
              " ".join(f"{accs[a]:>9.3f}" for a in atk_names), flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"\nsaved -> {args.out}", flush=True)
    print("[读法] PSNR 已对齐；若高 N 配置的 clean/jpeg 都更高，说明『把每个比特摊到"
          "更多频点上』是当前最划算的零训练杠杆（N=18 -> N=36 理论上 +3 dB）。",
          flush=True)


if __name__ == "__main__":
    main()

"""诊断（决定性实验）：在**同一载密图 PSNR** 下比较不同 r_max 的 JPEG 鲁棒性。

为什么必须"同一 PSNR"：`inject_pattern` 的 strength 是**逐系数幅度**
（scale = strength × median(|F|)），频点对数 n_pairs 越多总注入能量越大。
r_max 越小 → 可用频点越少 → bpb 越小 → n_pairs 越少 → 同一 strength 下
PSNR 反而更高。所以横比"同一 strength"的表格（如 diag_rmax_frontier 的原始输出）
会把"能量差异"混进"频率位置差异"里。

做法：对每个 r_max，用 secant 迭代找到使载密图 PSNR ≈ target 的 strength
（初值由 diag_rmax_frontier 的实测点线性插值给出），然后在该点测
mf clean / jpeg50 / jpeg75 —— 这才是"把载波搬到低频值不值"的干净答案。

用法（GPU，需等训练结束）：
  python scripts/diag_rmax_matched.py --n 32 --target-psnr 30 \
      --r-max-list 15,13,11,9 --out results/rmax_matched.json
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
from krd.pattern import ecc_collapse
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


def load_seed(path: str):
    """从 diag_rmax_frontier 的结果里取 (r_max -> [(strength, psnr), ...]) 作初值。"""
    seed = {}
    if not os.path.exists(path):
        return seed
    try:
        rows = json.load(open(path, encoding="utf-8"))
    except Exception:
        return seed
    for r in rows:
        seed.setdefault(r["r_max"], []).append((r["strength"], r["psnr"]))
    for k in seed:
        seed[k].sort()
    return seed


def interp_strength(points, target):
    """在 (log strength, psnr) 上做线性插值/外推，给出目标 PSNR 对应的 strength。"""
    pts = sorted(points)
    if len(pts) == 1:
        s, p = pts[0]
        return float(s * 10 ** ((p - target) / 10.0))
    for (s1, p1), (s2, p2) in zip(pts, pts[1:]):
        if (p1 - target) * (p2 - target) <= 0 and p1 != p2:
            w = (p1 - target) / (p1 - p2)
            ls = math.log10(s1) + w * (math.log10(s2) - math.log10(s1))
            return float(10 ** ls)
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
    ap.add_argument("--ecc-reps", type=int, default=3)
    ap.add_argument("--n-check-bits", type=int, default=32)
    ap.add_argument("--r-max-list", default="15,13,11,9")
    ap.add_argument("--target-psnr", type=float, default=30.0)
    ap.add_argument("--tol", type=float, default=0.35, help="PSNR 命中容差（dB）")
    ap.add_argument("--max-iters", type=int, default=3)
    ap.add_argument("--attacks", default="clean,jpeg50,jpeg75")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--inject-at", type=float, default=0.35)
    ap.add_argument("--n-inject", type=int, default=8)
    ap.add_argument("--pixel-res", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed-json", default="results/rmax_frontier.json")
    ap.add_argument("--out", default="results/rmax_matched.json")
    args = ap.parse_args()
    seed_everything(0)

    dev = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    seed_tab = load_seed(args.seed_json)
    covers = gather_covers(cifar_loader(args.data_root, train=False,
                                        batch_size=args.n, resize=args.pixel_res),
                           args.n).to(dev)
    keys = ["k%d" % i for i in range(args.n)]
    nonces = derive_nonces_from_keys(keys, start=0)
    r_max_list = [None if v in ("0", "none", "None") else int(v)
                  for v in args.r_max_list.split(",")]

    print(f"目标：所有配置在载密图 PSNR ≈ {args.target_psnr:.2f} dB 处比较 "
          f"（n={args.n}, S={args.steps}, n_inject={args.n_inject}）", flush=True)
    print(f"{'r_max':>6s} {'pairs':>6s} {'bpb':>4s} {'N':>3s} {'strength':>9s} "
          f"{'PSNR':>8s} {'SSIM':>8s} {'mf':>6s} " +
          " ".join(f"{a:>9s}" for a in args.attacks.split(",") if a != "clean"),
          flush=True)

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
        pts = list(seed_tab.get(rm, []))
        if not pts:
            pts = [(0.2, args.target_psnr + 2.0), (0.35, args.target_psnr - 1.0)]
        strength, sg, p = None, None, None
        for _ in range(args.max_iters):
            strength = interp_strength(pts, args.target_psnr)
            sg = s.hide(z, bits, keys, args.steps, strength / max(1, args.n_inject),
                        nonces=nonces, inject_at=args.inject_at, n_inject=args.n_inject)
            p = psnr(io.to_pixels(sg), covers)
            pts.append((strength, p))
            print(f"  [search] r_max={rm} strength={strength:.4f} -> PSNR {p:.2f} dB",
                  flush=True)
            if abs(p - args.target_psnr) <= args.tol:
                break
        ss = ssim(io.to_pixels(sg), covers)
        accs = {"clean": mf_acc(s, sg, keys, nonces, bits, args.steps)}
        for atk in args.attacks.split(","):
            if atk == "clean":
                continue
            q = int(atk.replace("jpeg", ""))
            x = io.attack(sg, lambda t, q=q: apply_attack(t, "jpeg", q))
            accs[atk] = mf_acc(s, x, keys, nonces, bits, args.steps)
        row = {"r_max": rm, "avail_pairs": s.n_pairs, "bpb": s.bpb, "N": s.bpb * s.ecc,
               "strength": strength, "psnr": p, "ssim": ss, **accs}
        rows.append(row)
        print(f"{str(rm):>6s} {s.n_pairs:>6d} {s.bpb:>4d} {s.bpb * s.ecc:>3d} "
              f"{strength:>9.4f} {p:>7.2f}dB {ss:>8.4f} {accs['clean']:>6.3f} " +
              " ".join(f"{accs[a]:>9.3f}" for a in args.attacks.split(",")
                       if a != "clean"), flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"\nsaved -> {args.out}", flush=True)
    print("[读法] PSNR 已被对齐，表内 mf/jpeg 列的差异**只**来自载波频率位置与 bpb；"
          "若小 r_max 的 jpeg50/jpeg75 明显更高，就说明把载波搬到低频是有效的"
          "抗 JPEG 手段，值得用它重训解码器。", flush=True)


if __name__ == "__main__":
    main()

"""诊断：名义观测数 N 里有多少是"有效"的？（解释 10 dB 差距从哪来）

背景：用最朴素的模型估算，总能算出"再多几个观测就够了"，但实测前沿比模型
差 ~10 dB。最可能的原因是**观测之间不独立**：JPEG 的量化误差按 8×8 块结构化，
扩散采样误差也带空间相关性，投到环带后相邻频点的误差是相关的，
于是 10log10(N) 的平均增益根本兑现不了。

本脚本直接量化这件事：
  1. 取每次观测的软统计量 v_j = 去旋转后的同相分量（已知真实比特符号 s_j）；
  2. 估计**单次观测**的偏转 d = E[s_j·v_j] / sd(v_j)（= 单观测信噪比的平方根）；
  3. 若 N 次观测独立，误码率应为 BER_indep = Q(d·√N)；
  4. 与**实测** BER 对比，反解有效观测数 N_eff = (Q⁻¹(BER)/d)²；
  5. 输出 N_eff/N —— 这就是"平均增益的兑现率"。

用法（CPU）：
  python scripts/diag_diversity.py --n 16 --energies 0.2,0.3,0.4 --out results/diversity.json
"""

import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from krd.distortions import apply_attack
from krd.metrics import bit_accuracy, psnr
from krd.pattern import ecc_collapse
from krd.utils import derive_nonces_from_keys, seed_everything
from scripts.eval_common import StegoIO, cifar_loader, gather_covers
from scripts.train_decoder import load_stego


def q_inv(p: float) -> float:
    """标准正态分位数的逆（用 erfinv 实现，避免额外的 scipy 依赖）。"""
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.sqrt(2.0) * float(torch.erfinv(torch.tensor(1.0 - 2.0 * p)))


def q_func(x: float) -> float:
    return 0.5 * (1.0 - math.erf(x / math.sqrt(2.0)))


def mf_slots(feats, n_pairs, total_bits):
    re = feats[:n_pairs].view(total_bits, n_pairs // total_bits)
    return re.mean(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddpm-ckpt", default="checkpoints/ddpm_latent32.pt")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--n-bits", type=int, default=8)
    ap.add_argument("--ecc-reps", type=int, default=3)
    ap.add_argument("--n-check-bits", type=int, default=32)
    ap.add_argument("--bpb", type=int, default=6)
    ap.add_argument("--r-max", type=int, default=None)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--inject-at", type=float, default=0.35)
    ap.add_argument("--n-inject", type=int, default=8)
    ap.add_argument("--energies", default="0.2,0.3,0.4")
    ap.add_argument("--attacks", default="clean,jpeg50",
                    help="逗号分隔；clean 或 jpeg<q>")
    ap.add_argument("--pixel-res", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/diversity.json")
    args = ap.parse_args()
    seed_everything(0)

    dev = args.device
    s = load_stego(args.ddpm_ckpt, dev, n_bits=args.n_bits, ecc_reps=args.ecc_reps,
                   bins_per_bit=args.bpb, n_check_bits=args.n_check_bits,
                   with_vae=True, inject_mode="add", r_max=args.r_max)
    io = StegoIO(s, pixel_res=args.pixel_res)
    covers = gather_covers(cifar_loader(args.data_root, train=False,
                                        batch_size=args.n, resize=args.pixel_res),
                           args.n).to(dev)
    keys = ["k%d" % i for i in range(args.n)]
    nonces = derive_nonces_from_keys(keys, start=0)
    bits = torch.randint(0, 2, (args.n, s.n_bits), device=dev).float()
    z = io.to_space(covers)

    N = s.bpb * s.ecc
    g = s.bpb
    print(f"配置: n_bits={s.n_bits} ecc={s.ecc} bpb={s.bpb} n_pairs={s.n_pairs} "
          f"名义 N={N} S={args.steps} n_inject={args.n_inject} n={args.n} "
          f"r_max={args.r_max}", flush=True)
    print(f"{'攻击':>8s} {'E':>5s} {'PSNR':>9s} {'单观测偏转d':>11s} "
          f"{'实测BER':>9s} {'独立模型BER':>12s} {'N_eff':>7s} {'兑现率':>8s}", flush=True)

    rows = []
    for E in [float(v) for v in args.energies.split(",")]:
        st = E / max(1, args.n_inject)
        sg = s.hide(z, bits, keys, args.steps, st, nonces=nonces,
                    inject_at=args.inject_at, n_inject=args.n_inject)
        p = psnr(io.to_pixels(sg), covers)
        for atk in args.attacks.split(","):
            if atk == "clean":
                x = sg
            else:
                q = int(atk.replace("jpeg", ""))
                x = io.attack(sg, lambda t, q=q: apply_attack(t, "jpeg", q))
            # 收集每次观测的软值与其真实符号（只用消息槽位：比特已知）
            stats, signs, accs = [], [], []
            for i in range(args.n):
                feats = s.recover_features(x[i:i + 1], [keys[i]], args.steps,
                                           nonces=[nonces[i]])[0]
                re = feats[:s.n_pairs]
                msg = bits[i][:s.n_bits]
                sgn_msg = torch.where(msg > 0.5, 1.0, -1.0)
                signs.append(sgn_msg.repeat_interleave(s.ecc))
                stats.append(re.view(s.total_embed_bits, g))
                accs.append(bit_accuracy(
                    ecc_collapse(mf_slots(feats, s.n_pairs, s.total_embed_bits)
                                 [None, :s.msg_embed_bits], s.n_bits, s.ecc),
                    bits[i][None]))
            # 用消息槽位的观测估计单观测偏转：按真实符号折叠后 mean/sd
            dev_msg = torch.stack(stats, 0)[:, :s.msg_embed_bits, :]
            sgn = torch.stack(signs, 0).view(args.n, s.msg_embed_bits, 1)
            folded = (dev_msg * sgn).reshape(-1)
            d = float(folded.mean() / folded.std().clamp(min=1e-9))
            ber = 1.0 - sum(accs) / len(accs)
            ber_indep = q_func(d * math.sqrt(N))
            # 反解有效观测数（BER 太大/太小时不反解，避免数值爆掉）
            n_eff = (q_inv(ber) / d) ** 2 if (1e-6 < ber < 0.5 and d > 1e-6) else float("nan")
            rows.append({"E": E, "attack": atk, "psnr": p, "d": d, "ber": ber,
                         "ber_indep_model": ber_indep, "N": N,
                         "n_eff": n_eff, "yield": n_eff / N if n_eff == n_eff else None})
            ne = "n/a" if n_eff != n_eff else f"{n_eff:.1f}"
            yl = "n/a" if n_eff != n_eff else f"{n_eff / N * 100:.0f}%"
            print(f"{atk:>8s} {E:>5.2f} {p:>8.2f}dB {d:>11.4f} {ber:>9.3f} "
                  f"{ber_indep:>12.4f} {ne:>7s} {yl:>8s}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"\nsaved -> {args.out}", flush=True)
    print("[读法] 若实测 BER 远高于『独立模型 BER』、兑现率 ≪100%，说明观测间强相关"
          "（JPEG 的 8×8 块结构 + 扩散误差的空间相关性），"
          "10log10(N) 的增益拿不到 —— 这就是 4~10 dB 差距的来源。", flush=True)


if __name__ == "__main__":
    main()

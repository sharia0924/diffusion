"""诊断：载波的"频带签名"与集合级（pattern-averaging）攻击。

背景（本实验的直接动机）：
  `scripts/steganalysis_null.py` 在**正确零假设**（null = D(E(cover))，非配对、
  像素域、学习型检测器）下测得：35 dB 操作点 CNN AUC = 0.641（此前报的 0.562
  是配对+线性+对 cover 的假象）。24.6 dB 的阳性对照 AUC = 1.000，说明实验有效。

  **可检测性的来源假设**：`key_params` 总是"从 r_min 往外填满"，所以每张载密图
  的**径向频带是固定的**（r∈[3,15]），与密钥/nonce 无关。于是存在一个与内容
  无关的"频带签名"，学习的检测器只要学到带通滤波就能拿到 AUC>0.5。

本脚本做三件事（都无需训练）：
  1. 测**边际残差的径向功率剖面**：stego 与 clean 的平均功率谱之差按半径积分，
     看签名是否集中在环带内；
  2. 测**无密钥模板攻击**：用训练集估计模板 T = mean|FFT(stego)|² − mean|FFT(clean)|²，
     再对测试图算 <|FFT(x)|², T> 的 AUC（这是攻击者在不知道密钥时的最优线性检验）；
  3. 测**集合级攻击（pattern averaging）**：k 张图取平均谱后再检验，AUC 随 k 的变化。
     如果签名是固定频带，AUC 应随 k 上升（内容方差被平均掉、签名相干累加）。

判据：
  - 若 (2) 的无密钥 AUC 已明显 >0.5 → 频带签名确实可被零成本利用；
  - 若 (3) 的 AUC 随 k 单调上升 → 逐图 nonce **不能**解决集合级检测（因为它只随机化
    相位/槽位，不随机化频带），这将成为论文的核心威胁模型。

用法：
  python scripts/diag_band_signature.py --n 300 --strength 0.5 --ks 1,2,4,8,16
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from krd.pattern import check_bits, ecc_encode, inject_pattern, key_params
from krd.utils import derive_nonces_from_keys, seed_everything, token_key
from krd.vae import build_vae
from scripts.eval_common import cifar_loader, gather_covers

RES_LAT, N_BITS, ECC, N_CHK, BPB = 32, 4, 16, 16, 4
TOTAL_SLOTS = N_BITS * ECC + N_CHK
N_PAIRS = TOTAL_SLOTS * BPB
PIX = 64


def build_params(key, nonce):
    p = key_params(key, N_PAIRS, RES_LAT, nonce=nonce)
    return {"bins": p["bins"], "phases": p["phases"], "base_mag": p["base_mag"]}


@torch.no_grad()
def make_pair(vae, cover, key, nonce, strength, dev):
    z = vae.encode(cover[None], sample=False)
    bits = torch.randint(0, 2, (N_BITS,), device=dev).float()
    fb = torch.cat([ecc_encode(bits, ECC), check_bits(key, nonce, N_CHK).to(dev)])
    z2 = inject_pattern(z[0], fb, build_params(key, nonce), strength, mode="add")[None]
    return vae.decode(z2, target_hw=(PIX, PIX)), vae.decode(z, target_hw=(PIX, PIX))


def auc_score(score, y):
    s, y = torch.as_tensor(score).float(), torch.as_tensor(y).float()
    order = torch.argsort(s)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, len(s) + 1, dtype=torch.float64)
    npos, nneg = float(y.sum()), float(len(y) - y.sum())
    a = (ranks[y > 0.5].sum() - npos * (npos + 1) / 2) / (npos * nneg)
    return float(max(a, 1 - a))


def radial_profile(power, edges):
    """power (N,64,64) 平均后按半径分段求和。"""
    H = power.shape[-1]
    c = (H - 1) / 2.0
    yy, xx = torch.meshgrid(torch.arange(H) - c, torch.arange(H) - c, indexing="ij")
    r = (yy ** 2 + xx ** 2).sqrt()
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (r >= a) & (r < b)
        out.append(float(power[:, m].sum(1).mean()) if m.sum() else 0.0)
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--vae-ckpt", default="checkpoints/vae32.pt")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--strength", type=float, default=0.5)
    ap.add_argument("--mag-profile", action="store_true",
                    help="测试 use_mag_profile=True（幅度 ∝ 局部 |F|，频带签名应被削弱）")
    ap.add_argument("--ks", default="1,2,4,8,16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/band_signature.json")
    args = ap.parse_args()
    seed_everything(0)
    dev = args.device if torch.cuda.is_available() else "cpu"
    vae = build_vae("native", ckpt=args.vae_ckpt, device=dev)

    n = args.n
    covers = gather_covers(cifar_loader(args.data_root, train=False, batch_size=500,
                                        resize=PIX), n).to(dev)
    # 配对：同一 cover 生成 (stego, clean)。对**剖面诊断**这是唯一正确的口径
    # （否则两类内容不同，均值谱差是内容差而非载荷签名）；对线性模板攻击也无害
    # （线性检验无法利用"配对"记忆）。
    keys = [token_key() for _ in range(n)]
    nonc = derive_nonces_from_keys(keys, start=0)
    stego, clean = [], []
    for i in range(n):
        xs, xc = make_pair(vae, covers[i], keys[i], nonc[i], args.strength, dev)
        stego.append(xs[0])
        clean.append(xc[0])
    xs = torch.stack(stego).clamp(-1, 1)          # (n,3,64,64)
    xc = torch.stack(clean).clamp(-1, 1)
    res_px = (xs - xc)                            # 配对残差（仅用于剖面）
    p_r = torch.fft.fftshift(torch.fft.fft2(res_px, norm="ortho"), dim=(-2, -1)).abs() ** 2
    p_s = torch.fft.fftshift(torch.fft.fft2(xs, norm="ortho"), dim=(-2, -1)).abs() ** 2
    p_c = torch.fft.fftshift(torch.fft.fft2(xc, norm="ortho"), dim=(-2, -1)).abs() ** 2

    edges = [0, 4, 8, 12, 16, 22, 28, 33]
    prof_r = radial_profile(p_r.mean(1), edges)     # 残差能量按半径
    prof_s = radial_profile(p_s.mean(1), edges)
    prof_c = radial_profile(p_c.mean(1), edges)
    tot = max(sum(prof_r), 1e-12)
    print(f"strength={args.strength} mag_profile={args.mag_profile} n={n}  "
          f"PSNR(逐图均)={10 * math.log10(4.0 / float((res_px ** 2).mean())):.2f} dB")
    print(f"{'band':>10} {'resid':>11} {'share%':>8} {'imgdiff(dB)':>12}")
    for k, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        share = 100 * prof_r[k] / tot
        dimg = 10 * math.log10(prof_s[k] / max(prof_c[k], 1e-12))
        print(f"{a:>4}-{b:<5} {prof_r[k]:11.3e} {share:7.2f}% {dimg:12.2f}")

    # 无密钥模板攻击（linear, key-less）
    F_s = torch.fft.fftshift(torch.fft.fft2(xs, norm="ortho"), dim=(-2, -1)).abs() ** 2
    F_c = torch.fft.fftshift(torch.fft.fft2(xc, norm="ortho"), dim=(-2, -1)).abs() ** 2
    ntr = n // 2
    T = (F_s[:ntr].mean(1) - F_c[:ntr].mean(1)).mean(0)          # (64,64) 模板
    T = T / T.norm().clamp(min=1e-12)
    sc_s = (F_s[ntr:].mean(1) * T).flatten(1).sum(1)
    sc_c = (F_c[ntr:].mean(1) * T).flatten(1).sum(1)
    score = torch.cat([sc_s, sc_c]).cpu().numpy()
    y = np.concatenate([np.ones(len(sc_s)), np.zeros(len(sc_c))])
    auc = auc_score(score, y)
    print(f"\n[无密钥模板攻击] AUC = {auc:.3f}  (单图, 线性, 不需要 key)")
    res = {"strength": args.strength, "mag_profile": bool(args.mag_profile),
           "psnr": 10 * math.log10(4.0 / float((res_px ** 2).mean())),
           "profile": {"edges": edges, "residual": prof_r, "stego": prof_s, "clean": prof_c},
           "template_auc_single": auc, "set_attack": {}}

    # 集合级攻击：k 张平均谱
    print(f"\n[集合级 pattern-averaging]  'key-less, k 张图取平均谱'")
    print(f"{'k':>4} {'AUC':>7}")
    rng = np.random.default_rng(0)
    for k in [int(v) for v in args.ks.split(",")]:
        g = n // 2 // k
        if g < 5:
            continue
        def group_scores(F):
            idx = rng.permutation(F.shape[0])[:g * k].reshape(g, k)
            m = F[idx].mean(1)                                    # (g,3,64,64)
            return (m.mean(1) * T).flatten(1).sum(1)
        ss, cc = group_scores(F_s[ntr:]), group_scores(F_c[ntr:])
        sc = torch.cat([ss, cc]).cpu().numpy()
        yy = np.concatenate([np.ones(len(ss)), np.zeros(len(cc))])
        a = auc_score(sc, yy)
        res["set_attack"][str(k)] = a
        print(f"{k:>4} {a:7.3f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()


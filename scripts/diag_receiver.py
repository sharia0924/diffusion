"""诊断：接收端判决方式能不能白拿几个 dB（零训练、零改动）。

背景：matched-filter（mf）接收在 §9.3 里全面优于训练解码器，说明"判决器"本身
还有空间。当前 mf 的写法是**等权平均**：

    re = feats[:n_pairs].view(total_bits, g).mean(-1)   # 槽位内 g 个频点对等权
    logits = ecc_collapse(re[:msg_bits], n_bits, ecc)   # 3 个重复等权

但 `ring_features` 返回的是 [同相均值; 正交均值]（长度 2*n_pairs），
**正交分量被完全丢掉了**。而在去旋转之后：
    同相 v_j ≈ 载波幅度 a_j（含比特符号）
    正交 q_j ≈ 该频点的噪声
即每个频点对自带一个**噪声估计**，于是可以做加权合并（MRC / 逆方差加权）：

    w_j = |v_j| / (q_j² + ε)        # 信号幅度 / 噪声功率
    slot = Σ w_j v_j / Σ w_j

不同权重的取值是否真的有用，必须实测（这类"理论上更优"的判决器在非高斯、
重尾的扩散+JPEG 信道上经常不work）。本脚本在**同一批载密潜在量**上比较多种
判决方式，因此 PSNR 完全相同，差异纯粹来自接收端。

用法（CPU 即可）：
  python scripts/diag_receiver.py --n 8 --energies 0.2,0.3,0.4 \
      --out results/receiver_ab.json
"""

import argparse
import json
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

EPS = 1e-6


def slot_values(feats, n_pairs, total_bits, scheme: str):
    """把 (2*n_pairs,) 特征按 scheme 合并成 (total_bits,) 的槽位软值。"""
    re, im = feats[:n_pairs], feats[n_pairs:]
    g = n_pairs // total_bits
    re = re.view(total_bits, g)
    im = im.view(total_bits, g)
    if scheme == "mean":                      # 现行做法
        return re.mean(-1)
    if scheme == "w_mag":                     # 按信号幅度加权
        w = re.abs() + EPS
    elif scheme == "w_invvar":                # 按正交（噪声）功率倒数加权
        w = 1.0 / (im.pow(2) + EPS)
    elif scheme == "w_mrc":                   # 幅度/噪声功率
        w = re.abs() / (im.pow(2) + EPS)
    elif scheme == "w_mrc_abs":               # 幅度/噪声幅度
        w = re.abs() / (im.abs() + EPS)
    else:
        raise ValueError(scheme)
    return (w * re).sum(-1) / w.sum(-1).clamp(min=EPS)


def acc_from_slots(slot, bits, n_bits, ecc, scheme_ecc: str):
    if scheme_ecc == "mean":
        return bit_accuracy(ecc_collapse(slot[None, :n_bits * ecc], n_bits, ecc), bits[None])
    # 按槽位软值幅度加权做重复码合并（软判决）
    v = slot[:n_bits * ecc].view(n_bits, ecc)
    w = v.abs() + EPS
    out = (w * v).sum(-1) / w.sum(-1).clamp(min=EPS)
    return bit_accuracy(out[None], bits[None])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddpm-ckpt", default="checkpoints/ddpm_latent32.pt")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--n-bits", type=int, default=8)
    ap.add_argument("--ecc-reps", type=int, default=3)
    ap.add_argument("--n-check-bits", type=int, default=32)
    ap.add_argument("--r-max", type=int, default=None)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--inject-at", type=float, default=0.35)
    ap.add_argument("--n-inject", type=int, default=8)
    ap.add_argument("--energies", default="0.2,0.3,0.4")
    ap.add_argument("--pixel-res", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/receiver_ab.json")
    args = ap.parse_args()
    seed_everything(0)

    dev = args.device
    s = load_stego(args.ddpm_ckpt, dev, n_bits=args.n_bits, ecc_reps=args.ecc_reps,
                   bins_per_bit=2, n_check_bits=args.n_check_bits,
                   with_vae=True, inject_mode="add", r_max=args.r_max)
    io = StegoIO(s, pixel_res=args.pixel_res)
    covers = gather_covers(cifar_loader(args.data_root, train=False,
                                        batch_size=args.n, resize=args.pixel_res),
                           args.n).to(dev)
    keys = ["k%d" % i for i in range(args.n)]
    nonces = derive_nonces_from_keys(keys, start=0)
    bits = torch.randint(0, 2, (args.n, s.n_bits), device=dev).float()
    z = io.to_space(covers)
    print(f"配置: n_bits={s.n_bits} ecc={s.ecc} bpb={s.bpb} n_pairs={s.n_pairs} "
          f"N={s.bpb * s.ecc} S={args.steps} n_inject={args.n_inject} "
          f"inject_at={args.inject_at} n={args.n} r_max={args.r_max}", flush=True)

    schemes = ["mean", "w_mag", "w_invvar", "w_mrc", "w_mrc_abs"]
    rows = []
    for E in [float(v) for v in args.energies.split(",")]:
        st = E / max(1, args.n_inject)
        sg = s.hide(z, bits, keys, args.steps, st, nonces=nonces,
                    inject_at=args.inject_at, n_inject=args.n_inject)
        sg_j = io.attack(sg, lambda t: apply_attack(t, "jpeg", 50))
        p = psnr(io.to_pixels(sg), covers)
        print(f"\nE={E:.2f}  载密 PSNR {p:.2f} dB", flush=True)
        print(f"{'判决方式':>12s} {'mf(clean)':>10s} {'mf@jpeg50':>10s} "
              f"{'+ecc加权':>10s}", flush=True)
        # 一次性取特征（clean / jpeg50），后面所有方案共用
        feats_c = torch.stack([s.recover_features(sg[i:i + 1], [keys[i]], args.steps,
                                                  nonces=[nonces[i]])[0]
                               for i in range(args.n)])
        feats_j = torch.stack([s.recover_features(sg_j[i:i + 1], [keys[i]], args.steps,
                                                  nonces=[nonces[i]])[0]
                               for i in range(args.n)])
        for sc in schemes:
            accs_c, accs_j, accs_c2 = [], [], []
            for i in range(args.n):
                sl_c = slot_values(feats_c[i], s.n_pairs, s.total_embed_bits, sc)
                sl_j = slot_values(feats_j[i], s.n_pairs, s.total_embed_bits, sc)
                accs_c.append(acc_from_slots(sl_c, bits[i], s.n_bits, s.ecc, "mean"))
                accs_j.append(acc_from_slots(sl_j, bits[i], s.n_bits, s.ecc, "mean"))
                accs_c2.append(acc_from_slots(sl_c, bits[i], s.n_bits, s.ecc, "soft"))
            ac, aj, ac2 = (sum(accs_c) / args.n, sum(accs_j) / args.n,
                           sum(accs_c2) / args.n)
            rows.append({"E": E, "psnr": p, "scheme": sc, "mf_clean": ac,
                         "mf_jpeg50": aj, "mf_clean_ecc_soft": ac2})
            print(f"{sc:>12s} {ac:>10.3f} {aj:>10.3f} {ac2:>10.3f}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"\nsaved -> {args.out}", flush=True)
    print("[读法] 同一 E 下 PSNR 完全相同（同一批载密量），所以表内差异纯粹来自"
          "接收端判决方式；w_mrc 若在 clean 与 jpeg50 上都不低于 mean，"
          "就是零成本净收益。", flush=True)


if __name__ == "__main__":
    main()

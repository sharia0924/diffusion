"""诊断：局部化扩散编辑的编辑比例 f 扫描（找"地板 vs 载波代价"的最优点）。

背景（ITERATION_LOG：采样器放大律）：整幅"反演→注入→重采样"的往返地板在
k=52 时只有 30.41 dB，而载密图 30.26 dB —— 也就是说**地板本身就是瓶颈**。
局部化编辑只让密钥指定的潜变量块经过重采样，其余恒等回填，于是

    E_tot(f) = f·E_rt + a²/f        （E_rt = 整幅重采样误差, a² = 载波能量）

对 f 求极小：f* = √(a²/E_rt)，最优误差 = 2√(E_rt·a²)，比整幅重采样好
10·log10(E_rt / (2√(E_rt a²))) dB。本脚本实测这条曲线：对每个 f，先测
**无嵌入地板**（strength=0），再用**保持判决性**的幅度（a/f）测载密质量与
mf 准确率（clean / jpeg50 / jpeg75），最后给出最优 f 与相对整幅的增益。

用法：
  python scripts/diag_localized.py --n 16 --fracs 0.05,0.1,0.2,0.3,0.5,1.0
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from krd.distortions import apply_attack
from krd.metrics import bit_accuracy, psnr, psnr_mean
from krd.pattern import ecc_collapse, edit_mask_for
from krd.utils import derive_nonces_from_keys, seed_everything
from scripts.eval_common import StegoIO, cifar_loader, gather_covers
from scripts.train_decoder import load_stego


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddpm-ckpt", default="checkpoints/ddpm_latent32.pt")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--n-bits", type=int, default=4)
    ap.add_argument("--ecc-reps", type=int, default=16)
    ap.add_argument("--n-check-bits", type=int, default=16)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--inject-at", type=float, default=0.35)
    ap.add_argument("--n-inject", type=int, default=8)
    ap.add_argument("--strength", type=float, default=0.3,
                    help="f=1 时的总注入能量；实际按 strength/f 放大以保持判决性")
    ap.add_argument("--fracs", default="0.05,0.1,0.2,0.3,0.5,1.0")
    ap.add_argument("--block", type=int, default=2, help="编辑块边长（latent 单元）")
    ap.add_argument("--pixel-res", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/localized.json")
    args = ap.parse_args()
    seed_everything(0)

    dev = args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu"
    s = load_stego(args.ddpm_ckpt, dev, n_bits=args.n_bits, ecc_reps=args.ecc_reps,
                   bins_per_bit=2, n_check_bits=args.n_check_bits, with_vae=True,
                   inject_mode="add", r_max=None)
    io = StegoIO(s, pixel_res=args.pixel_res, inject_at=args.inject_at,
                 n_inject=args.n_inject)
    n = args.n
    covers = gather_covers(cifar_loader(args.data_root, train=False, batch_size=n,
                                        resize=args.pixel_res), n).to(dev)
    keys = ["k%d" % i for i in range(n)]
    nonces = derive_nonces_from_keys(keys, start=0)
    bits = torch.randint(0, 2, (n, s.n_bits), device=dev).float()
    z = io.to_space(covers)
    g = s.n_pairs // s.total_embed_bits
    H, W = z.shape[-2], z.shape[-1]
    print(f"latent={tuple(z.shape)} slots={s.total_embed_bits} bpb={s.bpb} "
          f"N={s.bpb * s.ecc} k={round(args.steps * args.inject_at)} "
          f"strength(f=1)={args.strength} block={args.block} n={n}", flush=True)

    def mf_acc(x, keys_):
        accs = []
        feats = s.recover_features(x, keys_, args.steps, nonces=nonces)
        re = feats[:, :s.n_pairs].view(n, s.total_embed_bits, g).mean(-1)
        msg = ecc_collapse(re[:, :s.msg_embed_bits], s.n_bits, s.ecc)
        for i in range(n):
            accs.append(bit_accuracy(msg[i][None], bits[i][None]))
        return float(np.mean(accs))

    rows = []
    print("%6s %8s %8s %8s %8s %8s %8s" %
          ("f", "floor", "PSNR", "PSNRimg", "clean", "jpeg50", "jpeg75"), flush=True)
    for f in [float(v) for v in args.fracs.split(",")]:
        masks = [edit_mask_for(keys[i], nonces[i], H, W, f, args.block) for i in range(n)]
        mask = torch.stack(masks).to(dev)                       # (n,H,W) bool
        # 1) 无嵌入地板：strength=0（RePaint 掩码条件采样，未编辑区逐位回填）
        sg0 = s.hide_localized(z, bits, keys, args.steps, 0.0, nonces=nonces,
                               inject_at=args.inject_at, n_inject=args.n_inject,
                               edit_mask=mask)
        floor = psnr(io.to_pixels(sg0), covers)
        # 2) 注入：幅度按 1/f 放大（保持全局环带上的相干信号不变）
        st = args.strength / max(f, 1e-6) / max(1, args.n_inject)
        sg = s.hide_localized(z, bits, keys, args.steps, st, nonces=nonces,
                              inject_at=args.inject_at, n_inject=args.n_inject,
                              edit_mask=mask)
        px = io.to_pixels(sg)
        accs = {"clean": mf_acc(sg, keys)}
        for name, q in (("jpeg50", 50), ("jpeg75", 75)):
            x = io.attack(sg, lambda t, qq=q: apply_attack(t, "jpeg", qq))
            accs[name] = mf_acc(x, keys)
        row = {"frac": f, "floor": floor, "psnr": psnr(px, covers),
               "psnr_img": psnr_mean(px, covers), **accs}
        rows.append(row)
        print("%6.2f %8.2f %8.2f %8.2f %8.3f %8.3f %8.3f" %
              (f, floor, row["psnr"], row["psnr_img"], accs["clean"],
               accs["jpeg50"], accs["jpeg75"]), flush=True)

    # 理论最优：用同一套 E_rt / a² 口径（E_rt 用 f=1 的地板估计）
    r1 = next((r for r in rows if r["frac"] >= 0.999), None)
    if r1:
        E_rt = 10 ** (-r1["floor"] / 10)
        E_tot1 = 10 ** (-r1["psnr"] / 10)
        a2 = max(E_tot1 - E_rt, 1e-12)
        f_star = math.sqrt(a2 / E_rt)
        best = 2 * math.sqrt(E_rt * a2)
        print(f"\n[theory] E_rt={E_rt:.3e}  a2={a2:.3e}  f*={f_star:.3f}  "
              f"best_err={best:.3e} -> {10 * math.log10(4.0 / best):.2f} dB "
              f"(full floor {10 * math.log10(4.0 / E_rt):.2f} dB)", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, ensure_ascii=False)
    print(f"saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()

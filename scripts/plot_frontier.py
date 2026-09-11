"""P3 评测：鲁棒性前沿 —— strength 扫描下的 隐秘性(PSNR) vs 鲁棒性(BER) 权衡曲线。

论文主图素材：每方法一条前沿线，同坐标系对比 Tree-Ring/ZoDiac/WAM 等。

  python scripts/plot_frontier.py --ddpm-ckpt checkpoints/ddpm_cifar.pt \
      --decoder-ckpt checkpoints/decoder_best.pt --n 32 --out results/frontier.md
"""

import argparse
import csv
import os
import sys

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from krd import RingDecoder
from krd.distortions import apply_attack
from krd.metrics import bit_accuracy, psnr
from krd.perceptual import lpips
from krd.utils import seed_everything
from scripts.eval_common import cifar_loader, make_eval_inputs
from scripts.eval_setup import eval_setup
from scripts.eval_steps_grid import decode_acc
from scripts.train_decoder import load_stego


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddpm-ckpt", default="checkpoints/ddpm_cifar.pt")
    ap.add_argument("--decoder-ckpt", default="checkpoints/decoder_best.pt")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--rec-steps", type=int, default=50)
    ap.add_argument("--hide-steps", type=int, default=50)
    ap.add_argument("--strengths", default="0.5,0.75,1.0,1.25,1.5,2.0")
    ap.add_argument("--pixel-res", type=int, default=None,
                    help="cover 像素尺寸；由 eval_setup 按 checkpoint 自动匹配")
    ap.add_argument("--nonce-start", type=int, default=0)
    ap.add_argument("--out", default="results/frontier.md")
    ap.add_argument("--seed", type=int, default=19)
    args = ap.parse_args()
    seed_everything(args.seed)
    strengths = [float(v) for v in args.strengths.split(",")]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    stego, io, loader, cfg, pixel_res = eval_setup(
        args.ddpm_ckpt, args.data_root, args.batch, device, args.decoder_ckpt)
    print(f"[space] {io.describe()}  pixel_res={pixel_res}")
    dec = RingDecoder(2 * cfg["n_pairs"],
                      cfg["n_bits"] * cfg["ecc"] + cfg["n_check_bits"]).to(device)
    dec.load_state_dict(torch.load(args.decoder_ckpt, map_location=device,
                                   weights_only=True)["decoder"])
    dec.eval()

    covers, bits, keys, nonces = make_eval_inputs(
        loader, args.n, cfg["n_bits"], device, nonce_start=args.nonce_start)

    eval_attacks = [("clean", "clean", None), ("jpeg50", "jpeg", 50),
                    ("noise0.05", "noise", 0.05), ("blur3x3", "blur", 3),
                    ("crop2px", "crop", 2), ("rotate5", "rotate", 5.0)]
    rows = []
    for s in strengths:
        sg = torch.cat([
            io.hide(covers[i:i + args.batch], bits[i:i + args.batch],
                    keys[i:i + args.batch], args.hide_steps, s,
                    nonces=nonces[i:i + args.batch])
            for i in range(0, args.n, args.batch)
        ])
        sg_px = io.to_pixels(sg)
        p = psnr(sg_px, covers)
        row = {"strength": s, "psnr": p, "lpips": lpips(sg_px, covers)}
        for name, atk, param in eval_attacks:
            x_in = sg if atk == "clean" else io.attack(sg, lambda t: apply_attack(t, atk, param))
            row[name] = decode_acc(stego, x_in, keys, nonces, args.rec_steps,
                                   dec, bits, args.batch, io=io)
        rows.append(row)
        lp_s = "n/a" if row["lpips"] is None else f"{row['lpips']:.4f}"
        print(f"strength={s}: PSNR {p:.2f} dB LPIPS {lp_s} | " +
              " ".join(f"{k}={row[k]:.3f}" for k, _, _ in eval_attacks), flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    header = list(rows[0].keys())
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# 鲁棒性前沿（strength 扫描, S_hide=%d, S_rec=%d）\n\n" % (
            args.hide_steps, args.rec_steps))
        f.write(f"- nonce 协议: `H(key || nonce_start+i)`, nonce_start={args.nonce_start}\n")
        f.write("- 几何攻击为真实实现（crop 真裁剪 / rotate 旋转），LPIPS 为感知距离\n\n")
        f.write("| " + " | ".join(header) + " |\n" + "|" + "---|" * (len(header) + 1) + "\n")
        for row in rows:
            cells = []
            for k in header:
                v = row[k]
                if v is None:
                    cells.append("n/a")
                elif k in ("strength", "psnr"):
                    cells.append(f"{v:.2f}")
                else:
                    cells.append(f"{v:.3f}")
            f.write("| " + " | ".join(cells) + " |\n")
    with open(args.out.replace(".md", ".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            w.writerow(["" if row[k] is None else f"{row[k]:.4f}" for k in header])

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 4))
        for name, _, _ in eval_attacks:
            if name == "clean":
                continue
            ax.plot([r["psnr"] for r in rows], [r[name] for r in rows],
                    marker="o", label=name)
        ax.invert_xaxis()  # 右移 = 更隐秘
        ax.set_xlabel("PSNR of stego (dB) →")
        ax.set_ylabel("bit accuracy")
        ax.set_ylim(0.45, 1.0)
        ax.grid(alpha=0.3)
        ax.legend()
        ax.set_title("Robustness frontier (KRD-Steg)")
        fig.tight_layout()
        fig.savefig(args.out.replace(".md", ".png"), dpi=150)
        plt.close(fig)
    except ImportError:
        print("matplotlib 不可用，跳过绘图")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()

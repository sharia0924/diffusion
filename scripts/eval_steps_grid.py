"""P2 评测：S_hide × S_rec 网格 —— 步数不对称设计的主实验。

预期结论形态：
  - BER 主要随 S_rec 变化（复原端 ODE 积分精度），随 S_hide 几乎平坦；
  - PSNR 主要随 S_hide 变化。
=> "发送端任意质量等级的载密图，接收端一个固定提取器即可解码"。

v2 修复：nonce 协议改为自包含的 H(key||counter)；取数改用 eval_common，
避免旧写法反复取同一批数据。

  python scripts/eval_steps_grid.py --ddpm-ckpt checkpoints/ddpm_cifar.pt \
      --decoder-ckpt checkpoints/decoder_best.pt --n 24 --out results/steps_grid.md
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from krd import RingDecoder
from krd.distortions import diff_jpeg
from krd.metrics import bit_accuracy, psnr, ssim
from krd.utils import seed_everything
from scripts.eval_common import cifar_loader, make_eval_inputs
from scripts.eval_setup import eval_setup
from scripts.train_decoder import load_stego


def parse_ints(s: str) -> list[int]:
    return sorted({int(v) for v in s.split(",")})


def decode_acc(stego, x_in, keys, nonces, rec_steps, decoder, bits, batch, io=None):
    """比特准确率；io（StegoIO）不为空时走编码/解码桥接（隐空间模型必需）。"""
    accs = []
    for i in range(0, x_in.shape[0], batch):
        rx = io.recover if io is not None else stego.recover
        logits = rx(x_in[i:i + batch], keys[i:i + batch], rec_steps, decoder,
                    nonces=nonces[i:i + batch])
        accs.append(bit_accuracy(logits, bits[i:i + batch]))
    return float(np.mean(accs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddpm-ckpt", default="checkpoints/ddpm_cifar.pt")
    ap.add_argument("--decoder-ckpt", default="checkpoints/decoder_best.pt")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--hide-list", default="10,25,50,100")
    ap.add_argument("--rec-list", default="10,25,50,100")
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--pixel-res", type=int, default=None,
                    help="cover 像素尺寸；由 eval_setup 按 checkpoint 自动匹配")
    ap.add_argument("--nonce-start", type=int, default=0)
    ap.add_argument("--out", default="results/steps_grid.md")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    seed_everything(args.seed)
    hide_list, rec_list = parse_ints(args.hide_list), parse_ints(args.rec_list)

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

    stegos_by_sh, quality = {}, {}
    for sh in hide_list:
        sg = torch.cat([
            io.hide(covers[i:i + args.batch], bits[i:i + args.batch],
                    keys[i:i + args.batch], sh, args.strength,
                    nonces=nonces[i:i + args.batch])
            for i in range(0, args.n, args.batch)
        ])
        stegos_by_sh[sh] = sg
        sg_px = io.to_pixels(sg)
        quality[sh] = (psnr(sg_px, covers), ssim(sg_px, covers))
        print(f"S_hide={sh}: PSNR {quality[sh][0]:.2f} dB, SSIM {quality[sh][1]:.4f}",
              flush=True)

    acc = {}
    for sh in hide_list:
        sg = stegos_by_sh[sh]
        # JPEG 定义在像素域：隐空间模型先解码再加失真、再编码回去
        sg_j = io.attack(sg, lambda t: diff_jpeg(t, 50))
        for sr in rec_list:
            acc[("clean", sh, sr)] = decode_acc(stego, sg, keys, nonces, sr, dec,
                                                bits, args.batch, io=io)
            acc[("jpeg50", sh, sr)] = decode_acc(stego, sg_j, keys, nonces, sr, dec,
                                                 bits, args.batch, io=io)
            print(f"S_hide={sh} S_rec={sr}: clean {acc[('clean', sh, sr)]:.3f} "
                  f"jpeg50 {acc[('jpeg50', sh, sr)]:.3f}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# S_hide × S_rec 网格（步数不对称设计）\n\n"
                f"- nonce 协议: `H(key || nonce_start+i)`, nonce_start={args.nonce_start}\n\n"
                "## 载密图质量\n\n| S_hide | PSNR (dB) | SSIM |\n|---|---|---|\n")
        for sh in hide_list:
            f.write(f"| {sh} | {quality[sh][0]:.2f} | {quality[sh][1]:.4f} |\n")
        for atk, label in [("clean", "clean"), ("jpeg50", "JPEG q50")]:
            f.write(f"\n## 比特准确率 ({label})\n\n| S_hide \\ S_rec | " +
                    " | ".join(str(sr) for sr in rec_list) + " |\n"
                    + "|" + "---|" * (len(rec_list) + 1) + "\n")
            for sh in hide_list:
                f.write(f"| {sh} | " +
                        " | ".join(f"{acc[(atk, sh, sr)]:.3f}" for sr in rec_list) + " |\n")
    csv_path = args.out.replace(".md", ".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["attack", "s_hide", "s_rec", "bit_acc"])
        for (atk, sh, sr), v in acc.items():
            w.writerow([atk, sh, sr, f"{v:.4f}"])

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for atk, name in [("clean", "clean"), ("jpeg50", "jpeg50")]:
            fig, ax = plt.subplots(figsize=(5, 4))
            grid = np.array([[acc[(atk, sh, sr)] for sr in rec_list] for sh in hide_list])
            im = ax.imshow(grid, vmin=0.0, vmax=1.0, cmap="viridis", origin="lower")
            ax.set_xticks(range(len(rec_list)), rec_list)
            ax.set_yticks(range(len(hide_list)), hide_list)
            ax.set_xlabel("S_rec (recovery steps)")
            ax.set_ylabel("S_hide (hide steps)")
            ax.set_title(f"bit acc ({name})")
            for yi in range(len(hide_list)):
                for xi in range(len(rec_list)):
                    ax.text(xi, yi, f"{grid[yi, xi]:.2f}", ha="center", va="center",
                            color="w", fontsize=8)
            fig.colorbar(im)
            fig.tight_layout()
            fig.savefig(args.out.replace(".md", f"_{name}.png"), dpi=150)
            plt.close(fig)
    except ImportError:
        print("matplotlib 不可用，跳过热图")
    print(f"saved -> {args.out}, {csv_path}")


if __name__ == "__main__":
    main()

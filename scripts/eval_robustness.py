"""鲁棒性 / 隐秘性 / 密钥安全性 评测。

  python scripts/eval_robustness.py --ddpm-ckpt checkpoints/ddpm_cifar.pt \
      --decoder-ckpt checkpoints/decoder_best.pt --n 64 --out results/robustness.md

输出：各攻击下的比特准确率、stego 相对 cover 的 PSNR/SSIM/LPIPS、错密钥准确率（应≈50%）。

口径说明（v2 修复）：
  - 攻击表来自 `krd.distortions.STANDARD_ATTACKS`，其中 `crop` 是**真裁剪**。
    旧版的 "crop" 用的 torch.roll 循环平移，会系统性高估几何鲁棒性；
  - 几何攻击补齐 rotate / translate / zoom / cropresize；
  - nonce 协议改为自包含的 H(key||counter)，不再依赖 cover。
"""

import argparse
import csv
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from krd import RingDecoder
from krd.distortions import STANDARD_ATTACKS, apply_attack
from krd.metrics import bit_accuracy, psnr, ssim
from krd.perceptual import lpips, lpips_backend
from krd.utils import seed_everything, token_key
from scripts.eval_common import cifar_loader, make_eval_inputs
from scripts.train_decoder import load_stego


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddpm-ckpt", default="checkpoints/ddpm_cifar.pt")
    ap.add_argument("--decoder-ckpt", default="checkpoints/decoder_best.pt")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--hide-steps", type=int, default=50)
    ap.add_argument("--rec-steps", type=int, default=50)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--regen-t", type=int, default=400)
    ap.add_argument("--nonce-start", type=int, default=0)
    ap.add_argument("--out", default="results/robustness.md")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()
    seed_everything(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dec_ckpt = torch.load(args.decoder_ckpt, map_location=device, weights_only=True)
    cfg = dec_ckpt["config"]
    stego = load_stego(args.ddpm_ckpt, device, n_bits=cfg["n_bits"], ecc_reps=cfg["ecc"],
                       bins_per_bit=cfg["bpb"], n_check_bits=cfg["n_check_bits"])
    dec = RingDecoder(2 * cfg["n_pairs"],
                      cfg["n_bits"] * cfg["ecc"] + cfg["n_check_bits"]).to(device)
    dec.load_state_dict(dec_ckpt["decoder"])
    dec.eval()

    loader = cifar_loader(args.data_root, train=False, batch_size=args.batch)
    covers, bits_all, keys_all, nonces_all = make_eval_inputs(
        loader, args.n, cfg["n_bits"], device, nonce_start=args.nonce_start)

    stegos = torch.cat([
        stego.hide(covers[i:i + args.batch], bits_all[i:i + args.batch],
                   keys_all[i:i + args.batch], args.hide_steps, args.strength,
                   nonces=nonces_all[i:i + args.batch])
        for i in range(0, args.n, args.batch)
    ])

    p = psnr(stegos, covers)
    s = ssim(stegos, covers)
    lp = lpips(stegos, covers)
    lp_s = "n/a" if lp is None else f"{lp:.4f}"
    print(f"stego quality: PSNR {p:.2f} dB, SSIM {s:.4f}, LPIPS {lp_s} (n={args.n})")

    def decode_acc(x_in: torch.Tensor, keys: list[str]) -> float:
        accs = []
        for i in range(0, args.n, args.batch):
            logits = stego.recover(x_in[i:i + args.batch], keys[i:i + args.batch],
                                   args.rec_steps, dec,
                                   nonces=nonces_all[i:i + args.batch])
            accs.append(bit_accuracy(logits, bits_all[i:i + args.batch]))
        return sum(accs) / len(accs)

    rows = []
    for name, atk, param in STANDARD_ATTACKS:
        sg = stegos
        x_in = sg if atk == "clean" else apply_attack(sg, atk, param)
        rows.append((name, decode_acc(x_in, keys_all)))
        print(f"{name:>14s}: bit-acc {rows[-1][1]:.3f}", flush=True)

    # 扩散再生攻击（攻击代价同时报告相对 cover 与相对 stego，见 eval_regen.py 的说明）
    regen = torch.cat([
        stego.regeneration_attack(stegos[i:i + args.batch], t_reg=args.regen_t,
                                  steps=args.rec_steps)
        for i in range(0, args.n, args.batch)
    ])
    acc = decode_acc(regen, keys_all)
    rows.append((f"regen(t={args.regen_t})", acc))
    print(f"{'regen':>14s}: bit-acc {acc:.3f} | "
          f"cost vs cover {psnr(regen, covers):.2f} dB / vs stego {psnr(regen, stegos):.2f} dB")

    # 错密钥（密钥安全性: 应≈0.5；nonce 保持与隐藏一致）
    wrong = [token_key() for _ in range(args.n)]
    acc = decode_acc(stegos, wrong)
    rows.append(("wrong-key", acc))
    print(f"{'wrong-key':>14s}: bit-acc {acc:.3f} (期望≈0.5)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# KRD-Steg 鲁棒性评测\n\n")
        f.write(f"- 样本数 n={args.n}, 隐藏步数={args.hide_steps}, 复原步数={args.rec_steps}, "
                f"容量={cfg['n_bits']} bits, strength={args.strength}\n")
        f.write(f"- nonce 协议: `nonce_i = H(key || nonce_start+i)`（自包含, 不依赖 cover）, "
                f"nonce_start={args.nonce_start}\n")
        f.write(f"- stego 质量: **PSNR {p:.2f} dB / SSIM {s:.4f} / LPIPS {lp_s}**"
                f"（LPIPS backend: {lpips_backend() or 'unavailable'}）\n\n")
        f.write("| 攻击 | 比特准确率 |\n|---|---|\n")
        for name, a in rows:
            f.write(f"| {name} | {a:.3f} |\n")
        f.write("\n注: `crop*` 为**真裁剪**（裁边+边缘回填）；`translate*` 为零填充平移；\n"
                "旧版 `crop` 是 `torch.roll` 循环平移，不具备裁剪语义。\n")
    csv_path = args.out.replace(".md", ".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["attack", "bit_acc"])
        for name, a in rows:
            w.writerow([name, f"{a:.4f}"])
        w.writerow(["stego_psnr", f"{p:.4f}"])
        w.writerow(["stego_ssim", f"{s:.4f}"])
        w.writerow(["stego_lpips", "" if lp is None else f"{lp:.6f}"])
    print(f"saved -> {args.out}, {csv_path}")


if __name__ == "__main__":
    main()

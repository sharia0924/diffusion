"""P3 评测：扩散再生攻击网格（攻击预算化）—— v2 修正攻击代价口径。

**修复的问题**：旧版把"攻击代价"算成 `psnr(regen, stego)`，参照物是载密图本身。
但载密图相对原图（cover）可能只有 9.5dB，用它当参照会严重高估攻击者的图像质量，
"攻击者要付出 7.81dB 代价才能抹掉水印"的结论因此不成立 —— 读者关心的是
"攻击后的图相对**原图**还剩多少质量"。

现在每个攻击点同时报告：
  - `d_cover`  = PSNR(regen, cover)   ← 真正的攻击代价（越大越"温和"）
  - `d_stego`  = PSNR(regen, stego)   ← 攻击对载密图的改动幅度
  - `lpips_cover` = LPIPS(regen, cover) ← 感知代价（PSNR 在这一量级已失去意义）
  - `ber`      = 水印误码率

只报告 `ber` 与 `d_cover`/`lpips_cover` 的关系曲线，"攻击者困境"才成立。

  python scripts/eval_regen.py --ddpm-ckpt checkpoints/ddpm_cifar.pt \
      --decoder-ckpt checkpoints/decoder_best.pt --n 32 --out results/regen.md
"""

import argparse
import csv
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from krd import RingDecoder
from krd.metrics import bit_accuracy, psnr, ssim
from krd.perceptual import lpips, lpips_backend
from krd.utils import seed_everything
from scripts.eval_common import cifar_loader, make_eval_inputs
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
    ap.add_argument("--t-regs", default="200,400,600,800,999")
    ap.add_argument("--regen-steps-list", default="25,50")
    ap.add_argument("--hide-steps", type=int, default=50)
    ap.add_argument("--pixel-res", type=int, default=None,
                    help="cover 像素尺寸；隐空间模型由 eval_setup 自动匹配")
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--nonce-start", type=int, default=0)
    ap.add_argument("--out", default="results/regen.md")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    # 步数默认值从解码器 config 回填（训练/评测工作点必须一致）
    _explicit = {"--hide-steps", "--rec-steps"} & set(sys.argv)
    try:
        import torch as _t
        if os.path.exists(args.decoder_ckpt):
            _dcfg = _t.load(args.decoder_ckpt, map_location="cpu",
                            weights_only=True)["config"]
            if "--hide-steps" not in _explicit and _dcfg.get("hide_steps"):
                args.hide_steps = int(_dcfg["hide_steps"])
            if "--rec-steps" not in _explicit and _dcfg.get("rec_steps") \
                    and hasattr(args, "rec_steps"):
                args.rec_steps = int(_dcfg["rec_steps"])
            print(f"[steps] 从解码器回填: hide={getattr(args, 'hide_steps', None)} "
                  f"rec={getattr(args, 'rec_steps', None)}", flush=True)
    except Exception:
        pass
    seed_everything(args.seed)
    t_regs = sorted({int(v) for v in args.t_regs.split(",")})
    steps_list = sorted({int(v) for v in args.regen_steps_list.split(",")})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from scripts.eval_setup import eval_setup
    dec_ckpt = torch.load(args.decoder_ckpt, map_location=device, weights_only=True)
    cfg = dec_ckpt["config"]
    stego, io, loader, cfg, pixel_res = eval_setup(
        args.ddpm_ckpt, args.data_root, args.batch, device, args.decoder_ckpt)
    print(f"[space] {io.describe()}  pixel_res={pixel_res}")
    dec = RingDecoder(2 * cfg["n_pairs"],
                      cfg["n_bits"] * cfg["ecc"] + cfg["n_check_bits"]).to(device)
    dec.load_state_dict(dec_ckpt["decoder"])
    dec.eval()

    covers, bits, keys, nonces = make_eval_inputs(
        loader, args.n, cfg["n_bits"], device, nonce_start=args.nonce_start)

    sg = torch.cat([
        io.hide(covers[i:i + args.batch], bits[i:i + args.batch],
                   keys[i:i + args.batch], args.hide_steps, args.strength,
                   nonces=nonces[i:i + args.batch])
        for i in range(0, args.n, args.batch)
    ])
    print(f"stego 基线: PSNR(vs cover) {psnr(io.to_pixels(sg), covers):.2f} dB / SSIM {ssim(io.to_pixels(sg), covers):.4f}",
          flush=True)

    rows = []
    for steps in steps_list:
        for t_reg in t_regs:
            regen = torch.cat([
                io.regeneration_attack(sg[i:i + args.batch], t_reg=t_reg, steps=steps)
                for i in range(0, args.n, args.batch)
            ])
            d_cover = psnr(io.to_pixels(regen), covers)   # 真正的攻击代价
            d_stego = psnr(io.to_pixels(regen), io.to_pixels(sg))  # 对载密图的改动幅度
            lp = lpips(io.to_pixels(regen), covers)
            ber = 1.0 - decode_acc(stego, regen, keys, nonces, args.rec_steps,
                                   dec, bits, args.batch, io=io)
            rows.append((t_reg, steps, d_cover, d_stego, lp, ber))
            lp_s = "n/a" if lp is None else f"{lp:.4f}"
            print(f"regen t={t_reg} steps={steps}: cost(vs cover) {d_cover:.2f} dB | "
                  f"vs stego {d_stego:.2f} dB | LPIPS {lp_s} | BER {ber:.3f}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# 扩散再生攻击（预算化：攻击代价以 **cover** 为参照）\n\n")
        f.write(f"- 样本数 n={args.n}, S_hide={args.hide_steps}, S_rec={args.rec_steps}, "
                f"strength={args.strength}\n")
        f.write(f"- 载密图基线: PSNR(vs cover) **{psnr(io.to_pixels(sg), covers):.2f} dB**\n")
        f.write(f"- LPIPS backend: {lpips_backend() or 'unavailable'}\n\n")
        f.write("| t_reg | regen steps | 攻击代价 PSNR vs cover | PSNR vs stego | "
                "LPIPS vs cover | BER |\n|---|---|---|---|---|---|\n")
        for t_reg, steps, d_cover, d_stego, lp, ber in rows:
            lp_s = "n/a" if lp is None else f"{lp:.4f}"
            f.write(f"| {t_reg} | {steps} | {d_cover:.2f} | {d_stego:.2f} | {lp_s} | "
                    f"{ber:.3f} |\n")
        f.write("\n注: `PSNR vs cover` 才是攻击者付出的图像质量代价；`vs stego` 仅表示"
                "攻击对载密图的改动幅度。t_reg=999 相当于完全重生成，是攻击者上限。\n"
                "只有 BER 与 (PSNR/LPIPS vs cover) 的联合曲线才支持\"攻击者困境\"的结论。\n")
    csv_path = args.out.replace(".md", ".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t_reg", "regen_steps", "cost_psnr_vs_cover", "cost_psnr_vs_stego",
                    "lpips_vs_cover", "ber"])
        for t_reg, steps, d_cover, d_stego, lp, ber in rows:
            w.writerow([t_reg, steps, f"{d_cover:.3f}", f"{d_stego:.3f}",
                        "" if lp is None else f"{lp:.6f}", f"{ber:.4f}"])
    print(f"saved -> {args.out}, {csv_path}")


if __name__ == "__main__":
    main()

"""P1 评测：密钥安全性全表（v2 修正威胁模型与口径）。

内容：
  A. 错密钥/近密钥 BER 分布（需训练好的解码器）+ 真密钥对照
  B. matched-filter 密钥校验：真/错密钥距离分布 + FAR@tau（无需解码器）
  C. 多图差分攻击：**槽位定位 AUC** vs N（无 nonce / 逐图 nonce / 已知 nonce 攻击者）
     —— 该指标衡量"能否定位水印频点"，**不是密钥恢复**；AUC 明显 >0.5 即攻击有效
  D. 跨图残差方向一致性（|corr|），直接量化"逐图 nonce 把差分平均打散了多少"
  E. **盲水印检测 AUC**（隐写分析视角）：只给图，能否区分载密图与原图
  F. 密钥空间：分别报告"派生参数空间下界"与"密钥本身熵"

  python scripts/eval_key_security.py --ddpm-ckpt checkpoints/ddpm_cifar.pt \
      --decoder-ckpt checkpoints/decoder_best.pt --n 32 --n-wrong 200 \
      --out results/key_security.md
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from krd import RingDecoder, security
from krd.perceptual import lpips, lpips_backend
from krd.metrics import psnr, ssim
from krd.utils import derive_nonces_from_keys, seed_everything, token_key
from scripts.eval_common import cifar_loader, make_eval_inputs
from scripts.train_decoder import load_stego


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddpm-ckpt", default="checkpoints/ddpm_cifar.pt")
    ap.add_argument("--decoder-ckpt", default="checkpoints/decoder_best.pt")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--n", type=int, default=32, help="多图攻击与 BER 评测的图像数")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--n-wrong", type=int, default=200)
    ap.add_argument("--hide-steps", type=int, default=50)
    ap.add_argument("--rec-steps", type=int, default=50)
    ap.add_argument("--pixel-res", type=int, default=None,
                    help="cover 像素尺寸；隐空间模型由 eval_setup 自动匹配")
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--auc-ns", default="1,2,4,8,16,32")
    ap.add_argument("--nonce-start", type=int, default=0)
    ap.add_argument("--key-hex-len", type=int, default=16,
                    help="token_key 的 hex 字符数（16 -> 64 bit 熵）")
    ap.add_argument("--out", default="results/key_security.md")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    seed_everything(args.seed)
    auc_ns = [int(v) for v in args.auc_ns.split(",")]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from scripts.eval_setup import eval_setup
    stego, io, loader, cfg, pixel_res = eval_setup(
        args.ddpm_ckpt, args.data_root, args.batch, device, args.decoder_ckpt)
    print(f"[space] {io.describe()}  pixel_res={pixel_res}")

    dec_ckpt = torch.load(args.decoder_ckpt, map_location=device, weights_only=True) \
        if os.path.exists(args.decoder_ckpt) else None
    cfg = dec_ckpt["config"] if dec_ckpt else {}
    decoder = None
    if dec_ckpt:
        decoder = RingDecoder(2 * cfg["n_pairs"],
                              cfg["n_bits"] * cfg["ecc"] + cfg.get("n_check_bits", 0)).to(device)
        decoder.load_state_dict(dec_ckpt["decoder"])
        decoder.eval()
        print(f"loaded decoder from {args.decoder_ckpt}")
    else:
        print("WARNING: 未找到解码器 checkpoint，跳过 A 部分（BER 分布）")

    covers, bits_all, keys_all, nonces = make_eval_inputs(
        loader, args.n, stego.n_bits, device, nonce_start=args.nonce_start)

    stegos = torch.cat([
        io.hide(covers[i:i + args.batch], bits_all[i:i + args.batch],
                   keys_all[i:i + args.batch], args.hide_steps, args.strength,
                   nonces=nonces[i:i + args.batch])
        for i in range(0, args.n, args.batch)
    ])

    # 像素级指标必须在解码回像素后计算（隐空间模型下 stegos 是潜变量）
    stegos_px = io.to_pixels(stegos)
    p_stego = psnr(stegos_px, covers)
    lp_stego = lpips(stegos_px, covers)
    _ssim = ssim(stegos_px, covers)
    lines = ["# 密钥安全性评测（P1）\n",
             f"- 样本数 n={args.n}, S_hide={args.hide_steps}, S_rec={args.rec_steps}, "
             f"strength={args.strength}",
             f"- {io.describe()}",
             f"- 载密图基线: PSNR {p_stego:.2f} dB / SSIM {_ssim:.4f} / "
             f"LPIPS {'n/a' if lp_stego is None else f'{lp_stego:.4f}'}",
             f"- nonce 协议: `nonce_i = H(key || nonce_start+i)`（自包含, 不依赖 cover）\n"]
    csv_rows = [("stego_psnr", f"{p_stego:.4f}"),
                ("stego_ssim", f"{_ssim:.4f}"),
                ("stego_lpips", "" if lp_stego is None else f"{lp_stego:.6f}")]

    x_T = io.invert(stegos, args.rec_steps)

    # ---------- A. 错密钥 / 近密钥 BER 分布 ----------
    if decoder is not None:
        wrong = security.wrong_key_population(keys_all, n_random=args.n_wrong, seed=args.seed)
        bers = security.wrong_key_profile(stego, x_T, bits_all, nonces, decoder, wrong)
        t = bers.cpu().numpy()
        lines += [
            "## A. 错密钥/近密钥 BER 分布\n",
            f"- 错密钥数 {len(wrong)}（随机 {args.n_wrong} + 近密钥变体）",
            f"- BER: mean **{t.mean():.4f}** / std {t.std():.4f} / "
            f"min {t.min():.4f} / max {t.max():.4f}（期望 0.5，越集中越安全）\n",
        ]
        csv_rows += [("wrong_key_ber_mean", f"{t.mean():.4f}"),
                     ("wrong_key_ber_std", f"{t.std():.4f}"),
                     ("wrong_key_ber_max", f"{t.max():.4f}")]

        true_accs = []
        for i in range(0, args.n, args.batch):
            logits = io.recover(stegos[i:i + args.batch], keys_all[i:i + args.batch],
                                   args.rec_steps, decoder, nonces=nonces[i:i + args.batch])
            true_accs.append(((logits > 0).float() == bits_all[i:i + args.batch])
                             .float().mean().item())
        lines.append(f"- 真密钥比特准确率: **{np.mean(true_accs):.4f}**\n")
        csv_rows.append(("true_key_acc", f"{np.mean(true_accs):.4f}"))

    # ---------- B. matched-filter 校验距离与 FAR ----------
    true_d = torch.tensor([
        security.key_check_distances(stego, x_T[i:i + 1], keys_all[i], nonces[i])[0].item()
        for i in range(args.n)
    ])
    wrong_keys = security.wrong_key_population([keys_all[0]], n_random=args.n_wrong,
                                               seed=args.seed)
    wrong_d = torch.tensor([
        security.key_check_distances(stego, x_T, wk, nonces[0]).float().mean().item()
        for wk in wrong_keys
    ])
    lines += [
        "## B. matched-filter 密钥校验（无需解码器）\n",
        f"- 真密钥校验距离: mean {true_d.float().mean():.2f} / max {int(true_d.max())}"
        f" / {stego.n_check_bits}",
        f"- 错密钥距离: mean {wrong_d.mean():.2f} / std {wrong_d.std():.2f}"
        f"（期望 ≈ {stego.n_check_bits / 2}）",
    ]
    for tau in [4, 8, 12]:
        far = security.far_at_tau(wrong_d, tau)
        lines.append(f"- FAR@tau={tau}: **{far:.5f}**")
        csv_rows.append((f"far_tau{tau}", f"{far:.6f}"))
    lines.append("")

    # ---------- C+D. 多图差分攻击: 槽位定位 AUC 与残差一致性 ----------
    attack_key = keys_all[0]
    nonces_shared = [derive_nonces_from_keys([attack_key], start=0)[0]] * args.n   # 无 nonce 基线
    stegos_nn = torch.cat([
        io.hide(covers[i:i + args.batch], bits_all[i:i + args.batch],
                   [attack_key] * args.batch, args.hide_steps, args.strength,
                   nonces=[""] * args.batch)
        for i in range(0, args.n, args.batch)
    ])

    # covers 是像素图，必须编码到模型空间再做反演（隐空间模型下二者维度不同）
    xT_c = io.invert(io.to_space(covers), args.rec_steps)
    latents_none = (xT_c, io.invert(stegos_nn, args.rec_steps))
    latents_nonce = (xT_c, x_T)

    corr_none = security.residual_template_correlation(
        stego, covers, stegos_nn, [attack_key] * args.n, [""] * args.n,
        args.rec_steps, latents=latents_none)
    corr_nonce = security.residual_template_correlation(
        stego, covers, stegos, keys_all, nonces, args.rec_steps, latents=latents_nonce)
    lines += [
        "## D. 跨图残差方向一致性（|corr|, 0=完全正交, 1=图案固定可累积）\n",
        f"- 无 nonce 协议: mean **{corr_none['mean_abs_corr']:.3f}** / "
        f"max {corr_none['max_abs_corr']:.3f}",
        f"- 逐图 nonce（ours）: mean **{corr_nonce['mean_abs_corr']:.3f}** / "
        f"max {corr_nonce['max_abs_corr']:.3f}\n",
    ]
    csv_rows += [("resid_corr_mean_no_nonce", f"{corr_none['mean_abs_corr']:.4f}"),
                 ("resid_corr_mean_nonce", f"{corr_nonce['mean_abs_corr']:.4f}")]

    auc_rows = []
    for n_sub in auc_ns:
        if n_sub > args.n:
            continue
        sub_nonce = (xT_c[:n_sub], latents_nonce[1][:n_sub])
        sub_none = (xT_c[:n_sub], latents_none[1][:n_sub])
        auc_n = security.slot_detection_auc(stego, covers[:n_sub], stegos[:n_sub],
                                            bits_all[:n_sub], attack_key, nonces[:n_sub],
                                            args.rec_steps, latents=sub_nonce)
        auc_0 = security.slot_detection_auc(stego, covers[:n_sub], stegos_nn[:n_sub],
                                            bits_all[:n_sub], attack_key, [""] * n_sub,
                                            args.rec_steps, latents=sub_none)
        auc_rows.append((n_sub, auc_0, auc_n))
        csv_rows += [(f"slot_auc_no_nonce_N{n_sub}", f"{auc_0:.4f}"),
                     (f"slot_auc_nonce_N{n_sub}", f"{auc_n:.4f}")]
        print(f"slot AUC N={n_sub}: no-nonce {auc_0:.3f} | per-image nonce {auc_n:.3f}",
              flush=True)

    lines += ["## C. 多图差分攻击：槽位定位 AUC（0.5=失效；>0.5 即攻击有效）\n",
              "> ⚠️ 该指标衡量“能否把水印频点从频谱里挑出来”（存在性定位），"
              "**不是密钥恢复**。",
              "> 逐图 nonce 已知 counter 时攻击者仍可复现图案，因此它降低的是"
              "**跨图累积优势**，不等于密码学保证。\n",
              "| N | 无 nonce 协议 | 逐图 nonce（ours） |", "|---|---|---|"]
    for n_sub, a0, an in auc_rows:
        lines.append(f"| {n_sub} | {a0:.3f} | {an:.3f} |")
    lines.append("")

    # ---------- E. 盲水印检测 AUC ----------
    pres_none = security.watermark_presence_auc(xT_c, latents_none[1])
    pres_nonce = security.watermark_presence_auc(xT_c, x_T)
    lines += [
        "## E. 盲水印检测 AUC（隐写分析视角：只给图，能否区分载密图 vs 原图）\n",
        f"- 无 nonce 协议: **{pres_none['auc']:.3f}**"
        f"（direction {pres_none['direction']:+d}, "
        f"train {pres_none['train_size']} / test {pres_none['test_size']}）",
        f"- 逐图 nonce（ours）: **{pres_nonce['auc']:.3f}**"
        f"（direction {pres_nonce['direction']:+d}, "
        f"train {pres_nonce['train_size']} / test {pres_nonce['test_size']}）",
        "",
        "> 0.5 = 完全无法区分（安全）；越接近 1 说明载密图越容易被测出。",
        "> 该指标与载密图 PSNR 强相关：只要嵌入能量可观，它就接近 1。\n",
    ]
    csv_rows += [("presence_auc_no_nonce", f"{pres_none['auc']:.4f}"),
                 ("presence_auc_nonce", f"{pres_nonce['auc']:.4f}")]
    print(f"presence AUC: no-nonce {pres_none['auc']:.3f} | nonce {pres_nonce['auc']:.3f}")

    # ---------- F. 密钥空间 ----------
    ks = security.key_space_bounds(stego.n_pairs, stego.res)
    key_bits = security.effective_key_entropy_bits(None, args.key_hex_len)
    lines += [
        "## F. 密钥空间（必须区分两个量）\n",
        f"- **派生参数空间下界**（频点选择 × 8bit 相位量化）: "
        f"log2 ≈ **{ks['log2_total']:.1f} bit**",
        f"  - 环带 {ks['r']}, 可用频点 {ks['n_avail']}, 每图使用 {ks['n_pairs']} 对",
        f"  - log2(频点选择) = {ks['log2_bin_selection']:.1f}, "
        f"log2(相位量化) = {ks['log2_phases']:.1f}",
        f"- **密钥本身熵**: {args.key_hex_len} hex chars = **{key_bits:.0f} bit**"
        f"（`token_key()` 默认 16）",
        "",
        "> 攻击者枚举的是**密钥**而不是派生参数，因此真实安全强度由密钥熵封顶；"
        "参数空间下界只说明“图案有多少种可能”，不能直接当作抗暴力破解强度。\n",
    ]
    csv_rows += [("log2_param_space", f"{ks['log2_total']:.2f}"),
                 ("key_entropy_bits", f"{key_bits:.1f}")]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(args.out.replace(".md", ".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerows(csv_rows)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()

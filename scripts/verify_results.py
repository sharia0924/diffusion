"""结果复核（evaluation-only）：用现有 checkpoint 重验 ITERATION_LOG.md §4/§8 的关键数字。

不做任何训练。复核项：
  R1  VAE 往返上限            （声明 51.53 dB）
  R2  隐空间 DDPM 无嵌入往返   （声明 29.40 dB @S=50 / 38.00 dB @S=150）
  R3  端到端工作点             （声明 s=0.3 -> PSNR 28.10 / dec 0.883 / jpeg50 0.734）
       —— 用 eval_ldm_pipeline.py 原脚本独立重跑（子进程）
  R4  密钥安全性               （声明 错密钥 BER≈0.5 / FAR≈0 / AUC 差距）
  R5  "采样投影抹除小扰动"     （声明 s=0.05 与 0.1 在 S=50 下 PSNR 无差）

用法:
  python scripts/verify_results.py --n 8 --batch 8 --n-wrong 24
输出: results/verification_report.md + 控制台
"""

import argparse
import os
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from krd import security
from krd.metrics import bit_accuracy, psnr
from krd.utils import seed_everything, token_key
from scripts.eval_common import repeat_nonces
from scripts.eval_setup import eval_setup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddpm-ckpt", default="checkpoints/ddpm_latent32.pt")
    ap.add_argument("--decoder-ckpt", default="checkpoints/decoder_latent32_best.pt")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n-wrong", type=int, default=24)
    ap.add_argument("--strength", type=float, default=0.3)
    ap.add_argument("--out", default="results/verification_report.md")
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()
    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows: list[tuple[str, str, str]] = []   # (claim, re-measured, verdict)

    stego, io, loader, cfg, pixel_res = eval_setup(
        args.ddpm_ckpt, args.data_root, args.batch, device, args.decoder_ckpt)
    covers, bits, keys, nonces = [], None, [], None
    from scripts.eval_common import make_eval_inputs
    covers, bits, keys, nonces = make_eval_inputs(loader, args.n, stego.n_bits, device)

    # ---------- R1 VAE roundtrip ----------
    r1 = io.baseline_psnr(covers)
    rows.append(("R1 VAE 往返上限", "51.53 dB", f"{r1:.2f} dB"))

    # ---------- R2 latent DDPM no-embedding roundtrip ----------
    def roundtrip(steps: int) -> float:
        z = io.to_space(covers)
        z_T = io.invert(z, steps)
        z_rt = stego.sched.ddim_sample(stego.model, z_T, steps)
        return psnr(io.to_pixels(z_rt), covers)

    r2a, r2b = roundtrip(50), roundtrip(150)
    rows.append(("R2 无嵌入往返 @S=50", "29.40 dB", f"{r2a:.2f} dB"))
    rows.append(("R2 无嵌入往返 @S=150", "38.00 dB", f"{r2b:.2f} dB"))
    base50 = r2a

    # ---------- R3 end-to-end working point (their script, subprocess) ----------
    out3 = os.path.join("results", "_verify_ldm_pipeline.md")
    if not os.path.exists(out3):
        cmd = [sys.executable, "scripts/eval_ldm_pipeline.py",
               "--ddpm-ckpt", args.ddpm_ckpt, "--decoder-ckpt", args.decoder_ckpt,
               "--data-root", args.data_root, "--n", str(args.n),
               "--batch", str(args.batch), "--strengths", "0.02,0.1,0.3",
               "--out", out3]
        print("[R3]", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
    import re
    txt = open(out3, encoding="utf-8").read()
    m = re.search(r"\|\s*0\.3\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|[^|]*\|\s*([\d.]+)\s*\|"
                  r"\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|", txt)
    if m:
        rows.append(("R3 端到端 s=0.3: PSNR", "28.10 dB", f"{float(m.group(1)):.2f} dB"))
        rows.append(("R3 端到端 s=0.3: dec clean", "0.883", m.group(4)))
        rows.append(("R3 端到端 s=0.3: dec jpeg50", "0.734", m.group(5)))
    m01 = re.search(r"\|\s*0\.1\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|[^|]*\|\s*([\d.]+)\s*\|"
                    r"\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|", txt)
    if m01:
        rows.append(("R3 端到端 s=0.1: PSNR", "36.81 dB", f"{float(m01.group(1)):.2f} dB"))
        rows.append(("R3 端到端 s=0.1: dec clean", "0.641", m01.group(4)))

    # ---------- R4 key security ----------
    sg = torch.cat([io.hide(covers[i:i + args.batch], bits[i:i + args.batch],
                            keys[i:i + args.batch], cfg["hide_steps"], args.strength,
                            nonces=nonces[i:i + args.batch])
                    for i in range(0, args.n, args.batch)])
    x_T = io.invert(sg, cfg["rec_steps"])

    wrong_keys = [token_key() for _ in range(args.n_wrong)]
    bers = security.wrong_key_profile(stego, x_T, bits[:, :stego.n_bits], nonces,
                                      None, wrong_keys) if False else None
    # wrong_key_profile 需要 decoder；单独实现一遍（与 eval_key_security 相同口径）
    from krd.metrics import psnr as _p  # noqa
    dec = stego.load_decoder(args.decoder_ckpt) if hasattr(stego, "load_decoder") else None
    if dec is None:
        from krd import RingDecoder
        dec = RingDecoder.from_config(cfg).to(device)
        dec.load_state_dict(torch.load(args.decoder_ckpt, map_location=device,
                                       weights_only=True)["decoder"])
        dec.eval()
    bers = []
    for wk in wrong_keys:
        feats = stego.features_from_latents(x_T, [wk] * args.n, nonces)
        logits = stego.collapse(dec(feats))
        bers.append(1.0 - bit_accuracy(logits, bits[:, :stego.n_bits]))
    bers = np.array(bers)
    rows.append(("R4 错密钥 BER mean±std", "0.4977±0.0403", f"{bers.mean():.4f}±{bers.std():.4f}"))

    true_accs = []
    for i in range(0, args.n, args.batch):
        logits = io.recover(sg[i:i + args.batch], keys[i:i + args.batch],
                            cfg["rec_steps"], dec, nonces=nonces[i:i + args.batch])
        true_accs.append(bit_accuracy(logits, bits[i:i + args.batch, :stego.n_bits]))
    rows.append(("R4 真密钥解码 acc", "0.8984", f"{np.mean(true_accs):.4f}"))

    d_true = [int(security.key_check_distances(stego, x_T[i:i + 1], keys[i],
                                               nonces[i])[0]) for i in range(args.n)]
    d_wrong = []
    for wk in wrong_keys[:20]:
        for i in range(args.n):
            d_wrong.append(int(security.key_check_distances(
                stego, x_T[i:i + 1], wk, nonces[i])[0]))
    d_wrong = np.array(d_wrong)
    rows.append(("R4 mf 距离 真/错", "6.88 / 15.98",
                 f"{np.mean(d_true):.2f} / {d_wrong.mean():.2f}"))
    rows.append(("R4 FAR@tau=12", "0.037", f"{(d_wrong <= 12).mean():.4f}"))

    # 多图差分攻击 AUC @N=8
    sg_shared = torch.cat([io.hide(covers[i:i + args.batch], bits[i:i + args.batch],
                                   [keys[0]] * args.batch, cfg["hide_steps"], args.strength,
                                   nonces=repeat_nonces(keys[0], args.batch, shared=True))
                           for i in range(0, args.n, args.batch)])
    lat = (io.invert(io.to_space(covers), cfg["rec_steps"]), io.invert(sg, cfg["rec_steps"]))
    auc_n = security.slot_detection_auc(stego, covers, sg, bits[:, :stego.n_bits],
                                        keys[0], nonces, cfg["rec_steps"], latents=lat)
    lat0 = (lat[0], io.invert(sg_shared, cfg["rec_steps"]))
    auc_0 = security.slot_detection_auc(stego, covers, sg_shared, bits[:, :stego.n_bits],
                                        keys[0], repeat_nonces(keys[0], args.n, shared=True),
                                        cfg["rec_steps"], latents=lat0)
    rows.append(("R4 定位 AUC @N=8 (无nonce/ours)", "0.964 / 0.662",
                 f"{auc_0:.3f} / {auc_n:.3f}"))

    # ---------- R5 sampling projection erases small perturbations ----------
    r5 = {}
    for s in [0.05, 0.1, 0.3]:
        sg5 = io.hide(covers, bits, keys, 50, s, nonces=nonces)
        r5[s] = psnr(io.to_pixels(sg5), covers)
    rows.append(("R5 S=50: 往返基线", "29.40 (像素口径)", f"{base50:.2f} dB（隐空间口径）"))
    rows.append(("R5 S=50: s=0.05 vs 0.1", "应几乎相同（抹除）",
                 f"{r5[0.05]:.2f} / {r5[0.1]:.2f} dB"))
    rows.append(("R5 S=50: s=0.3 应显著下降", "—", f"{r5[0.3]:.2f} dB"))

    # ---------- 汇总 ----------
    lines = ["# 结果复核报告（evaluation-only）\n",
             f"- checkpoint: {args.ddpm_ckpt} + {args.decoder_ckpt}",
             f"- n={args.n}, wrong keys={args.n_wrong}, strength={args.strength}",
             f"- config: bpb={cfg['bpb']}, S={cfg['hide_steps']}/{cfg['rec_steps']}, "
             f"inject_mode={cfg['inject_mode']}\n",
             "| 复核项 | 声明值 | 复测值 |", "|---|---|---|"]
    for name, claim, got in rows:
        lines.append(f"| {name} | {claim} | {got} |")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

"""诊断：接收端"对齐搜索"能否救回几何攻击（旋转/平移/缩放/裁剪缩放）。

v3 鲁棒性全表（results/robustness_v3.md, E=0.3, 载密 29.7 dB）暴露的短板**全部**是
重采样类几何变换，而 JPEG/噪声/模糊/裁剪/亮度对比都在 0.95-0.99：

    cropresize0.8 0.539 | rotate5 0.691 | rotate15 0.490 | translate2px 0.434 | zoom1.1 0.707

机制很清楚：载波是"密钥指定频点 + 密钥相位"的**相干**图案，接收端靠按密钥相位
去旋转来同相叠加；一旦图像被旋转/平移/缩放，频谱就被"重排 + 加相位斜坡"，
接收端仍在**原来的频点**上读 → 相干叠加失效（不是能量被毁，而是对不齐）。

标准解法就是**同步/对齐搜索**，而且它可以是**纯接收端**的、不用重训：
- 平移 d：F(k) -> F(k)·e^{-j2π k·d/N}  ⇒ 只需给已取到的复数值乘上反向相位斜坡；
- 旋转 θ：F_rot(k) = F_orig(R_{-θ}k)  ⇒ 只需在**旋转后的坐标**上重新取点；
- 缩放 σ：频点半径 r -> r/σ ⇒ 在 r/σ 处取点。
三种都不需要重新过 VAE/UNet —— 只要对已经算好的 x_T 频谱做**索引与复数乘法**。

盲搜索的评分用**校验位**（接收端知道 key/nonce，因此知道 16 个校验位的期望值），
经典 template-based resync 做法：取让校验位偏转最大的那个 (θ, σ, d)。

本脚本对每种攻击报："不搜索"与"搜索后"的 mf 比特准确率，以及每张图选中的参数，
用来判断"对齐搜索"是否值得固化成方法的一部分。

用法（GPU）：
  python scripts/diag_sync_search.py --n 8 --strength 0.3 --rec-steps 20 \
      --attacks clean,rotate5,rotate15,translate2px,cropresize0.8,zoom1.1
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
from krd.metrics import bit_accuracy, psnr
from krd.pattern import ecc_collapse
from krd.utils import derive_nonces_from_keys, seed_everything
from scripts.eval_common import StegoIO, cifar_loader, gather_covers
from scripts.train_decoder import load_stego

EPS = 1e-8
ATTACK_FN = {
    "clean": None,
    "rotate5": ("rotate", 5.0),
    "rotate15": ("rotate", 15.0),
    "translate2px": ("translate", 2),
    "translate3px": ("translate", 3),
    "cropresize0.8": ("cropresize", 0.8),
    "zoom1.1": ("zoom", 1.1),
    "jpeg50": ("jpeg", 50),
}


def spec_gather(F_: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
    """F_ (C,H,W) -> (C, K) 在给定 (rows, cols) 上取点（越界用 0）。"""
    C, H, W = F_.shape
    valid = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
    r = rows.clamp(0, H - 1)
    c = cols.clamp(0, W - 1)
    out = F_[:, r, c]
    return out * valid[None, :].to(out.dtype)


def hypothesis_bins(bins0: torch.Tensor, res: int, theta_deg: float, sigma: float):
    """把原始频点坐标按"图像被旋转 θ、缩放 σ"反变换回去，得到应在何处取点。

    bins0 是 fftshift 坐标（中心在 res//2）。图像旋转 θ 后，原图案所在的频点
    出现在 R_{-θ}k 处；缩放 σ 后出现在 k/σ 处。
    """
    c = (res - 1) / 2.0
    k = bins0.float() - c
    t = math.radians(theta_deg)
    ct, st = math.cos(t), math.sin(t)
    # R_{-theta}
    x = ct * k[:, 1] + st * k[:, 0]
    y = -st * k[:, 1] + ct * k[:, 0]
    x = x / max(sigma, 1e-6)
    y = y / max(sigma, 1e-6)
    rows = torch.round(y + c).long()
    cols = torch.round(x + c).long()
    return rows, cols


def mf_from_fft(F_: torch.Tensor, s, params, bits_full, rows, cols, ramp=None):
    """给定频谱与取点位置，算出 mf 软值（槽位均值）与按真实期望符号的偏转。"""
    vals = spec_gather(F_, rows, cols)                      # (C, n_pairs)
    vals = vals * torch.exp(-1j * params["phases"].to(F_.device))[None, :]
    if ramp is not None:
        vals = vals * ramp[None, :]
    re = vals.real.mean(0)
    g = s.n_pairs // s.total_embed_bits
    slot = re.view(s.total_embed_bits, g).mean(-1)
    msg = ecc_collapse(slot[None, :s.msg_embed_bits], s.n_bits, s.ecc)[0]
    return slot, msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddpm-ckpt", default="checkpoints/ddpm_latent32.pt")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--strength", type=float, default=0.3)
    ap.add_argument("--hide-steps", type=int, default=150)
    ap.add_argument("--rec-steps", type=int, default=20)
    ap.add_argument("--inject-at", type=float, default=0.35)
    ap.add_argument("--n-inject", type=int, default=8)
    ap.add_argument("--pixel-res", type=int, default=64)
    ap.add_argument("--theta-list", default="-20,-15,-10,-5,0,5,10,15,20")
    ap.add_argument("--sigma-list", default="0.85,0.9,0.95,1.0,1.05,1.1,1.15")
    ap.add_argument("--shift-list", default="-3,-2,-1,0,1,2,3")
    ap.add_argument("--attacks", default="clean,rotate5,rotate15,translate2px,"
                                         "cropresize0.8,zoom1.1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/sync_search.json")
    args = ap.parse_args()
    seed_everything(0)

    dev = args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu"
    s = load_stego(args.ddpm_ckpt, dev, n_bits=8, ecc_reps=8, bins_per_bit=4,
                   n_check_bits=16, with_vae=True, inject_mode="add", r_max=None)
    io = StegoIO(s, pixel_res=args.pixel_res, n_inject=args.n_inject,
                 inject_at=args.inject_at)
    covers = gather_covers(cifar_loader(args.data_root, train=False,
                                        batch_size=args.n, resize=args.pixel_res),
                           args.n).to(dev)
    keys = ["k%d" % i for i in range(args.n)]
    nonces = derive_nonces_from_keys(keys, start=0)
    bits = torch.randint(0, 2, (args.n, s.n_bits), device=dev).float()

    thetas = [float(v) for v in args.theta_list.split(",")]
    sigmas = [float(v) for v in args.sigma_list.split(",")]
    shifts = [int(v) for v in args.shift_list.split(",")]
    # 预生成所有 (theta, sigma) 取点组合，以及每条频点的"平移相位斜坡基底"
    c = (s.res - 1) / 2.0
    print(f"配置: N={s.bpb * s.ecc} n_pairs={s.n_pairs} slots={s.total_embed_bits} "
          f"E={args.strength} S_hide={args.hide_steps}(k={round(args.hide_steps * args.inject_at)}) "
          f"S_rec={args.rec_steps} n={args.n}", flush=True)
    print(f"搜索空间: θ×σ×d = {len(thetas)}×{len(sigmas)}×{len(shifts) ** 2} = "
          f"{len(thetas) * len(sigmas) * len(shifts) ** 2}", flush=True)

    rows_out = []
    print(f"{'攻击':>14s} {'PSNR':>8s} {'无搜索':>8s} {'搜索后':>8s} "
          f"{'选中(θ,σ,dx,dy) 中位数':>26s}", flush=True)
    for atk in args.attacks.split(","):
        spec = ATTACK_FN.get(atk)
        acc_no, acc_sy, picks, p_psnr = [], [], [], []
        for i in range(args.n):
            z = io.to_space(covers[i:i + 1])
            sg = s.hide(z, bits[i:i + 1], [keys[i]], args.hide_steps,
                        args.strength / max(1, args.n_inject), nonces=[nonces[i]],
                        inject_at=args.inject_at, n_inject=args.n_inject)
            p_psnr.append(psnr(io.to_pixels(sg), covers[i:i + 1]))
            if spec is None:
                x = sg
            else:
                x = io.attack(sg, lambda t, sp=spec: apply_attack(t, sp[0], sp[1]))
            x_T = s.invert_latents(x, args.rec_steps)
            F_ = torch.fft.fftshift(torch.fft.fft2(x_T[0], norm="ortho"), dim=(-2, -1))
            params = s.params_for(keys[i], nonces[i])
            bins0 = params["bins"].to(dev)
            bits_full = s.full_bits(bits[i], keys[i], nonces[i])
            # 基准（不搜索）
            slot, msg = mf_from_fft(F_, s, params, bits_full, bins0[:, 0], bins0[:, 1])
            acc_no.append(bit_accuracy(msg[None], bits[i][None]))
            # 搜索：评分用校验位偏转（接收端已知 key/nonce -> 知道校验位期望值）
            ck = bits_full[s.msg_embed_bits:]
            best, best_score = (0.0, 1.0, 0, 0), -1e9
            k0 = bins0.float()
            for th in thetas:
                for sg_ in sigmas:
                    rr, cc = hypothesis_bins(bins0, s.res, th, sg_)
                    vals = spec_gather(F_, rr, cc)
                    vals = vals * torch.exp(-1j * params["phases"].to(dev))[None, :]
                    re = vals.real.mean(0)
                    g = s.n_pairs // s.total_embed_bits
                    slot0 = re.view(s.total_embed_bits, g).mean(-1)
                    for dx in shifts:
                        for dy in shifts:
                            ramp = torch.exp(1j * 2 * math.pi * (
                                (rr.float() - c) * dx + (cc.float() - c) * dy) / s.res)
                            sl = (vals * ramp[None, :]).real.mean(0)
                            sl = sl.view(s.total_embed_bits, g).mean(-1)
                            sc = float((sl[s.msg_embed_bits:] * (2 * ck - 1)).mean())
                            if sc > best_score:
                                best_score = sc
                                best = (th, sg_, dx, dy)
            th, sg_, dx, dy = best
            rr, cc = hypothesis_bins(bins0, s.res, th, sg_)
            ramp = torch.exp(1j * 2 * math.pi * (
                (rr.float() - c) * dx + (cc.float() - c) * dy) / s.res)
            slot, msg = mf_from_fft(F_, s, params, bits_full, rr, cc, ramp=ramp)
            acc_sy.append(bit_accuracy(msg[None], bits[i][None]))
            picks.append(best)
        a0, a1 = float(np.mean(acc_no)), float(np.mean(acc_sy))
        med = np.median(np.array(picks), axis=0)
        rows_out.append({"attack": atk, "psnr": float(np.mean(p_psnr)),
                         "acc_no_sync": a0, "acc_sync": a1, "picks": picks})
        print(f"{atk:>14s} {np.mean(p_psnr):>7.2f} {a0:>8.3f} {a1:>8.3f} "
              f"   θ={med[0]:>5.1f} σ={med[1]:.2f} d=({med[2]:.0f},{med[3]:.0f})", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows_out, f, indent=2, ensure_ascii=False)
    print(f"\nsaved -> {args.out}", flush=True)
    print("[读法] 若「搜索后」明显高于「无搜索」，说明几何短板是**同步问题**而非能量问题，"
          "应把对齐搜索固化进接收端（纯接收端改动，不需要重训解码器）。", flush=True)


if __name__ == "__main__":
    main()

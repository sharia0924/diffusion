"""验证：密钥幅度轮廓（mag_profile）+ 接收端同步（sync）能否救回几何攻击。

这是 §21 结论的直接验证，也是"明天上远程"之前必须本地跑通的那一步：
  - 注入端：`use_mag_profile=True`（逐频点幅度 = 密钥轮廓，均值归一）
  - 接收端：`estimate_alignment`（幅度轮廓定位 θ/σ + 相位斜坡精修平移）→ 取点解码
对照：同一批载密量、同一攻击，mag_profile 关/开 + sync 关/开 四种组合。

判定门槛（写进 ITERATION_LOG §22）：
  - 定位成功率 ≥ 90%（θ/σ 与真值一致，平移在 ±1px 内）；
  - rotate15 / translate2px / cropresize0.8 / zoom1.1 的 mf 准确率从 0.43-0.71
    提到 ≥ 0.85；
  - clean / jpeg50 不因 mag_profile 明显变差（能量 +0.35 dB 以内）。

用法（GPU，约 10 分钟）：
  python scripts/diag_mag_sync.py --n 8 --strength 0.3 --rec-steps 20
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from krd import sync as ksync
from krd.distortions import apply_attack
from krd.metrics import bit_accuracy, psnr, ssim
from krd.pattern import ecc_collapse
from krd.utils import derive_nonces_from_keys, seed_everything
from scripts.eval_common import StegoIO, cifar_loader, gather_covers
from scripts.train_decoder import fit_capacity, load_stego

ATTACK_FN = {
    "clean": None,
    "jpeg50": ("jpeg", 50),
    "rotate5": ("rotate", 5.0),
    "rotate15": ("rotate", 15.0),
    "translate2px": ("translate", 2),
    "translate3px": ("translate", 3),
    "cropresize0.8": ("cropresize", 0.8),
    "zoom1.1": ("zoom", 1.1),
    # 真值（用于核对定位是否命中）
    "truth": {"rotate5": (5.0, 1.0, 0, 0), "rotate15": (15.0, 1.0, 0, 0),
              "translate2px": (0.0, 1.0, 2, 2), "translate3px": (0.0, 1.0, 3, 3),
              "cropresize0.8": (0.0, 1.0, 0, 0), "zoom1.1": (0.0, 1.1, 0, 0),
              "clean": (0.0, 1.0, 0, 0)},
}


def mf_acc(s, feats, bits):
    accs = []
    for i in range(feats.shape[0]):
        g = s.n_pairs // s.total_embed_bits
        re = feats[i][:s.n_pairs].view(s.total_embed_bits, g).mean(-1)
        lg = ecc_collapse(re[None, :s.msg_embed_bits], s.n_bits, s.ecc)
        accs.append(bit_accuracy(lg, bits[i][None]))
    return float(np.mean(accs))


def build(mag_profile: bool, sync_on: bool, args, dev):
    nb, ecc, check, bpb = fit_capacity(32, args.n_bits, args.ecc_reps,
                                       args.n_check_bits, requested_bpb=2,
                                       r_max=None, maximize_bpb=True)
    s = load_stego(args.ddpm_ckpt, dev, n_bits=nb, ecc_reps=ecc, bins_per_bit=bpb,
                   n_check_bits=check, with_vae=True, inject_mode="add", r_max=None,
                   mag_profile=mag_profile, sync=sync_on)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddpm-ckpt", default="checkpoints/ddpm_latent32.pt")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--n-bits", type=int, default=8)
    ap.add_argument("--ecc-reps", type=int, default=8)
    ap.add_argument("--n-check-bits", type=int, default=16)
    ap.add_argument("--strength", type=float, default=0.3)
    ap.add_argument("--hide-steps", type=int, default=150)
    ap.add_argument("--rec-steps", type=int, default=20)
    ap.add_argument("--inject-at", type=float, default=0.35)
    ap.add_argument("--n-inject", type=int, default=8)
    ap.add_argument("--pixel-res", type=int, default=64)
    ap.add_argument("--attacks", default="clean,jpeg50,rotate15,translate2px,"
                                         "cropresize0.8,zoom1.1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/mag_sync.json")
    args = ap.parse_args()
    seed_everything(0)

    dev = args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu"
    covers = gather_covers(cifar_loader(args.data_root, train=False,
                                        batch_size=args.n, resize=args.pixel_res),
                           args.n).to(dev)
    keys = ["k%d" % i for i in range(args.n)]
    nonces = derive_nonces_from_keys(keys, start=0)

    # 同一批载密量（mag_profile 决定注入方式，所以两种注入各生成一次）
    s_plain = build(False, False, args, dev)
    s_mag = build(True, True, args, dev)
    io_plain = StegoIO(s_plain, pixel_res=args.pixel_res, inject_at=args.inject_at,
                       n_inject=args.n_inject)
    io_mag = StegoIO(s_mag, pixel_res=args.pixel_res, inject_at=args.inject_at,
                     n_inject=args.n_inject)
    print(f"{'攻击':>14s} {'PSNR 无轮廓/有轮廓':>22s} "
          f"{'mf 无轮廓无同步':>16s} {'mf 有轮廓+同步':>15s} {'定位':>6s}", flush=True)
    rows = []
    for atk in args.attacks.split(","):
        spec = ATTACK_FN.get(atk)
        acc_plain, acc_sync, hits, ps = [], [], [], [None, None]
        for i in range(args.n):
            z = io_plain.to_space(covers[i:i + 1])
            bits = torch.randint(0, 2, (1, s_plain.n_bits), device=dev).float()
            outs = []
            for s, io, mp, sy in ((s_plain, io_plain, False, False),
                                  (s_mag, io_mag, True, True)):
                sg = s.hide(z, bits, [keys[i]], args.hide_steps,
                            args.strength / max(1, args.n_inject), nonces=[nonces[i]],
                            inject_at=args.inject_at, n_inject=args.n_inject)
                if i == 0:
                    ps[1 if mp else 0] = psnr(io.to_pixels(sg), covers[i:i + 1])
                x = sg if spec is None else io.attack(
                    sg, lambda t, sp=spec: apply_attack(t, sp[0], sp[1]))
                feats = s.recover_features(x, [keys[i]], args.rec_steps,
                                           nonces=[nonces[i]], sync=sy)
                outs.append(mf_acc(s, feats, bits))
                if mp and i == 0 and atk in ATTACK_FN["truth"]:
                    x_T = s.invert_latents(x, args.rec_steps)
                    F_ = torch.fft.fftshift(torch.fft.fft2(x_T[0], norm="ortho"),
                                            dim=(-2, -1))
                    est = ksync.estimate_alignment(F_, s.params_for(keys[i], nonces[i]),
                                                   s.res)
                    t = ATTACK_FN["truth"][atk]
                    hits.append(bool(abs(est["theta"] - t[0]) <= 2.5
                                     and abs(est["sigma"] - t[1]) <= 0.06
                                     and abs(est["dx"] - t[2]) <= 1
                                     and abs(est["dy"] - t[3]) <= 1))
            acc_plain.append(outs[0])
            acc_sync.append(outs[1])
        a0, a1 = float(np.mean(acc_plain)), float(np.mean(acc_sync))
        rows.append({"attack": atk, "psnr_plain": ps[0], "psnr_mag": ps[1],
                     "acc_plain": a0, "acc_mag_sync": a1, "hit": hits})
        hp = "-" if not hits else ("命中" if all(hits) else "未命中")
        p0 = "-" if ps[0] is None else f"{ps[0]:.2f}"
        p1 = "-" if ps[1] is None else f"{ps[1]:.2f}"
        print(f"{atk:>14s} {p0:>10s} /{p1:>8s} {a0:>16.3f} {a1:>15.3f} {hp:>6s}",
              flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"\nsaved -> {args.out}", flush=True)
    print("[判定] 1) 有轮廓+同步在几何攻击上应显著高于无轮廓；"
          "2) clean/jpeg50 不应明显变差（轮廓非均匀约 +0.35 dB 能量）；"
          "3) 定位应命中真值。", flush=True)


if __name__ == "__main__":
    main()

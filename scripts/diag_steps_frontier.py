"""诊断：把隐藏/复原步数 S 拉大，值不值得？

背景（来自 §11 的差距分析）：目标 35 dB 与当前工作点的差距里，很大一块被
"DDIM 往返上限"吃掉——S=150 时往返只有 38.00 dB（无嵌入），也就是说**即使一个比特
都不藏**，S=150 也到不了 35 dB 以上的好成绩之外还有余量。而往返上限随 S 提升：
29.40 dB@50 / 34.86@100 / 38.00@150 / 40.25@200 / 41.89@250（§4）。

但"上限抬高"不等于"载密质量抬高"：步数越多，采样过程对 x_T 上小扰动的投影抹除
越强（§6 实测 strength<0.3 基本被抹平）。所以必须实测**同一注入能量下，
PSNR-准确率前沿是否随 S 左移**，而不是只看往返上限。

本脚本用 matched-filter 接收（**不需要训练解码器**），对每个 S 同时给出：
  - 往返上限（无嵌入）
  - 载密图 PSNR / SSIM
  - mf 比特准确率（clean / jpeg50）
决策规则：若"mf=0.95 处的 PSNR"随 S 上升，则 S=250 重训解码器值得（代价 1.7x 计算）；
否则 S 不是杠杆，应该去动 latent 容量/编码增益（见 §12）。

用法（CPU 即可，避免与训练抢 GPU）：
  python scripts/diag_steps_frontier.py --n 4 --steps-list 100,150,250 \
      --energies 0.3,0.45 --n-inject 8 --out results/steps_frontier.json
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from krd.distortions import apply_attack
from krd.metrics import bit_accuracy, psnr, ssim
from krd.pattern import ecc_collapse
from krd.utils import derive_nonces_from_keys, seed_everything
from scripts.eval_common import StegoIO, cifar_loader, gather_covers
from scripts.train_decoder import load_stego


def mf_acc(s, z, keys, nonces, bits, steps):
    """matched-filter 接收：不依赖训练解码器（与 diag_multi_inject 同口径）。"""
    out = []
    for i in range(z.shape[0]):
        f = s.recover_features(z[i:i + 1], [keys[i]], steps, nonces=[nonces[i]])[0]
        g = s.n_pairs // s.total_embed_bits
        re = f[:s.n_pairs].view(s.total_embed_bits, g).mean(-1)
        lg = ecc_collapse(re[None, :s.msg_embed_bits], s.n_bits, s.ecc)
        out.append(bit_accuracy(lg, bits[i][None]))
    return sum(out) / len(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddpm-ckpt", default="checkpoints/ddpm_latent32.pt")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--steps-list", default="100,150,250")
    ap.add_argument("--energies", default="0.3,0.45")
    ap.add_argument("--n-inject", type=int, default=8)
    ap.add_argument("--inject-at", type=float, default=0.35)
    ap.add_argument("--pixel-res", type=int, default=64)
    ap.add_argument("--device", default="cpu",
                    help="默认 CPU：训练在跑时不要抢 GPU")
    ap.add_argument("--out", default="results/steps_frontier.json")
    args = ap.parse_args()
    seed_everything(0)

    dev = args.device
    s = load_stego(args.ddpm_ckpt, dev, with_vae=True, inject_mode="add")
    io = StegoIO(s, pixel_res=args.pixel_res)
    covers = gather_covers(cifar_loader(args.data_root, train=False,
                                        batch_size=args.n, resize=args.pixel_res),
                           args.n).to(dev)
    keys = ["k%d" % i for i in range(args.n)]
    nonces = derive_nonces_from_keys(keys, start=0)
    bits = torch.randint(0, 2, (args.n, s.n_bits), device=dev).float()
    z = io.to_space(covers)

    steps_list = [int(v) for v in args.steps_list.split(",")]
    energies = [float(v) for v in args.energies.split(",")]
    print(f"配置: res={s.res} bpb={s.bpb} n_pairs={s.n_pairs} n={args.n} "
          f"n_inject={args.n_inject} inject_at={args.inject_at} device={dev}",
          flush=True)
    print(f"{'S':>5s} {'E':>6s} {'往返(上限)':>11s} {'PSNR':>9s} {'SSIM':>8s} "
          f"{'mf':>6s} {'mf@jpeg50':>10s}", flush=True)

    rows = []
    for S in steps_list:
        # 上限：不嵌入，纯往返
        x_rt = s.sched.ddim_sample(s.model, s.sched.ddim_invert(s.model, z, S), S)
        rt_psnr = psnr(io.to_pixels(x_rt), covers)
        for E in energies:
            st = E / max(1, args.n_inject)
            sg = s.hide(z, bits, keys, S, st, nonces=nonces,
                        inject_at=args.inject_at, n_inject=args.n_inject)
            px = io.to_pixels(sg)
            p, ss = psnr(px, covers), ssim(px, covers)
            a = mf_acc(s, sg, keys, nonces, bits, S)
            sg_j = io.attack(sg, lambda t: apply_attack(t, "jpeg", 50))
            a_j = mf_acc(s, sg_j, keys, nonces, bits, S)
            rows.append({"S": S, "E": E, "roundtrip_psnr": rt_psnr, "psnr": p,
                         "ssim": ss, "mf": a, "mf_jpeg50": a_j})
            print(f"{S:>5d} {E:>6.2f} {rt_psnr:>10.2f}dB {p:>8.2f}dB {ss:>8.4f} "
                  f"{a:>6.3f} {a_j:>10.3f}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"saved -> {args.out}", flush=True)

    # 决策提示：在可达的 mf 水平上比 PSNR
    print("\n[决策提示] 同一 E 下，若大 S 的 PSNR 不升（甚至降），说明多出来的往返上限"
          "被投影抹除吃掉了，S 不是杠杆；此时应转向 latent 容量/编码增益（§12）。")


if __name__ == "__main__":
    main()

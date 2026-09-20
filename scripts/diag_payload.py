"""算清「降载荷」到底能买多少：观测数账 + 准确率-强度曲线。

两个问题：
  Q1. 16bit -> 8bit 时，每比特观测数增加多少？（含 ECC/校验位不变的前提）
  Q2. 准确率对什么敏感——观测数，还是注入强度？（扫 strength 找 0.95 的位置）
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from krd.metrics import bit_accuracy, psnr
from krd.pattern import ecc_collapse, ring_capacity
from krd.utils import derive_nonces_from_keys, seed_everything
from scripts.eval_common import StegoIO, cifar_loader, gather_covers
from scripts.train_decoder import load_stego


def mf_acc(s, feats_all, bits):
    """对给定特征算 matched-filter 比特准确率（复用已提取特征，避免重复反演）。"""
    accs = []
    for i in range(len(bits)):
        f = feats_all[i]
        g = s.n_pairs // s.total_embed_bits
        re = f[:s.n_pairs].view(s.total_embed_bits, g).mean(-1)
        lg = ecc_collapse(re[None, :s.msg_embed_bits], s.n_bits, s.ecc)
        accs.append(bit_accuracy(lg, bits[i][None]))
    return sum(accs) / len(accs)


def main():
    seed_everything(0)
    dev = "cuda"
    print("=== Q1: 观测数账（res=32 预算 342 对/通道，但跨通道共享同一组频点）===")
    avail = ring_capacity(32)
    print(f"  单通道可用频点对 = {avail}")
    for n_bits, ecc, chk, bpb in ((16, 3, 32, 2), (16, 3, 32, 4),
                                  (8, 3, 32, 2), (8, 3, 8, 2), (8, 1, 8, 2),
                                  (4, 3, 32, 2), (4, 1, 8, 2)):
        slots = n_bits * ecc + chk
        pairs = slots * bpb
        print(f"  载荷{n_bits:>3d}bit ecc{ecc} chk{chk:>3d} bpb{bpb} -> "
              f"slots={slots:>3d} pairs={pairs:>3d} "
              f"每比特对数={pairs / n_bits:>6.2f} "
              f"{'(超预算!)' if pairs > avail else ''}")

    print("\n=== Q2: 准确率对 strength 的敏感性（bpb=4, inject_at=0.35）===")
    covers = gather_covers(cifar_loader("./data", train=False, batch_size=4, resize=64),
                           4).to(dev)
    keys = ["k1", "k2", "k3", "k4"]
    nonces = derive_nonces_from_keys(keys, start=0)
    for bpb in (2, 4):
        s = load_stego("checkpoints/ddpm_latent32.pt", dev, n_bits=16, ecc_reps=3,
                       bins_per_bit=bpb, n_check_bits=32, with_vae=True,
                       inject_mode="add")
        io = StegoIO(s, pixel_res=64)
        z = io.to_space(covers)
        bits = torch.randint(0, 2, (4, s.n_bits), device=dev).float()
        print(f"  --- bpb={s.bpb} (pairs={s.n_pairs}) ---")
        for st in (0.2, 0.3, 0.4, 0.6):
            sg = s.hide(z, bits, keys, 150, st, nonces=nonces, inject_at=0.35)
            feats = [s.recover_features(sg[i:i + 1], [keys[i]], 150,
                                        nonces=[nonces[i]])[0] for i in range(4)]
            # 训练好的解码器也给一遍（若有）
            print(f"    strength={st:<4} PSNR={psnr(io.to_pixels(sg), covers):>6.2f}dB "
                  f"mf={mf_acc(s, feats, bits):.3f}", flush=True)


if __name__ == "__main__":
    main()

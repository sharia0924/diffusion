"""核查 n_inject / inject_at / r_max 是否从各自的 decoder config 正确回填。

若两个不同训练配置的解码器跑出完全相同的 stego 指标（PSNR/mf），
说明评测端没有把新参数生效——本脚本专门查这一点。
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from krd.metrics import psnr
from krd.utils import derive_nonces_from_keys, seed_everything
from scripts.eval_common import StegoIO, cifar_loader, gather_covers
from scripts.eval_setup import eval_setup


def main():
    seed_everything(0)
    dev = "cpu"
    covers = gather_covers(cifar_loader("./data", train=False, batch_size=4, resize=64),
                           4)
    keys = ["k1", "k2", "k3", "k4"]
    nonces = derive_nonces_from_keys(keys, start=0)

    for ck in ("decoder_latent32_b8r13_best.pt", "decoder_latent32_combo_best.pt"):
        path = "checkpoints/" + ck
        if not os.path.exists(path):
            print(f"{ck}: 不存在")
            continue
        s, io, loader, cfg, pres = eval_setup("checkpoints/ddpm_latent32.pt", "./data",
                                              4, dev, path)
        print(f"\n=== {ck} ===")
        print(f"  io: inject_at={io.inject_at} n_inject={io.n_inject} "
              f"latent={io.latent}")
        print(f"  stego: res={s.res} r_max={getattr(s, 'r_max', None)} "
              f"bpb={s.bpb} n_pairs={s.n_pairs}")
        print(f"  cfg: inject_at={cfg.get('inject_at')} n_inject={cfg.get('n_inject')} "
              f"r_max={cfg.get('r_max')} n_bits={cfg.get('n_bits')} bpb={cfg.get('bpb')}")
        bits = torch.randint(0, 2, (4, s.n_bits)).float()
        sg = io.hide(covers, bits, keys, cfg.get("hide_steps", 150), 0.3, nonces=nonces)
        print(f"  -> stego latent 校验和 {float(sg.sum()):.6f}  "
              f"PSNR {psnr(io.to_pixels(sg), covers):.2f} dB")


if __name__ == "__main__":
    main()

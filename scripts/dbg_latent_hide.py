"""定位隐空间隐藏路径的 CUDA device-side assert（逐步同步 + shape 检查）。"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from krd.metrics import psnr
from krd.utils import derive_nonces_from_keys, seed_everything
from krd.vae import build_vae
from scripts.eval_common import StegoIO, cifar_loader, gather_covers
from scripts.train_decoder import load_stego


def sync(tag):
    torch.cuda.synchronize()
    print(f"  ok: {tag}", flush=True)


def main():
    seed_everything(0)
    dev = "cuda"
    stego = load_stego("checkpoints/ddpm_latent.pt", dev, with_vae=True)
    io = StegoIO(stego)
    print("io:", io.describe(), flush=True)
    print("clip_denoised:", stego.sched.clip_denoised, " res:", stego.res,
          " n_pairs:", stego.n_pairs, " slots:", stego.total_embed_bits, flush=True)

    loader = cifar_loader("./data", train=False, batch_size=4)
    covers = gather_covers(loader, 4).to(dev)
    keys = ["k1", "k2", "k3", "k4"]
    nonces = derive_nonces_from_keys(keys, start=0)
    bits = torch.randint(0, 2, (4, stego.n_bits), device=dev).float()

    print("[step 1] VAE encode", flush=True)
    z = io.to_space(covers)
    sync(f"encode -> {tuple(z.shape)} dtype={z.dtype}")

    print("[step 2] DDIM invert（隐藏端）", flush=True)
    x_T = stego.sched.ddim_invert(stego.model, z, 50)
    sync(f"invert -> {tuple(x_T.shape)}")

    print("[step 3] params_for", flush=True)
    p = stego.params_for(keys[0], nonces[0])
    b = p["bins"]
    print(f"  bins shape={tuple(b.shape)} dtype={b.dtype} device={b.device} "
          f"min={int(b.min())} max={int(b.max())}", flush=True)
    print(f"  x_T shape={tuple(x_T[0].shape)}  H={x_T.shape[1]} W={x_T.shape[2]}", flush=True)

    print("[step 4] inject_pattern（单张）", flush=True)
    from krd.pattern import inject_pattern
    b_full = stego.full_bits(bits[0], keys[0], nonces[0])
    y = inject_pattern(x_T[0], b_full, p, 1.0)
    sync(f"inject -> {tuple(y.shape)}")

    print("[step 5] ring_features", flush=True)
    from krd.pattern import ring_features
    f = ring_features(y, p)
    sync(f"features -> {tuple(f.shape)}")

    print("全部通过：隐藏路径在 CUDA 上可用", flush=True)


if __name__ == "__main__":
    main()

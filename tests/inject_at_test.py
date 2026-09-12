"""验证 inject_at 全链路贯通：StegoIO 默认值、decoder config 回填、hide 行为。"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from krd.metrics import psnr
from krd.utils import derive_nonces_from_keys
from scripts.eval_common import StegoIO, cifar_loader, gather_covers
from scripts.eval_setup import eval_setup
from scripts.train_decoder import load_stego


def main():
    dev = "cpu"
    ck = "checkpoints/decoder_latent32_best.pt"
    s, io, loader, cfg, pres = eval_setup("checkpoints/ddpm_latent32.pt", "./data", 2, dev, ck)
    print(f"[eval_setup] pixel_res={pres} latent={io.latent} "
          f"inject_at={io.inject_at} bpb={s.bpb} mode={s.inject_mode}")
    print(f"[decoder cfg] inject_at={cfg.get('inject_at') if cfg else None} "
          f"hide_steps={cfg.get('hide_steps') if cfg else None}")

    # 像素空间 StegoIO 仍是直通
    print(f"[passthrough] latent={io.latent} "
          f"to_space 直通={not io.latent}")

    # 用不同 inject_at 跑 hide，确认参数真的生效（PSNR 应随 inject_at 变化）
    stego = load_stego("checkpoints/ddpm_latent32.pt", dev, with_vae=True,
                       inject_mode="add")
    io2 = StegoIO(stego, pixel_res=64)
    covers = gather_covers(cifar_loader("./data", train=False, batch_size=2, resize=64), 2)
    keys = ["k1", "k2"]
    nonces = derive_nonces_from_keys(keys, start=0)
    bits = torch.randint(0, 2, (2, stego.n_bits)).float()
    z = io2.to_space(covers)
    print(f"[hide] {'inject_at':>10s} {'PSNR':>9s}")
    for ia in (1.0, 0.5, 0.25):
        sg = stego.hide(z, bits, keys, 20, 0.2, nonces=nonces, inject_at=ia)
        print(f"       {ia:>10.2f} {psnr(io2.to_pixels(sg), covers):>8.2f}dB")

    # StegoIO 默认注入时刻（None 时用 self.inject_at）
    sg_def = io2.hide(covers, bits, keys, 20, 0.2, nonces=nonces)
    io2.inject_at = 0.25
    sg_025 = io2.hide(covers, bits, keys, 20, 0.2, nonces=nonces)
    print(f"[StegoIO] inject_at=1.0(默认) PSNR={psnr(io2.to_pixels(sg_def), covers):.2f}dB | "
          f"inject_at=0.25 PSNR={psnr(io2.to_pixels(sg_025), covers):.2f}dB")
    assert not torch.allclose(sg_def, sg_025), "inject_at 未生效"
    print("inject_at 全链路贯通验证通过 [OK]")


if __name__ == "__main__":
    main()

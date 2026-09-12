"""核对 VAE checkpoint 的真实往返质量与评测链路实际用到的配置。"""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from krd.vae import NativeVAE, build_vae
from scripts.eval_common import cifar_loader, gather_covers
from scripts.eval_setup import eval_setup


def psnr(a, b):
    mse = ((a - b) ** 2).mean().item()
    return 10 * math.log10(4.0 / max(mse, 1e-12))


def main():
    dev = "cpu"
    ck = torch.load("checkpoints/vae32.pt", map_location=dev, weights_only=False)
    a = ck.get("args", {})
    print("=== checkpoint args ===")
    for k in ("base", "z_ch", "downsample", "ch_mults", "resize", "epochs", "kl_weight"):
        print(f"  {k} = {a.get(k)!r}")

    # 直接用存储的 args 重建模型（训练脚本的做法）
    cm = a.get("ch_mults", "1,2")
    cm = tuple(int(v) for v in cm.split(",")) if isinstance(cm, str) else tuple(cm)
    m = NativeVAE(in_ch=3, base=a.get("base", 64), z_ch=a.get("z_ch", 4),
                  downsample=a.get("downsample", 1), ch_mults=cm)
    m.load_state_dict(ck["model"])
    m.eval()
    print(f"=== 用训练 args 重建 NativeVAE ===")
    print(f"  enc_in.weight shape = {tuple(m.enc_in.weight.shape)}")

    loader = cifar_loader("./data", train=False, batch_size=8, resize=64)
    x = gather_covers(loader, 8)
    with torch.no_grad():
        z = m.encode(x)
        y = m.decode(z, target_hw=(64, 64))
    print(f"  64×64 输入 -> latent {tuple(z.shape)} -> 往返 PSNR {psnr(y, x):.2f} dB")

    loader32 = cifar_loader("./data", train=False, batch_size=8)
    x32 = gather_covers(loader32, 8)
    with torch.no_grad():
        z32 = m.encode(x32)
        y32 = m.decode(z32, target_hw=(32, 32))
    print(f"  32×32 输入 -> latent {tuple(z32.shape)} -> 往返 PSNR {psnr(y32, x32):.2f} dB")

    print("=== VAEWrapper（评测用的加载路径）===")
    v = build_vae("native", ckpt="checkpoints/vae32.pt", device=dev)
    print(f"  {v.describe()}  latent_shape(64,64)={v.latent_shape(64, 64)}")
    with torch.no_grad():
        zw = v.encode(x)
        yw = v.decode(zw, target_hw=(64, 64))
    print(f"  64×64 往返 PSNR {psnr(yw, x):.2f} dB")

    print("=== eval_setup 实际用的配置 ===")
    stego, io, loader_e, cfg, pres = eval_setup(
        "checkpoints/ddpm_latent32.pt", "./data", 4, dev,
        "checkpoints/decoder_latent32_best.pt")
    print(f"  pixel_res={pres}  stego.res={stego.res}  bpb={stego.bpb}  "
          f"n_pairs={stego.n_pairs}  latent_shape={stego.latent_shape}")
    print(f"  io.pixel_res={io.pixel_res}  vae={io.vae.describe() if io.vae else None}")
    xb = next(iter(loader_e))[0]
    print(f"  loader 取到的图尺寸 {tuple(xb.shape)}")


if __name__ == "__main__":
    main()

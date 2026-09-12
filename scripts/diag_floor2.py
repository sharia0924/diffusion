"""定位"注入后 PSNR 卡在 ~15 dB 且与 strength 无关"的真因。

三种链路对比（同一注入 x_T'）：
  A. 不采样，直接解码 x_T'        -> 注入本身的直接损伤
  B. 采样回 x_0 再解码             -> 扩散采样带来的额外损伤
  C. 采样回 x_0 -> 解码 -> 重编码  -> 再叠加 VAE 编解码往返
若 A 好而 B 崩，说明是**扩散轨迹对扰动的数值放大**；
若 A 就已经崩，说明注入本身过大。
"""

import sys

sys.path.insert(0, "/".join(__file__.split("/")[:-2]))
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(
    __import__("os").path.abspath(__file__))))

import torch

from krd.metrics import psnr
from krd.pattern import inject_pattern
from krd.utils import derive_nonces_from_keys, seed_everything
from scripts.eval_common import StegoIO, cifar_loader, gather_covers
from scripts.train_decoder import load_stego


def main():
    seed_everything(0)
    dev = "cuda"
    s = load_stego("checkpoints/ddpm_latent32.pt", dev, with_vae=True)
    io = StegoIO(s, pixel_res=64)
    covers = gather_covers(cifar_loader("./data", train=False, batch_size=4, resize=64),
                           4).to(dev)
    z = io.to_space(covers)
    keys = ["k1", "k2", "k3", "k4"]
    nonces = derive_nonces_from_keys(keys, start=0)
    bits = torch.randint(0, 2, (4, s.n_bits), device=dev).float()
    steps = 50

    with torch.no_grad():
        z_T = s.sched.ddim_invert(s.model, z, steps)
        print(f"{'strength':>9s} {'A:直接解码x_T':>14s} {'B:采样后':>10s} "
              f"{'C:B+重编码':>11s} {'latent相对MSE(B)':>16s}")
        for st in (0.0, 0.02, 0.1, 0.5):
            zs = []
            for i in range(4):
                b = s.full_bits(bits[i], keys[i], nonces[i])
                zs.append(inject_pattern(z_T[i], b, s.params_for(keys[i], nonces[i]),
                                         st, mode="add"))
            z_inj = torch.stack(zs)
            a = psnr(io.to_pixels(z_inj), covers)
            z_rec = s.sched.ddim_sample(s.model, z_inj, steps)
            b_ = psnr(io.to_pixels(z_rec), covers)
            z_re = io.to_space(io.to_pixels(z_rec))
            c = psnr(io.to_pixels(z_re), covers)
            rel = ((z_rec - z) ** 2).mean().item() / z.pow(2).mean().item()
            print(f"{st:>9} {a:>13.2f}dB {b_:>9.2f}dB {c:>10.2f}dB {rel:>16.4f}")


if __name__ == "__main__":
    main()

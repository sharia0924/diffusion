"""PoC：把注入时刻从 x_T（轨迹末端）提前到中间 t*，看扰动存活率是否提升。

逻辑：反演到 t* -> 注入 -> 从 t* 采样回 x_0。
t* 越小（越靠近 x_0），注入后剩下的采样步数越少，被流形投影抹除的机会越少，
但可承载扰动的"噪声预算"也越小，需要实测找平衡点。

指标：最终图像的 PSNR 与 latent 相对 MSE（相对未注入的往返），
以及 matched-filter（mf）准确率——mf 高说明图案确实存活到了 x_0。
"""

import sys

sys.path.insert(0, "/".join(__file__.split("/")[:-2]))
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(
    __import__("os").path.abspath(__file__))))

import torch

from krd.metrics import bit_accuracy, psnr
from krd.pattern import ecc_collapse, inject_pattern
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
    total = 150

    def mf(zx):
        accs = []
        for i in range(4):
            f = s.recover_features(zx[i:i + 1], [keys[i]], total, nonces=[nonces[i]])[0]
            g = s.n_pairs // s.total_embed_bits
            re = f[:s.n_pairs].view(s.total_embed_bits, g).mean(-1)
            lg = ecc_collapse(re[None, :s.msg_embed_bits], s.n_bits, s.ecc)
            accs.append(bit_accuracy(lg, bits[i][None]))
        return sum(accs) / len(accs)

    print(f"{'t* 比例':>8s} {'剩余步':>7s} {'PSNR':>9s} {'latent相对MSE':>14s} {'mf':>6s}")
    for frac in (1.0, 0.5, 0.25, 0.1):
        k = max(2, int(total * (1 - frac)))     # 反演到第 k 个时刻
        seq = s.sched.timestep_seq(total)
        t_star = int(seq[k]) if k < len(seq) else int(seq[-1])
        with torch.no_grad():
            z_t = s.sched.ddim_invert(s.model, z, k)          # 只反演 k 步
            zs = []
            for i in range(4):
                b = s.full_bits(bits[i], keys[i], nonces[i])
                zs.append(inject_pattern(z_t[i], b, s.params_for(keys[i], nonces[i]),
                                         0.1, mode="add"))
            z_inj = torch.stack(zs)
            z_rec = s.sched.ddim_sample(s.model, z_inj, k)     # 只采样 k 步
        rel = ((z_rec - z) ** 2).mean().item() / z.pow(2).mean().item()
        print(f"{frac:>8.2f} {k:>7d} {psnr(io.to_pixels(z_rec), covers):>8.2f}dB "
              f"{rel:>14.5f} {mf(z_rec):>6.3f}", flush=True)


if __name__ == "__main__":
    main()

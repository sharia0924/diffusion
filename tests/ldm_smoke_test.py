"""LDM 链路冒烟测试：用随机权重的小 VAE + 小扩散模型验证端到端可运行性。

不依赖任何训练结果，只验证"隐空间隐藏 -> 解码 -> 失真 -> 再编码 -> 复原"
这条链路在代码层面是通的（形状、空间桥接、指标计算）。

  python tests/ldm_smoke_test.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from krd import RingDecoder, Schedule, TrajStego, UNet
from krd.distortions import apply_attack
from krd.metrics import bit_accuracy, psnr, ssim
from krd.utils import derive_nonces_from_keys, resolve_num_workers, seed_everything
from krd.vae import NativeVAE
from scripts.eval_common import StegoIO


def check(name: str, cond: bool):
    print(f"[{'ok' if cond else 'FAIL'}] {name}")
    if not cond:
        raise AssertionError(name)


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    seed_everything(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}")

    # 1) 小 VAE（随机权重即可，验证链路形状）
    vae_model = NativeVAE(in_ch=3, base=16, z_ch=4, downsample=2,
                          ch_mults=(1, 2, 4)).to(dev).eval()
    x = torch.randn(4, 3, 32, 32, device=dev)

    class _V:
        def encode(self, t, sample=False):
            return vae_model.encode(t, sample=sample)

        def decode(self, t, target_hw=None):
            return vae_model.decode(t, target_hw=target_hw)

        def describe(self):
            return "test-vae"

    z = _V().encode(x)
    check("VAE 下采样 4x -> latent 8x8x4", tuple(z.shape) == (4, 4, 8, 8))
    rec = _V().decode(z, target_hw=(32, 32))
    check("VAE 解码回原尺寸", tuple(rec.shape) == tuple(x.shape))

    # 2) 隐空间 TrajStego + StegoIO 桥接
    #    注意：latent 8×8 的环带频点预算比像素 32×32 小得多（半平面只有 ~7 对，
    #    对比 32² 的 176 对），所以这里用小容量配置 —— 这正是 LDM 迁移必须
    #    用更高分辨率 / 更多通道的原因，见 krd.latent.latent_capacity_report。
    model = UNet(in_ch=4, base=16, ch_mults=(1, 2, 2), attn_levels=(2,)).to(dev)
    sched = Schedule(1000, device=dev)
    stego = TrajStego(model, sched, n_bits=2, ecc_reps=1, bins_per_bit=2,
                      n_check_bits=0, res=8, device=dev)
    stego.vae = _V()
    stego.latent_shape = [4, 8, 8]
    stego.pixel_res = 32
    io = StegoIO(stego)
    check("StegoIO 识别为隐空间", io.latent)
    check("to_space 产生潜变量", tuple(io.to_space(x).shape) == (4, 4, 8, 8))
    check("to_pixels 回到图像", tuple(io.to_pixels(z).shape) == tuple(x.shape))

    # 3) 隐藏 -> 复原（隐空间）
    keys = ["k1", "k2", "k3", "k4"]
    nonces = derive_nonces_from_keys(keys, start=0)
    bits = torch.randint(0, 2, (4, 2), device=dev).float()
    sg = io.hide(x, bits, keys, 4, 1.0, nonces=nonces)
    check("hide 在隐空间输出", tuple(sg.shape) == (4, 4, 8, 8))
    sg_px = io.to_pixels(sg)
    check("载密图可解码为像素图", tuple(sg_px.shape) == tuple(x.shape))
    check("载密图像素级 PSNR 有限", psnr(sg_px, x) == psnr(sg_px, x))

    # 4) 信道：像素空间失真 -> 回到隐空间
    sg_j = io.attack(sg, lambda t: apply_attack(t, "jpeg", 50))
    check("失真后回到隐空间形状不变", tuple(sg_j.shape) == (4, 4, 8, 8))

    # 5) 特征/解码器路径
    feats = io.recover_features(sg_j, keys, 4, nonces=nonces)
    check("recover_features 形状", feats.shape == (4, 2 * stego.n_pairs))
    dec = RingDecoder(feats.shape[1], stego.total_embed_bits).to(dev)
    logits = io.recover(sg_j, keys, 4, dec, nonces=nonces)
    check("recover logits 形状", logits.shape == (4, 2))
    acc = bit_accuracy(logits, bits)
    check("随机权重下准确率在 [0,1]", 0.0 <= acc <= 1.0)

    # 6) 容量报告：隐空间每比特观测数应远高于像素空间
    rep = io.capacity_report(4)
    print(f"    隐空间容量报告: {rep}")
    check("容量报告可用", rep is not None and rep["obs_per_bit_ceiling"] > 0)

    # 7) 无嵌入往返上限（VAE 直通）
    rt = io.baseline_psnr(x)
    print(f"    VAE 直通往返 PSNR（随机权重，数值无意义）: {rt:.2f} dB")
    check("baseline_psnr 可用", rt == rt)

    # 8) 像素空间下 StegoIO 必须完全直通（回归保护）
    stego_px = TrajStego(UNet(in_ch=3, base=16, ch_mults=(1, 2, 2), attn_levels=(2,)).to(dev),
                         sched, n_bits=16, ecc_reps=3, bins_per_bit=2,
                         n_check_bits=32, res=32, device=dev)
    io_px = StegoIO(stego_px)
    check("像素空间 StegoIO 为直通", not io_px.latent
          and torch.equal(io_px.to_space(x), x)
          and torch.equal(io_px.to_pixels(x), x))

    print("\nLDM 链路冒烟测试全部通过 [OK]")


if __name__ == "__main__":
    main()

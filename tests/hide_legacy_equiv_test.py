"""回归测试：`hide(n_inject=1)` 必须与"老写法"（反演到 x_T → 注入 → 采样）逐位一致。

为什么重要：项目里绝大多数已发布数字（§4 往返、§9 扫描、verification_report 的
R1–R5）都是在"末端注入"的老写法下测的。`n_inject` 改造把注入点写成"下坡轨迹的
第 len(seq)-2 个下标"，对 n=1 是否仍等价于"注入在 x_T"必须钉死，否则新老数字
不可比，而且会出现"看不出哪里变了但 R4 的校验位距离从 6.88 漂到 8.00"这种怪现象。

用法：python tests/hide_legacy_equiv_test.py [--steps 20] [--n 2]
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from krd.pattern import inject_pattern
from krd.utils import derive_nonces_from_keys, seed_everything
from scripts.eval_common import StegoIO, cifar_loader, gather_covers
from scripts.train_decoder import load_stego


def legacy_hide(stego, z, bits, keys, nonces, steps, strength, inject_at=1.0):
    """改造前的写法：整段反演到 inject_at 位置 -> 一次性注入 -> 采样回来。"""
    k = min(max(1, int(round(steps * inject_at))), steps)
    x = stego.sched.ddim_invert(stego.model, z, k)
    outs = []
    for i in range(x.shape[0]):
        params = stego.params_for(keys[i], nonces[i])
        bits_full = stego.full_bits(bits[i], keys[i], nonces[i])
        outs.append(inject_pattern(x[i], bits_full, params, float(strength),
                                   mode=stego.inject_mode))
    x = torch.stack(outs)
    return stego.sched.ddim_sample(stego.model, x, k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddpm-ckpt", default="checkpoints/ddpm_latent32.pt")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--strength", type=float, default=0.3)
    args = ap.parse_args()
    seed_everything(0)

    s = load_stego(args.ddpm_ckpt, "cpu", with_vae=True, inject_mode="add")
    io = StegoIO(s, pixel_res=64)
    covers = gather_covers(cifar_loader("./data", train=False, batch_size=args.n,
                                        resize=64), args.n)
    keys = ["k%d" % i for i in range(args.n)]
    nonces = derive_nonces_from_keys(keys, start=0)
    bits = torch.randint(0, 2, (args.n, s.n_bits)).float()
    z = io.to_space(covers)

    for ia in (1.0, 0.5, 0.35):
        a = legacy_hide(s, z, bits, keys, nonces, args.steps, args.strength, inject_at=ia)
        b = s.hide(z, bits, keys, args.steps, args.strength, nonces=nonces,
                   inject_at=ia, n_inject=1)
        d = float((a - b).abs().max())
        rel = d / float(a.abs().max().clamp(min=1e-9))
        print(f"inject_at={ia}: max|Δ|={d:.3e} 相对={rel:.2e} "
              f"{'一致' if rel < 1e-5 else '不一致 ← 新老数字不可比'}")
        if ia == 1.0:
            assert rel < 1e-5, (
                "hide(n_inject=1, inject_at=1.0) 与老写法不一致：注入时刻的语义被改了，"
                "所有以老写法测量的历史数字都需要重测或标注")
    print("hide 老写法等价性测试通过 [OK]")


if __name__ == "__main__":
    main()

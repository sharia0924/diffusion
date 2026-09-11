"""验证 pattern.py 的扁平 gather 改造：CPU/CUDA 一致、支持 latent 通道数、密钥门控仍生效。"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from krd.pattern import (ecc_collapse, inject_pattern, key_params, ring_features,
                         template_features)
from krd.utils import seed_everything


def main():
    seed_everything(0)
    has_cuda = torch.cuda.is_available()
    for shape, res, n_pairs in (((3, 32, 32), 32, 8), ((4, 16, 16), 16, 8),
                                ((4, 8, 8), 8, 6), ((6, 32, 32), 32, 8)):
        C, H, W = shape
        p = key_params("k", n_pairs, res=res, nonce="n")
        bits = torch.randint(0, 2, (n_pairs,)).float()
        x = torch.randn(*shape)
        y_cpu = inject_pattern(x, bits, p, 1.0)
        f_cpu = ring_features(y_cpu, p)
        tmpl = template_features(bits, p)
        corr_cpu = (f_cpu / f_cpu.norm() * tmpl / tmpl.norm()).sum().item()

        line = (f"{str(shape):14s} inject={tuple(y_cpu.shape)} "
                f"feats={tuple(f_cpu.shape)} corr(true)={corr_cpu:.3f}")
        if has_cuda:
            xg = x.cuda()
            pg = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in p.items()}
            bg = bits.cuda()
            y_gpu = inject_pattern(xg, bg, pg, 1.0)
            f_gpu = ring_features(y_gpu, pg)
            same_y = torch.allclose(y_cpu, y_gpu.cpu(), atol=1e-4)
            same_f = torch.allclose(f_cpu, f_gpu.cpu(), atol=1e-4)
            line += f" | CUDA 一致: inject={same_y} feats={same_f}"
        print(line)

    # 错密钥仍应去相关（密钥门控没被破坏）
    p_true = key_params("alice", 8, res=16, nonce="n")
    p_wrong = key_params("bob", 8, res=16, nonce="n")
    bits = torch.randint(0, 2, (8,)).float()
    y = inject_pattern(torch.randn(4, 16, 16), bits, p_true, 1.0)
    ft = ring_features(y, p_true)
    fw = ring_features(y, p_wrong)
    c_t = (ft / ft.norm() * template_features(bits, p_true) /
           template_features(bits, p_true).norm()).sum().item()
    c_w = (fw / fw.norm() * template_features(bits, p_wrong) /
           template_features(bits, p_wrong).norm()).sum().item()
    print(f"\n密钥门控: corr(true)={c_t:.3f}（应≈1） corr(wrong)={c_w:.3f}（应≈0）")
    assert c_t > 0.9 and abs(c_w) < 0.3, "密钥门控被破坏"
    print("pattern.py 扁平 gather 改造验证通过 [OK]")


if __name__ == "__main__":
    main()

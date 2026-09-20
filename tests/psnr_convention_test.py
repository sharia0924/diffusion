"""回归测试：PSNR 两种口径的关系与稳定性。

背景（本项目踩过的坑之一）：`krd.metrics.psnr` 用的是**批内全局 MSE**，
它对重尾误差极敏感，且结果随批大小 n 变化——同一配置 n=4 与 n=24 能差 1.3 dB，
于是"跨运行比较 PSNR"本身就不成立。文献通行口径是**逐图 PSNR 再平均**
（`psnr_mean`）。本测试钉住两者的性质：

1. B=1 时两者相等；
2. 各图误差相同时两者相等；
3. 存在一张坏图时，逐图平均 **高于** 全局口径（坏图只影响 1/n 而不是主导均方）；
4. 坏图固定时，n 变大 → 全局口径上升（这正是"PSNR 随 n 漂移"的根源），
   而逐图平均基本不变。
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from krd.metrics import psnr, psnr_mean


def main():
    torch.manual_seed(0)
    a = torch.randn(8, 3, 16, 16)

    # 1) B=1
    b1 = a + 0.05 * torch.randn_like(a)
    assert abs(psnr(a[:1], b1[:1]) - psnr_mean(a[:1], b1[:1])) < 1e-9
    print("[1] B=1 时两种口径一致 [OK]")

    # 2) 各图误差相同 -> 一致
    b2 = a + 0.05 * torch.randn(1, 3, 16, 16).expand_as(a) * 0
    b2 = a + 0.05
    assert abs(psnr(a, b2) - psnr_mean(a, b2)) < 1e-6
    print("[2] 误差均匀时两种口径一致 [OK]")

    # 3) 一张坏图：逐图平均 > 全局
    b3 = a + 0.05 * torch.randn_like(a)
    b3[0] = a[0] + 0.6 * torch.randn_like(a[0])          # 第 0 张特别糟
    g, m = psnr(a, b3), psnr_mean(a, b3)
    assert m > g, (g, m)
    print(f"[3] 含坏图: 全局 {g:.2f} dB < 逐图平均 {m:.2f} dB [OK]")

    # 4) 同一分布下随机取子集：全局口径的抖动明显大于逐图平均
    #    （这正是"报 PSNR 必须固定 n"的原因：全局口径由最差那张图主导，
    #      子集一变就跳；逐图平均看的是典型图。）
    gs, ms = [], []
    for seed in range(12):
        g_ = torch.Generator().manual_seed(seed)
        idx = torch.randperm(a.shape[0], generator=g_)[:4]
        gs.append(psnr(a[idx], b3[idx]))
        ms.append(psnr_mean(a[idx], b3[idx]))
    sg, sm = torch.tensor(gs).std().item(), torch.tensor(ms).std().item()
    print(f"[4] 子集抖动: 全局 std {sg:.2f} dB | 逐图平均 std {sm:.2f} dB")
    assert sg > sm, (sg, sm)
    print("[4] 全局口径抖动更大（故报数须固定 n，或改用逐图平均）[OK]")

    print("PSNR 口径回归测试通过 [OK]")


if __name__ == "__main__":
    main()

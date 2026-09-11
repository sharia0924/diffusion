"""容量对照表：像素空间 vs 各分辨率的隐空间（每比特可用观测数）。

  python scripts/report_capacity.py --slots 80
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from krd.latent import latent_capacity_report
from krd.pattern import default_r_min, ring_capacity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", type=int, default=80, help="嵌入槽位数（含 ECC 展开）")
    args = ap.parse_args()

    print(f"嵌入槽位数 = {args.slots}（含 ECC 展开）\n")
    print(f"{'空间':24s} {'环带对(单通道)':>14s} {'可用对(全通道)':>14s} {'obs/bit 上限':>12s}")
    print("-" * 70)
    rows = [("像素 32x32, 1 ch", (1, 32, 32))]
    for zs in [(4, 8, 8), (4, 16, 16), (4, 32, 32), (4, 64, 64), (4, 128, 128)]:
        rows.append((f"隐空间 {zs[1]}x{zs[2]}, {zs[0]} ch", zs))
    for name, zs in rows:
        r = latent_capacity_report(zs, args.slots)
        print(f"{name:24s} {r['avail_bins_single_channel']:14d} "
              f"{r['avail_pairs_all_channels']:14d} {r['obs_per_bit_ceiling']:12.2f}")

    print("\n当前配置（80 槽位 × 2 对/槽 = 160 对）下的实际每比特对数：")
    print("  像素 32x32 1ch : 160/80 = 2.00 对/比特   <- 当前工作点，功率必须开到 36% 能量")
    print("  隐空间 16x16 4ch: 160/80 = 2.00 对/比特（可用 296 对，有 1.85x 余量）")
    print("  隐空间 32x32 4ch: 160/80 = 2.00 对/比特（可用 1416 对，有 8.8x 余量）")
    print("\n结论：分辨率每翻倍，可用频点预算约翻 4 倍（∝ 分辨率²）。")
    print("      隐空间 32x32x4 的预算已等价于像素 32x32 的 8.8 倍；")
    print("      这也是把嵌入功率降下来、把载密图 PSNR 推到 30dB+ 的物质基础。")
    print("\n环带容量随分辨率（单通道，r_min 自适应）：")
    for res in (8, 16, 32, 64, 128):
        print(f"  res={res:4d}  r_min={default_r_min(res)}  可用频点对={ring_capacity(res)}")


if __name__ == "__main__":
    main()

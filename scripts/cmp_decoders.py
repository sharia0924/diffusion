"""对比所有 latent 解码器变体的配置与能力（含 inject_at / n_inject / r_max / 载荷）。

这张表是"训练/评测口径"的第一道体检：任何一个影响 hide 的参数只要在这里
对不上，评出来的数就不是这个模型的数（§11.1 的 n_inject 事故）。
"""

import glob
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    files = sorted(glob.glob("checkpoints/decoder_latent32*.pt"))
    print(f"{'文件':34s} {'载荷':>5s} {'ecc':>4s} {'bpb':>4s} {'slots':>6s} "
          f"{'inject_at':>10s} {'n_inj':>6s} {'r_max':>6s} {'S':>4s} {'mode':>8s} "
          f"{'E_eval':>7s} {'E_train':>12s}")
    print("-" * 125)
    for f in files:
        if f.endswith("_best.pt") and os.path.exists(f.replace("_best.pt", ".pt")):
            continue  # 只列主文件，避免重复
        try:
            c = torch.load(f, map_location="cpu", weights_only=True).get("config", {})
        except Exception as e:
            print(f"{os.path.basename(f):34s} 读取失败 {type(e).__name__}")
            continue
        nb = c.get("n_bits", "?")
        ecc = c.get("ecc", "?")
        chk = c.get("n_check_bits", 0)
        slots = (nb * ecc + chk) if isinstance(nb, int) and isinstance(ecc, int) else "?"
        lo, hi = c.get("strength_min"), c.get("strength_max")
        rng = (f"[{lo},{hi}]" if lo is not None and hi is not None
               else f"[{lo},{hi}]" if (lo or hi) else "-")
        print(f"{os.path.basename(f):34s} {str(nb):>5s} {str(ecc):>4s} {str(c.get('bpb')):>4s} "
              f"{str(slots):>6s} {str(c.get('inject_at')):>10s} {str(c.get('n_inject')):>6s} "
              f"{str(c.get('r_max')):>6s} {str(c.get('hide_steps')):>4s} "
              f"{str(c.get('inject_mode')):>8s} "
              f"{str(c.get('eval_strength', '-')):>7s} {rng:>12s}")
    print("\n注：E_eval 是训练内 eval/_best.pt 的挑选口径；早期 checkpoint 无此字段（硬编码 1.0）。")


if __name__ == "__main__":
    main()

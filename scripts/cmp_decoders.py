"""对比所有 latent 解码器变体的配置与能力（含 inject_at / r_max / 载荷）。"""

import glob
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    files = sorted(glob.glob("checkpoints/decoder_latent32*.pt"))
    print(f"{'文件':34s} {'载荷':>5s} {'ecc':>4s} {'bpb':>4s} {'slots':>6s} "
          f"{'inject_at':>10s} {'r_max':>6s} {'S':>4s} {'mode':>8s}")
    print("-" * 100)
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
        print(f"{os.path.basename(f):34s} {str(nb):>5s} {str(ecc):>4s} {str(c.get('bpb')):>4s} "
              f"{str(slots):>6s} {str(c.get('inject_at')):>10s} {str(c.get('r_max')):>6s} "
              f"{str(c.get('hide_steps')):>4s} {str(c.get('inject_mode')):>8s}")


if __name__ == "__main__":
    main()

"""汇总对比多次 diag_psnr_probe 结果（新旧基础模型）。"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CANDIDATES = [
    ("旧模型 60ep（远程重训, CPU n=4）", "results_remote_20260910/diag_psnr_probe.json"),
    ("新模型 300ep（本机 CUDA n=16）", "results/diag_psnr_probe_300ep.json"),
    ("当前 results/diag_psnr_probe.json", "results/diag_psnr_probe.json"),
]


def main():
    print(f"{'来源':38s} {'往返PSNR':>9s} {'往返SSIM':>9s} {'stego@1.0':>10s}")
    print("-" * 72)
    for tag, p in CANDIDATES:
        if not os.path.exists(p):
            print(f"{tag:38s} {'(缺失)':>9s}")
            continue
        d = json.load(open(p, encoding="utf-8"))
        rec = [r for r in d["rows"]
               if r["variant"] == "replace" and abs(r["strength"] - 1.0) < 1e-6]
        stego = f"{rec[0]['psnr']:.2f} dB" if rec else "-"
        print(f"{tag:38s} {d['ddim_roundtrip_psnr']:8.2f} dB "
              f"{d['ddim_roundtrip_ssim']:9.4f} {stego:>10s}")


if __name__ == "__main__":
    main()

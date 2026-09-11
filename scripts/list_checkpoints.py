"""盘点所有 checkpoint 的配置与训练进度，判断能否被后续阶段复用。

用于排查「残留的 tiny 冒烟产物污染正式训练」这类问题：
逐一打印 base / in_ch / latent_shape / resize / tiny / epoch。
"""

import glob
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FIELDS = ("base", "in_ch", "downsample", "ch_mults", "resize", "latent_shape",
          "vae_backend", "vae_ckpt", "inject_mode", "tiny",
          "n_bits", "ecc", "bpb", "n_check_bits")


def main():
    pat = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/*.pt"
    files = sorted(glob.glob(pat))
    if not files:
        print(f"没有匹配 {pat} 的文件")
        return
    print(f"{'文件':44s} {'epoch':>6s}  " + "  ".join(f"{k}" for k in FIELDS))
    print("-" * 150)
    for f in files:
        try:
            ck = torch.load(f, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"{f:44s} 读取失败: {type(e).__name__}")
            continue
        a = ck.get("args") or ck.get("config") or {}
        ep = ck.get("epoch", ck.get("epochs_done", ck.get("epochs", "?")))
        vals = []
        for k in FIELDS:
            v = a.get(k, "-")
            if isinstance(v, (list, tuple)):
                v = "x".join(str(x) for x in v)
            vals.append(f"{k}={v}" if k in ("base", "in_ch", "downsample", "resize",
                                            "tiny", "n_bits", "ecc", "bpb") else "")
        short = "  ".join(x for x in vals if x)
        print(f"{os.path.basename(f):44s} {str(ep):>6s}  {short}")
        extra = {k: a.get(k) for k in ("ch_mults", "latent_shape", "vae_ckpt",
                                       "inject_mode", "n_check_bits")
                 if a.get(k) is not None}
        if extra:
            print(f"{'':44s} {'':>6s}  {extra}")
    print("\n提示：base/in_ch/latent_shape/resize 与目标配置不一致的 checkpoint 不能复用；"
          "\n      tiny=True 的是冒烟产物，正式训练会跳过它们（resume 前有架构校验）。")


if __name__ == "__main__":
    main()

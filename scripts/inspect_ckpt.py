"""检查 checkpoint 元信息：训练进度、参数量、EMA 完整性。"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    paths = sys.argv[1:] or ["checkpoints/ddpm_cifar.pt", "checkpoints/ddpm_cifar.last.pt"]
    for p in paths:
        if not os.path.exists(p):
            print(f"{p}: 不存在")
            continue
        st = torch.load(p, map_location="cpu", weights_only=False)
        done = st.get("epochs_done", st.get("epoch", "?"))
        keys = sorted(st.keys())
        ema = st.get("ema", {})
        n_ema = sum(v.numel() for v in ema.values() if hasattr(v, "numel"))
        n_mod = sum(v.numel() for v in st.get("model", {}).values() if hasattr(v, "numel"))
        print(f"{p}")
        print(f"  epochs_done / epoch : {done}")
        print(f"  keys                : {keys}")
        print(f"  ema params          : {n_ema / 1e6:.2f} M")
        print(f"  model params        : {n_mod / 1e6:.2f} M")
        print(f"  has optimizer state : {'opt' in st}")
        a = st.get("args", {})
        if a:
            print(f"  train args          : epochs={a.get('epochs')} batch={a.get('batch_size')} "
                  f"base={a.get('base')} lr={a.get('lr')} amp={a.get('amp')}")


if __name__ == "__main__":
    main()

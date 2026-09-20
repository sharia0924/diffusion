"""给已训练的解码器 checkpoint 补写缺失的 config 字段（如 n_inject）。

背景：train_decoder 的 config 曾漏写 n_inject，导致评测端回填成 1，
与训练时（8 次注入）不一致。此处直接补写，避免重训。
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PATCHES = {
    "decoder_latent32_combo.pt": {"n_inject": 8},
    "decoder_latent32_combo_best.pt": {"n_inject": 8},
}


def main():
    for name, kv in PATCHES.items():
        p = os.path.join("checkpoints", name)
        if not os.path.exists(p):
            print(f"  {name}: 不存在，跳过")
            continue
        ck = torch.load(p, map_location="cpu", weights_only=True)
        cfg = ck.setdefault("config", {})
        changed = []
        for k, v in kv.items():
            if cfg.get(k) != v:
                cfg[k] = v
                changed.append(f"{k}: {v}")
        torch.save(ck, p)
        print(f"  {name}: 补写 {changed or '无变化'}")
        print(f"    当前 config: n_bits={cfg.get('n_bits')} bpb={cfg.get('bpb')} "
              f"r_max={cfg.get('r_max')} inject_at={cfg.get('inject_at')} "
              f"n_inject={cfg.get('n_inject')}")


if __name__ == "__main__":
    main()

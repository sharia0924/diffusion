"""给已训练的解码器 checkpoint 补写缺失的 config 字段（如 n_inject）。

背景：train_decoder 的 config 曾漏写 n_inject，导致评测端回填成 1，
与训练时（8 次注入）不一致。此处直接补写，避免重训。

⚠️ 只补**元数据**，不改权重；补写后请立刻用 scripts/check_ninject.py 核对
"配置回填后的注入口径"与训练命令行一致（inject_at / n_inject / r_max / S）。
注意：正在训练的进程会在下一次 eval 时用旧格式覆盖回来，
所以补写必须在**训练进程结束之后**执行。
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PATCHES = {
    "decoder_latent32_combo.pt": {"n_inject": 8, "eval_strength": 1.0,
                                  "strength_min": 0.15, "strength_max": 0.4},
    "decoder_latent32_combo_best.pt": {"n_inject": 8, "eval_strength": 1.0,
                                       "strength_min": 0.15, "strength_max": 0.4},
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

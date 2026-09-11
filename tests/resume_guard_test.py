"""验证续训断点的架构校验逻辑（不依赖子进程，避免 --tiny 覆盖参数导致场景失效）。

背景：用户遇到的崩溃是——`checkpoints/vae32.pt` 残留了之前 `--tiny` 试跑的断点
（base=16、downsample=2），正式参数（base=64、downsample=1）去加载它时形状全不匹配，
`load_state_dict` 直接抛 RuntimeError。现在应改为「检测到不一致 → 跳过断点、重新训练」。
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from krd.vae import NativeVAE


def arch_mismatch(saved_args: dict, cur: dict, keys) -> list[str]:
    """复刻 train_vae.py / train_ddpm.py 中的判定逻辑。"""
    mism = []
    for k in keys:
        old = saved_args.get(k)
        if old is not None and str(old) != str(cur.get(k)):
            mism.append(k)
    return mism


def main():
    keys = ("base", "z_ch", "downsample", "ch_mults")
    tiny_args = {"base": 16, "z_ch": 4, "downsample": 2, "ch_mults": "1,2,4"}
    real_args = {"base": 64, "z_ch": 4, "downsample": 1, "ch_mults": "1,2"}

    m = arch_mismatch(tiny_args, real_args, keys)
    print(f"tiny 断点 + 正式参数 -> 检出不一致: {m}")
    assert set(m) == {"base", "downsample", "ch_mults"}, m

    m2 = arch_mismatch(real_args, real_args, keys)
    print(f"同架构断点 -> 检出不一致: {m2}（应为空）")
    assert m2 == []

    # 交叉验证：两个架构的 state_dict 确实不兼容（证明校验是必要的）
    a = NativeVAE(base=16, z_ch=4, downsample=2, ch_mults=(1, 2, 4))
    b = NativeVAE(base=64, z_ch=4, downsample=1, ch_mults=(1, 2))
    try:
        b.load_state_dict(a.state_dict())
        raise AssertionError("两个架构居然兼容，校验没有必要？")
    except RuntimeError as e:
        first = str(e).splitlines()[1] if len(str(e).splitlines()) > 1 else str(e)[:60]
        print(f"跨架构加载确实失败（校验必要）: {first.strip()[:80]}")

    # 反向确认：同架构可加载
    c = NativeVAE(base=16, z_ch=4, downsample=2, ch_mults=(1, 2, 4))
    c.load_state_dict(a.state_dict())
    print("同架构加载成功")

    print("续训架构校验逻辑验证通过 [OK]")


if __name__ == "__main__":
    main()

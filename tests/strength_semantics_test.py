"""回归测试：StegoIO.hide 的 strength 语义 = **总注入能量**。

背景（真实的 bug，不是假想）：`n_inject=n` 会把一次注入拆成 n 次、每次能量
strength/n（见 krd/stego.py）。训练端 `train_decoder` 自己做了除法，而评测端
`StegoIO.hide` 曾把 strength 原样传下去，于是评测时总能量被放大 n 倍：
n=8、strength=0.3 时载密图 PSNR 从 ~30 dB 崩到 6.02 dB，而准确率看起来"还行"，
极易被误读成"多步注入没用"。本测试用假 stego 直接检查传参，不需要 GPU/扩散模型。
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.eval_common import StegoIO


class RecordingStego:
    """只记录 hide 收到的参数，不做任何计算。"""

    pixel_res = 32

    def __init__(self):
        self.calls = []

    def hide(self, z, bits, keys, steps, strength=None, nonces=None,
             inject_at=None, n_inject=None):
        self.calls.append({"strength": strength, "inject_at": inject_at,
                           "n_inject": n_inject, "steps": steps})
        return z


COVERS = torch.zeros(2, 3, 32, 32)
BITS = torch.zeros(2, 16)
KEYS = ["k1", "k2"]


def main():
    # 1) n_inject 与 inject_at 从实例值回填，strength 被均摊
    fs = RecordingStego()
    io = StegoIO(fs, pixel_res=32, inject_at=0.35, n_inject=8)
    io.hide(COVERS, BITS, KEYS, 150, 0.3)
    c = fs.calls[-1]
    assert abs(c["strength"] - 0.3 / 8) < 1e-12, c
    assert c["n_inject"] == 8 and abs(c["inject_at"] - 0.35) < 1e-12, c
    assert c["steps"] == 150, c
    print(f"[1] n_inject=8 均摊: 总能量 0.3 -> 单次 {c['strength']:.6f} [OK]")

    # 2) 总能量守恒：无论拆成几次，n * 单次能量 都是 0.3
    for n in (1, 2, 3, 8, 16):
        fs = RecordingStego()
        io = StegoIO(fs, pixel_res=32, n_inject=n)
        io.hide(COVERS, BITS, KEYS, 150, 0.3)
        total = fs.calls[-1]["strength"] * n
        assert abs(total - 0.3) < 1e-12, (n, fs.calls[-1])
        print(f"[2] n={n:>2d} 单次 {fs.calls[-1]['strength']:.6f} 总能量 {total:.6f} [OK]")

    # 3) n_inject=1（老 checkpoint 的默认值）必须与历史行为逐位一致
    fs = RecordingStego()
    io = StegoIO(fs, pixel_res=32, n_inject=1)
    io.hide(COVERS, BITS, KEYS, 150, 0.3)
    assert fs.calls[-1]["strength"] == 0.3, fs.calls[-1]
    print("[3] n_inject=1 向后兼容: strength 原样传递 [OK]")

    # 4) 张量 strength（每样本不同强度）也要被均摊，保持张量类型
    fs = RecordingStego()
    io = StegoIO(fs, pixel_res=32, n_inject=4)
    st = torch.tensor([0.2, 0.4])
    io.hide(COVERS, BITS, KEYS, 150, st)
    got = fs.calls[-1]["strength"]
    assert torch.is_tensor(got), got
    assert torch.allclose(got, st / 4), got
    print(f"[4] 张量 strength 均摊: {st.tolist()} -> {got.tolist()} [OK]")

    # 5) 调用级覆盖参数优先于实例值
    fs = RecordingStego()
    io = StegoIO(fs, pixel_res=32, inject_at=0.35, n_inject=8)
    io.hide(COVERS, BITS, KEYS, 150, 0.4, inject_at=1.0, n_inject=2)
    c = fs.calls[-1]
    assert c["n_inject"] == 2 and abs(c["inject_at"] - 1.0) < 1e-12, c
    assert abs(c["strength"] - 0.2) < 1e-12, c
    print("[5] 显式覆盖 inject_at/n_inject 生效 [OK]")

    print("strength 语义回归测试通过 [OK]")


if __name__ == "__main__":
    main()

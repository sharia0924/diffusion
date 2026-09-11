"""二次重构：把 eval 脚本里的 stego.* 调用改为 StegoIO 桥接，并在像素空间算指标。

隐空间模型下：
  - stego.hide/recover/recover_features/invert_latents/regeneration_attack
    必须换成 io.* （io.hide 会先把像素 cover 编码成潜变量）；
  - 失真（JPEG/噪声/几何）定义在像素上，必须走 io.attack；
  - PSNR/SSIM 要在 io.to_pixels(...) 之后计算。
"""

import io
import os
import re
import sys

FILES = ["eval_robustness.py", "eval_key_security.py", "eval_steps_grid.py",
         "eval_step_mismatch.py", "eval_regen.py", "plot_frontier.py"]

# 顺序敏感：先长后短，避免 stego.recover 命中 stego.recover_features 的前缀
SUBS = [
    (r"\bstego\.recover_features\(", "io.recover_features("),
    (r"\bstego\.regeneration_attack\(", "io.regeneration_attack("),
    (r"\bstego\.invert_latents\(", "io.invert("),
    (r"\bstego\.recover\(", "io.recover("),
    (r"\bstego\.hide\(", "io.hide("),
]


def patch(path: str) -> int:
    src = io.open(path, encoding="utf-8").read()
    n = 0
    for pat, rep in SUBS:
        src, k = re.subn(pat, rep, src)
        n += k

    # 失真调用：apply_attack(x, ...) -> io.attack(x, lambda t: apply_attack(t, ...))
    if "apply_attack(" in src and "io.attack(" not in src:
        src, k = re.subn(
            r"apply_attack\(([A-Za-z_][A-Za-z0-9_\[\]:+ ]*?), (atk\[0\]|atk|name), "
            r"(atk\[1\]|param)\)",
            r"io.attack(\1, lambda t: apply_attack(t, \2, \3))", src)
        n += k

    if n:
        io.open(path, "w", encoding="utf-8", newline="\n").write(src)
    return n


def main():
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "scripts")
    for f in FILES:
        p = os.path.join(base, f)
        if os.path.exists(p):
            print(f"  {f}: {patch(p)} 处替换")


if __name__ == "__main__":
    sys.exit(main())

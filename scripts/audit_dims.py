"""静态审计：扫描「尺寸/配置未贯通」类隐患（同类 bug 的批量排查）。

检查项：
  A. 硬编码的像素尺寸推断（args.get("resize") or 32 / size=32 / size: int = 32）
  B. 直接使用 dataloader/数据集但没带 resize 的地方
  C. pattern 调用点是否可能发生 res 与张量尺寸不一致
  D. 默认参数与 checkpoint 配置未对齐（如 inject_mode 默认 replace 而训练用 add）
"""

import io
import os
import re
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(BASE, "scripts")
KRD = os.path.join(BASE, "krd")

PATTERNS = [
    (r'args\.get\(["\']resize["\'][^)]*\bor\s+32\b', "A 硬编码回退 32"),
    (r'latent_res["\'],\s*32\)', "A latent_res 回退 32"),
    (r'default=32\b', "A 默认尺寸 32"),
    (r'\bsize\s*[:=]\s*32\b', "A 硬编码 size=32"),
    (r'target_hw=\(32,\s*32\)', "A 硬编码 target_hw 32"),
    (r'datasets\.CIFAR10\(', "B 直接构造数据集（需确认是否带 resize）"),
    (r'transforms\.ToTensor\(\)', "B 变换链（需确认前面有 Resize）"),
    (r'load_stego\(' , "C 加载模型（需确认 res/inject_mode 来源）"),
    (r'stego\.hide\(|stego\.recover\(', "C 直接调用 stego（隐空间应走 StegoIO）"),
    (r'default="replace"|or "replace"', "D inject_mode 默认 replace（隐空间应为 add）"),
]


def scan(path: str):
    src = io.open(path, encoding="utf-8", errors="replace").read()
    lines = src.splitlines()
    hits = []
    for pat, tag in PATTERNS:
        for m in re.finditer(pat, src):
            ln = src[:m.start()].count("\n") + 1
            hits.append((tag, ln, lines[ln - 1].strip()[:100]))
    return hits


def main():
    targets = []
    for d in (SCRIPTS, KRD):
        for f in sorted(os.listdir(d)):
            if f.endswith(".py") and not f.startswith("_"):
                targets.append(os.path.join(d, f))

    total = 0
    for p in targets:
        hits = scan(p)
        if not hits:
            continue
        print(f"\n=== {os.path.relpath(p, BASE)} ===")
        for tag, ln, line in hits:
            print(f"  [{tag}] {ln}: {line}")
            total += 1
    print(f"\n合计 {total} 处待人工确认；A/D 类多为真实隐患，B/C 类需逐个确认。")


if __name__ == "__main__":
    main()

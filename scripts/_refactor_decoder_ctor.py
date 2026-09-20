"""一次性机械改写：把所有"从 cfg 手搓 RingDecoder"的地方换成 RingDecoder.from_config(cfg)。

动机：`mf_residual` 是写进 config 的架构开关。若评测脚本仍按
`RingDecoder(2*n_pairs, total)` 构造，带残差路径的 checkpoint 会被装进一个
**没有残差路径**的网络里（state_dict 恰好兼容，因为残差矩阵是非持久 buffer），
结果静默错误——这正是本项目反复踩到的"口径漂移"类 bug。改为统一入口后，
架构由 config 决定，漏字段会在构造时报错而不是悄悄跑出错误数字。

用法: python scripts/_refactor_decoder_ctor.py [--apply]
不带 --apply 只打印将要发生的替换。
"""

import argparse
import glob
import re
import sys

PAT = re.compile(
    r'RingDecoder\(\s*2 \* cfg\["n_pairs"\],\s*'
    r'cfg\["n_bits"\] \* cfg\["ecc"\] \+ cfg(?:\.get\("n_check_bits", 0\)|\["n_check_bits"\])\s*\)'
)
REPL = "RingDecoder.from_config(cfg)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    files = sorted(glob.glob("scripts/*.py")) + sorted(glob.glob("tests/*.py"))
    total = 0
    for f in files:
        # newline="" 读写：保留原文件的 CRLF/LF，不做换行转换
        with open(f, encoding="utf-8", newline="") as fh:
            src = fh.read()
        new, n = PAT.subn(REPL, src)
        if n:
            total += n
            print(f"{f}: {n} 处")
            if args.apply:
                with open(f, "w", encoding="utf-8", newline="") as fh:
                    fh.write(new)
    print(f"共 {total} 处" + ("（已写入）" if args.apply else "（预览，未写入）"))
    if args.apply and total == 0:
        sys.exit("没有匹配到任何构造点，检查正则")


if __name__ == "__main__":
    main()

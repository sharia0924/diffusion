"""验证 key_space_bounds 在各分辨率下不再抛错（隐空间 16×16 只有 70 对）。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from krd.security import key_space_bounds


def main():
    for res, np_req in ((32, 160), (16, 70), (16, 160), (8, 12)):
        r = key_space_bounds(np_req, res=res)
        print(f"res={res:3d} 请求 {np_req:4d} 对 -> 有效 {r['n_pairs']:4d} "
              f"可用 {r['n_avail']:4d} 环带 {r['r']} log2={r['log2_total']:.1f} bit")
    print("key_space_bounds 自适应验证通过 [OK]")


if __name__ == "__main__":
    main()

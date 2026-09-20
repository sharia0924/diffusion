"""回归护栏：所有评测脚本都必须通过 `eval_setup` 构造 StegoIO。

为什么需要这个测试（都是真实踩过的坑，同一类 bug 出现了四次）：
  1. `train_decoder` 漏写 n_inject 到 config → 评测回填成 1；
  2. `StegoIO.hide` 没按 n_inject 均摊 strength → 总能量放大 8 倍（PSNR 30→6 dB）；
  3. `train_decoder.evaluate()` 硬编码 strength=1.0 → _best.pt 按错误口径挑；
  4. `eval_strength_sweep.py` / `eval_ldm_pipeline.py` 手搓
     `StegoIO(stego, pixel_res=...)`，**完全绕过** inject_at/n_inject 回填 ——
     于是"按 inject_at=0.35, n_inject=8 训练的模型"在评测时被当成
     inject_at=1.0, n_inject=1 跑，扫描表上一点异常都看不出来。

这类 bug 的共同特征：**指标看起来正常，只是不是这个模型的数**。
唯一可靠的防线是"只有一条构造路径"，并把它钉成测试。
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
# 允许直接构造 StegoIO 的文件：eval_common 定义它，eval_setup 是唯一出口；
# diag_* 诊断脚本要按实验意图逐次指定 r_max/参数，可自行构造。
EXEMPT = {"eval_common.py", "eval_setup.py"}


def main():
    files = sorted(f for f in os.listdir(SCRIPTS)
                   if f.startswith("eval_") and f.endswith(".py") and f not in EXEMPT)
    assert files, "没找到任何 eval_*.py，路径不对？"

    bad_ctor, bad_direct, no_setup, bad_dec = [], [], [], []
    for f in files:
        src = open(os.path.join(SCRIPTS, f), encoding="utf-8").read()
        code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
        if re.search(r"\bStegoIO\s*\(", code):
            bad_ctor.append(f)
        # 绕过 io.hide 直接调 stego.hide 也会漏掉 strength/n_inject 分摊
        if re.search(r"\bstego\.hide\s*\(", code):
            bad_direct.append(f)
        if "eval_setup(" not in code:
            no_setup.append(f)
        # 解码头也必须从 config 构造（mf_residual 这类架构开关写在 config 里；
        # 手搓构造会把带残差路径的权重装进没有残差路径的网络，且 state_dict 恰好兼容）
        if re.search(r"\bRingDecoder\s*\(", code):
            bad_dec.append(f)

    print(f"检查 {len(files)} 个评测脚本：{', '.join(files)}")
    for name, lst, why in (("直接构造 StegoIO", bad_ctor,
                            "必须改用 eval_setup(...)，否则 inject_at/n_inject/r_max 不会回填"),
                           ("绕过 io.hide 调用 stego.hide", bad_direct,
                            "必须走 io.hide(...)，否则 strength 不按 n_inject 均摊"),
                           ("未调用 eval_setup", no_setup,
                            "评测脚本必须通过 eval_setup 取得 (stego, io, loader, cfg, res)"),
                           ("直接构造 RingDecoder", bad_dec,
                            "必须用 RingDecoder.from_config(cfg)，否则架构开关（mf_residual）会丢")):
        if lst:
            print(f"[FAIL] {name}: {', '.join(lst)}\n       {why}")
        else:
            print(f"[OK] 无脚本{name}")

    assert not bad_ctor, f"有评测脚本绕过 eval_setup 构造 StegoIO: {bad_ctor}"
    assert not bad_direct, f"有评测脚本直接调用 stego.hide: {bad_direct}"
    assert not no_setup, f"有评测脚本没调用 eval_setup: {no_setup}"
    assert not bad_dec, f"有评测脚本绕过 from_config 构造 RingDecoder: {bad_dec}"
    print("评测工作点护栏测试通过 [OK]")


if __name__ == "__main__":
    main()

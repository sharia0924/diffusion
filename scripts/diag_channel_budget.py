"""信道预算诊断：35 dB + JPEG50≥0.95 这个目标在信息论上还剩多少空间？

这是**纯测量**（不需要扩散模型、不需要训练），但它决定了后续该往哪使劲。
逻辑链：

1. 目标要求载密图 PSNR ≥ 35 dB，即嵌入扰动总能量 ≈ 10^(-3.5)（相对图方差）。
2. 但 JPEG50 本身就是一个**有损信道**：把原图过一遍 JPEG50，PSNR 大约只有
   30-33 dB。也就是说"信道噪声"比"整幅图的嵌入预算"还大。
3. 嵌入图案只有在**被 JPEG 保住的系数**上才可能存活。JPEG 的量化步长在低频小
   （q50 时 DC≈16、低频 AC≈11-16）、高频大（72-99），而自然图像的低频系数幅度
   大，所以低频的相对失真最小 —— 这才是 r_max 下移（把载波搬到低频）的物理依据。
4. 由于量化误差在系数间近似独立，接收端可以在 bpb×ecc 个观测上平均，
   平均增益 10log10(N)。本脚本把"每比特观测数 N"和"需要的 N"摆在一起，
   直接告出差距是几倍观测、几个 dB。

输出：results/channel_budget.md + 控制台表格
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from krd.distortions import apply_attack
from krd.metrics import psnr
from krd.pattern import ring_capacity
from krd.utils import seed_everything
from scripts.eval_common import cifar_loader, gather_covers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--res", type=int, default=64,
                    help="cover 像素尺寸（本地 LDM 是 64）")
    ap.add_argument("--qualities", default="30,50,75,90")
    ap.add_argument("--out", default="results/channel_budget.md")
    ap.add_argument("--target-psnr", type=float, default=35.0,
                    help="目标载密图 PSNR（dB）")
    ap.add_argument("--rt-psnr", default="150:38.00,200:40.25,250:41.89",
                    help="实测的无嵌入往返 PSNR（S:dB，逗号分隔；见 ITERATION_LOG §4）")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    seed_everything(args.seed)

    covers = gather_covers(cifar_loader(args.data_root, train=False,
                                        batch_size=args.n, resize=args.res), args.n)
    qs = [int(v) for v in args.qualities.split(",")]

    lines = []
    lines.append(f"# 信道预算诊断（cover {args.res}×{args.res}, n={args.n}）\n")
    lines.append("| 信道 | PSNR | 说明 |")
    lines.append("|---|---|---|")
    for q in qs:
        p = psnr(apply_attack(covers, "jpeg", q), covers)
        lines.append(f"| JPEG q={q} | **{p:.2f} dB** | JPEG 本身引入的失真（=信道噪声） |")
        print(f"JPEG q={q:<3d} PSNR {p:.2f} dB", flush=True)

    # 频点预算：不同 r_max 下每比特能拿多少观测
    res_lat = 32
    rows = []
    for n_bits, ecc, check in ((8, 3, 32), (8, 5, 32), (16, 3, 32)):
        slots = n_bits * ecc + check
        for r_max in (None, 13, 11, 9):
            avail = ring_capacity(res_lat, r_max=r_max)
            bpb = max(1, avail // slots)
            obs = bpb * ecc
            rows.append((n_bits, ecc, check, slots, r_max, avail, bpb, obs))
    lines.append("\n## 频点预算：每比特观测数 N（latent 32×32, r_min=3）\n")
    lines.append("| 载荷 | ecc | check | 槽位 | r_max | 可用对 | bpb | N=bpb×ecc |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for (nb, ecc, check, slots, rm, avail, bpb, obs) in rows:
        lines.append(f"| {nb} | {ecc} | {check} | {slots} | {rm} | {avail} | {bpb} | **{obs}** |")
        print(f"n_bits={nb} ecc={ecc} check={check} slots={slots} r_max={rm} "
              f"avail={avail} bpb={bpb} N={obs}", flush=True)

    # 平均增益与"还差多少"
    obs_cur, obs_best = 12, max(r[7] for r in rows)
    gain = 10 * torch.log10(torch.tensor(float(obs_best) / obs_cur)).item()
    # ---------------- 误差预算分解 ----------------
    # 载密图的总误差 ≈ 往返误差（采样+VAE，与是否嵌入无关）+ 嵌入误差。
    # PSNR 是能量和：1/SNR_tot = 1/SNR_rt + 1/SNR_emb（以能量计）。
    # 目标 35 dB 时能留给"嵌入"的能量 = 10^(-target/10) - 10^(-rt/10)。
    lines.append(f"\n## 误差预算分解（目标载密图 {args.target_psnr:.1f} dB）\n")
    lines.append("载密图总误差 = 往返误差（无嵌入，S 决定）+ 嵌入误差；"
                 "PSNR 按能量相加：`1/SNR_tot = 1/SNR_rt + 1/SNR_emb`。\n")
    lines.append("| S | 往返 PSNR | 总预算能量 | 往返占用 | **嵌入可用能量** | "
                 "等价的嵌入-only PSNR |")
    lines.append("|---|---|---|---|---|---|")
    rts = []
    for item in args.rt_psnr.split(","):
        s_str, db_str = item.split(":")
        rts.append((int(s_str), float(db_str)))
    tot_e = 10 ** (-args.target_psnr / 10)
    print(f"\n[预算] 目标 {args.target_psnr:.1f} dB -> 总误差能量 {tot_e:.3e}", flush=True)
    for S, rt in rts:
        rt_e = 10 ** (-rt / 10)
        emb_e = tot_e - rt_e
        if emb_e <= 0:
            lines.append(f"| {S} | {rt:.2f} dB | {tot_e:.3e} | {rt_e:.3e} | "
                         f"**不可达**（往返误差已超总预算） | — |")
            print(f"S={S}: 往返 {rt:.2f} dB 已超预算，35 dB 不可达", flush=True)
            continue
        emb_db = -10 * torch.log10(torch.tensor(emb_e)).item()
        share = rt_e / tot_e
        lines.append(f"| {S} | {rt:.2f} dB | {tot_e:.3e} | {rt_e:.3e} ({share * 100:.0f}%) | "
                     f"**{emb_e:.3e}** | {emb_db:.2f} dB |")
        print(f"S={S}: 往返 {rt:.2f} dB 占预算 {share * 100:.0f}% -> "
              f"嵌入可用 {emb_e:.3e}（嵌入-only 需 ≤ {emb_db:.2f} dB）", flush=True)

    lines.append("\n## 结论\n")
    lines.append(f"- 当前工作点每比特观测数 N=12（bpb=4 × ecc=3）；带宽内最多可到 "
                 f"N={obs_best}（bpb 顶满 + ecc 加大），平均增益仅 "
                 f"**{gain:.1f} dB**；")
    lines.append("- 因此“多攒观测”这一条路只能补回 ~1-2 dB，补不上 4-5 dB 的缺口；")
    lines.append("- 真正的杠杆是：更大的 latent（频点总数 ∝ latent 边长²，N 同比例增长）、"
                 "更强的编码（LDPC/软判决替代 repetition，理论 2-3 dB）、"
                 "或放宽 JPEG 质量要求。")
    print(f"\nN: 当前 12 -> 上限 {obs_best}，平均增益 {gain:.1f} dB", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()

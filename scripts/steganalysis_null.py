"""关键实验（kill/go）：**正确零假设下的隐写分析**——我们的载波真的不可检测吗？

为什么必须重做：项目此前的 `presence AUC 0.562` 是用
`krd.security.watermark_presence_auc` 测的，而它是
  - **配对**的（`te_s - te_c`，即减去了 cover），
  - **线性**的（均值差分方向），
  - 且零假设是"原图 cover"。
真实隐写分析里攻击者：
  - 拿不到 cover（只能看到**独立同分布的一堆图**）；
  - 用的是**学习的非线性检测器**；
  - 零假设是"**同一生成器/同一管线的无嵌入输出**"（这里是 `D(E(cover))`，
    而不是原图——因为如果服务对未嵌入图也走 VAE 重建，那才是正确的 null）。

本脚本的协议（对齐隐写分析社区惯例）：
  1. cover 划分 train/test，且**两类使用不相交的 cover**（消除配对捷径）；
  2. clean 类 = `D(E(cover_clean))`，stego 类 = `D(E(cover_stego) + δ)`
     （同一 VAE、同一分辨率、同一 CIFAR 分布，只差载荷）；
  3. 检测器：(a) 无训练的**频谱统计 + 逻辑回归**；(b) 从零训练的**小 CNN**；
  4. 指标：AUC、TPR@1%FPR（含 bootstrap 置信区间）；
  5. **阳性对照**：把载荷放大到 ~22–26 dB，检测器必须能检出；
     否则说明实验本身有问题（而不是"我们不可检测"）。
  6. 顺带做"第二格（阈值有效性）"的第一次测量：wrong-key 校验距离的**经验零分布**
     与其二项分布假设的对比（含"同图内相关性导致有效样本量下降"的量化）。

用法（本机 4GB 足够，全流程无需扩散模型）：
  python scripts/steganalysis_null.py --n-train 1500 --n-test 800 \
      --strengths 0.4,1.6 --out results/steganalysis_null.json
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from krd.metrics import psnr
from krd.pattern import check_bits, ecc_collapse, ecc_encode, inject_pattern, key_params
from krd.utils import derive_nonces_from_keys, seed_everything, token_key
from krd.vae import build_vae
from scripts.eval_common import cifar_loader, gather_covers

RES_LAT = 32
N_BITS, ECC, N_CHK = 4, 16, 16
TOTAL_SLOTS = N_BITS * ECC + N_CHK
BPB = 4
N_PAIRS = TOTAL_SLOTS * BPB
PIX = 64


def build_params(key, nonce):
    p = key_params(key, N_PAIRS, RES_LAT, nonce=nonce)
    return {"bins": p["bins"], "phases": p["phases"], "base_mag": p["base_mag"]}


def full_bits(bits, key, nonce):
    return torch.cat([ecc_encode(bits, ECC), check_bits(key, nonce, N_CHK).to(bits.device)])


@torch.no_grad()
def make_sets(vae, covers, keys, nonces, strength, dev, need_clean=True):
    """返回 (stego_images, clean_images 或 None, psnr_list)。

    clean_images 在这里**不是**隐写分析的零假设类（那类来自另一批不相交的 cover，
    在 main 里单独构造）；这里只在需要配对 sanity check 时才解码。
    """
    z = vae.encode(covers, sample=False)
    stegos, cleans, psnrs = [], [], []
    for i in range(covers.shape[0]):
        params = build_params(keys[i], nonces[i])
        bits = torch.randint(0, 2, (N_BITS,), device=dev).float()
        delta_z = inject_pattern(z[i], full_bits(bits, keys[i], nonces[i]), params,
                                 strength, mode="add")
        x_s = vae.decode(delta_z[None], target_hw=(PIX, PIX))[0]
        stegos.append(x_s)
        if need_clean:
            cleans.append(vae.decode(z[i][None], target_hw=(PIX, PIX))[0])
        psnrs.append(float(psnr(x_s[None].clamp(-1, 1), covers[i][None])))
    return torch.stack(stegos), (torch.stack(cleans) if need_clean else None), psnrs


# ---------------- 检测器 1：频谱统计 + 逻辑回归（无训练深度） ----------------
@torch.no_grad()
def spectral_features(x):
    """x (N,3,64,64) -> (N, 12)：径向频带能量 + 高频残差统计。"""
    F = torch.fft.fftshift(torch.fft.fft2(x.mean(1), norm="ortho"), dim=(-2, -1)).abs()
    H = F.shape[-1]
    c = (H - 1) / 2.0
    yy, xx = torch.meshgrid(torch.arange(H, device=x.device) - c,
                            torch.arange(H, device=x.device) - c, indexing="ij")
    r = (yy ** 2 + xx ** 2).sqrt()
    feats = []
    edges = [0, 4, 8, 12, 16, 22, 28, 33]
    for a, b in zip(edges[:-1], edges[1:]):
        m = (r >= a) & (r < b)
        if m.sum() == 0:
            feats.append(torch.zeros(x.shape[0], device=x.device))
        else:
            feats.append(F[:, m].mean(1))
    # 高频残差（相邻像素差的幅度）统计
    hp = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean((1, 2, 3))
    hp2 = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean((1, 2, 3))
    feats += [hp, hp2]
    return torch.stack(feats, dim=1)


def fit_logreg(Xtr, ytr, Xte, yte, steps=800, lr=0.1):
    ytr = ytr.view(-1, 1).float()          # (N,1) —— 否则 (N,1)-(N,) 会广播成 (N,N)
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True).clamp(min=1e-8)
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    w = torch.zeros(Xtr.shape[1], 1, device=Xtr.device)
    b = torch.zeros(1, device=Xtr.device)
    for _ in range(steps):
        p = torch.sigmoid(Xtr @ w + b)
        g = (p - ytr)
        w = w - lr * (Xtr.t() @ g / Xtr.shape[0] + 1e-4 * w)
        b = b - lr * g.mean(0)
    return (Xte @ w + b).squeeze(-1)


def auc_score(score, y):
    """y=1 为正类（stego）。方向归一化到 >=0.5。"""
    s, y = score.detach().cpu(), y.detach().cpu()
    order = torch.argsort(s)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, len(s) + 1, dtype=torch.float64)
    npos = float(y.sum())
    nneg = float(len(y) - npos)
    auc = (ranks[y > 0.5].sum() - npos * (npos + 1) / 2) / (npos * nneg)
    return float(max(auc, 1 - auc))


def tpr_at_fpr(score, y, fpr=0.01):
    s, y = score.detach().cpu(), y.detach().cpu()
    neg = s[y < 0.5]
    pos = s[y > 0.5]
    # 方向：让正类均值更大（若不然取反）
    if pos.mean() < neg.mean():
        s, pos, neg = -s, -pos, -neg
    thr = torch.quantile(neg, 1 - fpr)
    return float((pos >= thr).float().mean())


def bootstrap_auc_ci(score, y, n_boot=200, seed=0):
    rng = np.random.default_rng(seed)
    s, yy = score.detach().cpu().numpy(), y.detach().cpu().numpy().astype(bool)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(s), len(s))
        if yy[idx].all() or (~yy[idx]).all():
            continue
        st, yt = torch.tensor(s[idx]), torch.tensor(yy[idx].astype(np.float32))
        vals.append(auc_score(st, yt))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (float("nan"),) * 2


# ---------------- 检测器 2：小 CNN（从零训练） ----------------
class StegNet(nn.Module):
    def __init__(self, ch=32):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.SiLU(),
                                 nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.SiLU(),
                                 nn.MaxPool2d(2))
        self.net = nn.Sequential(blk(3, ch), blk(ch, ch * 2), blk(ch * 2, ch * 4),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                 nn.Linear(ch * 4, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_cnn(Xtr, ytr, Xte, yte, dev, epochs=15, bs=64, seed=0):
    torch.manual_seed(seed)
    net = StegNet().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    n = Xtr.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss = F.binary_cross_entropy_with_logits(net(Xtr[idx]), ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        return net(Xte).float()


# ---------------- 第二格：wrong-key 零分布的尾部有效性 ----------------
@torch.no_grad()
def null_tail_study(vae, covers, n_keys=200, dev="cuda"):
    """经验零分布 vs 二项分布假设。

    对固定 nonce、K 个随机错密钥，计算 16 校验位的汉明距离；
    若各位独立，应服从 Binomial(16, 0.5)（均值 8、标准差 2）。
    同时量化"同一张图内不同密钥的距离相关"导致的有效样本量下降。
    """
    nonce = "nullnonce"
    keys = [token_key() for _ in range(n_keys)]
    bins = torch.stack([build_params(k, nonce)["bins"] for k in keys]).long().to(dev)
    phs = torch.stack([build_params(k, nonce)["phases"] for k in keys]).float().to(dev)
    cb = torch.stack([check_bits(k, nonce, N_CHK) for k in keys]).float().to(dev)   # (K,16)

    z = vae.encode(covers, sample=False)
    Fz = torch.fft.fftshift(torch.fft.fft2(z, norm="ortho"), dim=(-2, -1))          # (N,C,H,W)
    rows, cols = bins[..., 0], bins[..., 1]
    # 逐图（避免 (N,K,C,P) 爆内存）。注意 PyTorch 高级索引：x[:, rows, cols]
    # 在 (C,H,W) 上会返回 (C, K, P)，需要 permute 成 (K, C, P)。
    d_all = []
    for i in range(Fz.shape[0]):
        v = Fz[i][:, rows, cols].permute(1, 0, 2)                      # (K,C,P)
        rot = torch.exp(-1j * phs)[:, None, :].to(v.dtype)             # (K,1,P)
        v = (v * rot).real.mean(1)                                     # (K,P)
        slot = v.reshape(n_keys, TOTAL_SLOTS, BPB).mean(-1)            # (K,80)
        chk = slot[:, TOTAL_SLOTS - N_CHK:]                            # (K,16)
        hard = (chk > 0).float()
        d = (hard != cb).sum(1).float()                                # (K,)
        d_all.append(d)
    D = torch.stack(d_all)                                              # (N,K)
    emp_mean, emp_std = float(D.mean()), float(D.std())
    binom_std = math.sqrt(N_CHK * 0.25)
    # 阈值 τ 的经验 FAR 与二项预测
    rows_out = []
    for tau in (4, 5, 6, 7):
        emp_far = float((D <= tau).float().mean())
        # 二项预测 P(X<=tau)，X~Bin(16,0.5)
        binom_far = sum(math.comb(N_CHK, k) for k in range(0, int(tau) + 1)) / 2 ** N_CHK
        rows_out.append({"tau": tau, "emp_far": emp_far, "binom_far": binom_far,
                         "ratio": (emp_far / binom_far) if binom_far > 0 else None})
    # 有效样本量：图内相关使"K 个错密钥"不等于 K 个独立样本
    within_var = float(D.var(dim=1, unbiased=False).mean())
    total_var = float(D.var(unbiased=False))
    return {"n_img": int(D.shape[0]), "n_keys": int(D.shape[1]),
            "emp_mean": emp_mean, "emp_std": emp_std, "binom_std": binom_std,
            "std_ratio": emp_std / binom_std,
            "within_image_var": within_var, "total_var": total_var,
            "effective_ratio": within_var / max(total_var, 1e-12),
            "far_table": rows_out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--vae-ckpt", default="checkpoints/vae32.pt")
    ap.add_argument("--n-train", type=int, default=1500)
    ap.add_argument("--n-test", type=int, default=800)
    ap.add_argument("--strengths", default="0.4,1.6")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--null-images", type=int, default=200)
    ap.add_argument("--null-keys", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/steganalysis_null.json")
    args = ap.parse_args()
    seed_everything(0)
    dev = args.device if torch.cuda.is_available() else "cpu"

    vae = build_vae("native", ckpt=args.vae_ckpt, device=dev)
    need = 2 * (args.n_train + args.n_test) + args.null_images
    covers_all = gather_covers(cifar_loader(args.data_root, train=False, batch_size=64,
                                            resize=PIX), need).to(dev)
    tr_clean = covers_all[:args.n_train]
    tr_stego = covers_all[args.n_train:2 * args.n_train]
    te_clean = covers_all[2 * args.n_train:2 * args.n_train + args.n_test]
    te_stego = covers_all[2 * args.n_train + args.n_test:2 * args.n_train + 2 * args.n_test]
    null_covers = covers_all[2 * (args.n_train + args.n_test):]
    print(f"covers: train {args.n_train}×2, test {args.n_test}×2, null {null_covers.shape[0]}"
          f" | 两类 cover 不相交 ✓ | 零假设 = D(E(cover))", flush=True)

    results = {"protocol": {"null": "D(E(cover))", "paired": False, "detector": "learned",
                            "res": PIX, "n_bits": N_BITS, "N": BPB * ECC},
               "strengths": []}
    for st in [float(v) for v in args.strengths.split(",")]:
        keys_tr = [token_key() for _ in range(args.n_train)]
        keys_te = [token_key() for _ in range(args.n_test)]
        nonc_tr = derive_nonces_from_keys(keys_tr, start=0)
        nonc_te = derive_nonces_from_keys(keys_te, start=0)
        xs_tr, _, p_tr = make_sets(vae, tr_stego, keys_tr, nonc_tr, st, dev, need_clean=False)
        xs_te, _, p_te = make_sets(vae, te_stego, keys_te, nonc_te, st, dev, need_clean=False)
        # clean 类来自**另一批不相交的 cover**，走同一 VAE 重建（正确零假设）
        xc_tr_clean = vae.decode(vae.encode(tr_clean, sample=False), target_hw=(PIX, PIX))
        xc_te_clean = vae.decode(vae.encode(te_clean, sample=False), target_hw=(PIX, PIX))

        Xtr = torch.cat([xs_tr, xc_tr_clean]).clamp(-1, 1)
        ytr = torch.cat([torch.ones(xs_tr.shape[0]), torch.zeros(xs_tr.shape[0])]).to(dev)
        Xte = torch.cat([xs_te, xc_te_clean]).clamp(-1, 1)
        yte = torch.cat([torch.ones(xs_te.shape[0]), torch.zeros(xs_te.shape[0])]).to(dev)
        perm = torch.randperm(Xtr.shape[0], device=dev)
        Xtr, ytr = Xtr[perm], ytr[perm]
        perm = torch.randperm(Xte.shape[0], device=dev)
        Xte, yte = Xte[perm], yte[perm]

        # 检测器 1：频谱统计 + 逻辑回归
        Ftr, Fte = spectral_features(Xtr), spectral_features(Xte)
        sc_lin = fit_logreg(Ftr, ytr, Fte, yte)
        auc_lin = auc_score(sc_lin, yte)
        # 检测器 2：小 CNN
        sc_cnn = train_cnn(Xtr, ytr, Xte, yte, dev, epochs=args.epochs)
        auc_cnn = auc_score(sc_cnn, yte)
        lo, hi = bootstrap_auc_ci(sc_cnn, yte)
        row = {"strength": st, "psnr_stego": float(np.mean(p_te)),
               "psnr_clean": float(np.mean([
                   psnr(vae.decode(vae.encode(c[None], sample=False), target_hw=(PIX, PIX)),
                        c[None]) for c in te_clean[:16]])),
               "auc_linear": auc_lin, "auc_cnn": auc_cnn, "auc_cnn_ci": [lo, hi],
               "tpr1_cnn": tpr_at_fpr(sc_cnn, yte, 0.01),
               "tpr1_lin": tpr_at_fpr(sc_lin, yte, 0.01),
               "n_train": int(Xtr.shape[0]), "n_test": int(Xte.shape[0])}
        results["strengths"].append(row)
        print(f"[st={st:<4}] 载密 {row['psnr_stego']:.2f} dB | 线性 AUC {auc_lin:.3f} "
              f"(TPR@1% {row['tpr1_lin']:.3f}) | CNN AUC {auc_cnn:.3f} "
              f"[{lo:.3f},{hi:.3f}] (TPR@1% {row['tpr1_cnn']:.3f})", flush=True)

    print("\n[第二格] wrong-key 零分布 vs 二项假设", flush=True)
    nt = null_tail_study(vae, null_covers, n_keys=args.null_keys, dev=dev)
    results["null_tail"] = nt
    print(f"  经验均值/标准差 = {nt['emp_mean']:.3f} / {nt['emp_std']:.3f} "
          f"(二项假设 std=2.0 -> 比值 {nt['std_ratio']:.3f})", flush=True)
    print(f"  图内方差/总方差 = {nt['effective_ratio']:.3f} "
          f"(越接近 0 说明同一张图的 K 个错密钥高度相关，有效样本量远小于 K)", flush=True)
    for r in nt["far_table"]:
        print(f"  τ={r['tau']}: 经验 FAR {r['emp_far']:.3e} vs 二项 {r['binom_far']:.3e} "
              f"(比值 {r['ratio'] if r['ratio'] is None else round(r['ratio'], 3)})", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nsaved -> {args.out}", flush=True)
    print("[判据] 操作点(≈35 dB) CNN AUC ≤0.60 → 不可检测成立，进 P1；"
          "阳性对照(≈22-26 dB) AUC 必须 ≥0.9，否则实验无效。", flush=True)


if __name__ == "__main__":
    main()

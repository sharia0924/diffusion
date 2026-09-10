"""密钥安全性评测组件（P1 护栏）。

把"错密钥 ≈50%"从经验观察升级为可量化实验，并**明确区分三个不同的威胁模型**
（早期报告把三者混为一谈，需要修正）：

  1. wrong_key_profile   —— 错密钥解码：BER 分布（mean/var/min/max）+ 真密钥对照。
     回答"没有密钥能否读出消息"。
  2. matched-filter 校验 —— 无需解码器的密钥有效性检验 + 错密钥虚警率 FAR。
  3. slot_detection_auc  —— **槽位定位/水印存在性**攻击：给定 (cover, stego) 对，
     用消息符号加权平均差分谱，看能否把"该密钥真正使用的频点"从其余频点里挑出来。
     注意：这是**存在性定位**指标，不是密钥恢复；AUC 明显 > 0.5 即攻击有效。
  4. watermark_presence_auc —— **盲检测**（隐写分析视角）：只给 stego（或 cover/stego
     潜变量），用均值差分方向做线性判别，看能否区分"载密图 vs 原图"。
     这是"载密信息是否泄漏"的硬指标，0.5 附近才算安全。
  5. key_space_bounds    —— 密钥派生参数空间的组合学下界。
     必须与 `effective_key_entropy_bits`（密钥字符串本身的熵，如 token_key 的 64 bit）
     一起报告：攻击者枚举的是**密钥**而不是派生参数，真实空间受后者约束。

威胁模型说明（slot_detection_auc）：攻击者持有 N 对 (cover, stego)、密钥固定、模型公开。
其做法：把两侧都 DDIM 反演到 x_T，频域差分 Δ_i = F(x_T(stego_i)) − F(x_T(cover_i))，
图案分量跨图同分布而噪声独立 → 按消息符号加权平均可凸显图案频点。
对每个槽位 j 构造判别量 D_j[b] = (2/N)·Σ_i s_j^(i)·Δ_i[b]，用 |D_j| 对"真槽位频点 vs 其余"
做 Mann-Whitney AUC：无 nonce 时 AUC→1（可定位图案），有 nonce 时趋近 0.5（失败）。

nonce 协议：nonce = H(key || counter)，**不依赖 cover**（见 utils.derive_nonce_from_key）。
攻击者若已知 counter（公开参数），仍可复现图案；此时防护依赖密钥本身，
这一点在报告中必须写明，不能把"逐图 nonce"宣传成密码学保证。
"""

import math

import numpy as np
import torch

from . import pattern


# ---------------- 1. 错密钥 / 近密钥 BER 分布 ----------------

def near_keys(key: str) -> list[str]:
    """近密钥变体：大小写翻转、单字符替换、增删字符、倒序。SHA256 雪崩应使其与真密钥等价无关。"""
    variants = set()
    variants.add(key.upper())
    variants.add(key.lower())
    variants.add(key + "x")
    variants.add(key[::-1])
    for i in range(min(4, len(key))):
        c = key[i]
        variants.add(key[:i] + ("a" if c != "a" else "b") + key[i + 1:])
    variants.discard(key)
    return sorted(variants)


def wrong_key_population(true_keys: list[str], n_random: int = 200,
                         seed: int = 0) -> list[str]:
    """随机错密钥 + 真密钥的近密钥变体。"""
    import random as _random
    rng = _random.Random(seed)
    pool = {"%016x" % rng.getrandbits(64) for _ in range(n_random)}
    for k in true_keys[:8]:
        pool.update(near_keys(k))
    return sorted(pool)


@torch.no_grad()
def wrong_key_profile(traj, x_T: torch.Tensor, bits: torch.Tensor, nonces: list[str],
                      decoder, wrong_keys: list[str]) -> torch.Tensor:
    """对预反演潜变量 x_T，用每个错密钥解码，返回 (n_wrong,) 每密钥平均 BER。

    x_T 由 traj.invert_latents 预先算好（反演与密钥无关，只算一次）。
    """
    assert decoder is not None
    B = x_T.shape[0]
    bits01 = (bits > 0.5).float()
    bers = []
    for wk in wrong_keys:
        feats = traj.features_from_latents(x_T, [wk] * B, nonces)
        msg_logits = traj.collapse(decoder(feats))
        pred = (msg_logits > 0).float()
        bers.append(1.0 - (pred == bits01).float().mean().item())
    return torch.tensor(bers)


# ---------------- 2. matched-filter 密钥校验（无需解码器） ----------------

@torch.no_grad()
def effective_key_entropy_bits(key: str | None = None, n_hex: int = 16) -> float:
    """密钥字符串本身的有效熵（bit）。

    报告密钥安全性时必须区分两个量：
      - `key_space_bounds(...)["log2_total"]`：**派生参数空间**的组合下界
        （频点选择 × 相位量化），描述"图案有多少种可能"；
      - 本函数：**密钥**的熵，描述"攻击者要枚举多少种密钥"。
    攻击者枚举的是密钥，因此真实安全强度由后者封顶。
    `token_key()` 生成 16 个 hex 字符 = 64 bit 随机量。
    """
    if key is None:
        return float(4 * n_hex)
    return float(len(key) * 4)  # 仅当 key 是纯 hex 随机串时成立


def key_check_distances(traj, x_T: torch.Tensor, key: str, nonce: str) -> torch.Tensor:
    """对一批潜变量逐张做 matched-filter 密钥校验，返回 (B,) 槽位汉明距离。

    ring_features 已按密钥相位去旋转：校验槽位 j 的同相分量 ≈ m·s_j·scale，
    以已知幅度 m 加权求和后取符号即恢复的校验比特，与 SHA256(key|nonce) 期望比特
    比较计距离。真密钥（未受扰）距离 = 0；错密钥 ≈ Binomial(n_check, 0.5)。
    """
    params = traj.params_for(key, nonce)
    g = traj.n_pairs // traj.total_embed_bits
    n_check = traj.n_check_bits
    rows = slice(traj.msg_embed_bits * g, traj.n_pairs)
    base = params["base_mag"][rows]

    expected = pattern.check_bits(key, nonce, n_check).to(x_T.device)
    dists = []
    for i in range(x_T.shape[0]):
        feats = pattern.ring_features(x_T[i], params)          # (2*n_pairs,) 已去旋
        re_v = feats[: traj.n_pairs][rows].view(n_check, g)     # 同相分量
        proj = (re_v * base.view(n_check, g)).sum(1)            # (n_check,)
        pred = (proj > 0).float()
        dists.append(int((pred != expected).sum().item()))
    return torch.tensor(dists)


def far_at_tau(wrong_dists: torch.Tensor, tau: int) -> float:
    """错密钥在阈值 tau（距离 ≤ tau 判为有效）下的虚警率。"""
    return (wrong_dists <= tau).float().mean().item()


# ---------------- 3. 有效密钥空间下界 ----------------

def key_space_bounds(n_pairs: int, res: int = 32, r_min: int = 3,
                     phase_bits: int = 8) -> dict:
    """有效密钥空间（对数）：频点子集选择 × 每频点 phase_bits 位量化相位。

    组合学下界；攻击者的现实代价由解码 oracle 决定（见 wrong_key_profile）。
    """
    from .pattern import _halfplane_bins
    r_max_cap = res // 2 - 1
    chosen = None
    for rm in range(r_min, r_max_cap + 1):
        if len(_halfplane_bins(res, r_min, rm)) >= n_pairs:
            chosen = rm
            break
    if chosen is None:
        raise ValueError("n_pairs 超出环带容量")
    n_avail = len(_halfplane_bins(res, r_min, chosen))
    log2_choose = (math.lgamma(n_avail + 1) - math.lgamma(n_pairs + 1)
                   - math.lgamma(n_avail - n_pairs + 1)) / math.log(2)
    return {
        "res": res, "r": (r_min, chosen), "n_avail": n_avail, "n_pairs": n_pairs,
        "log2_bin_selection": log2_choose,
        "log2_phases": n_pairs * phase_bits,
        "log2_total": log2_choose + n_pairs * phase_bits,
    }


# ---------------- 4. 多图差分密钥恢复攻击 ----------------

def _rank_auc(scores: torch.Tensor, pos_mask: torch.Tensor) -> float:
    """Mann-Whitney AUC（含并列值近似处理）。"""
    n_pos = int(pos_mask.sum())
    n_neg = scores.numel() - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = torch.argsort(scores)
    ranks = torch.empty(scores.numel(), dtype=torch.float64)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=torch.float64)
    auc = (ranks[pos_mask].sum().item() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


@torch.no_grad()
def slot_detection_auc(traj, covers: torch.Tensor, stegos: torch.Tensor,
                       bits: torch.Tensor, key: str, nonces: list[str],
                       rec_steps: int = 50, latents=None) -> float:
    """多图差分攻击的槽位可检测性 AUC（对全部嵌入槽位取平均）。

    key: 攻击目标密钥（攻击模型中所有图共用同一密钥）。
    nonces: 各图实际使用的 nonce（传 [""]*N 复现无随机化协议；逐图 nonce 为我们的方案）。
    latents: 可选预计算的 (xT_covers, xT_stegos)，供 AUC-N 曲线复用。
    返回标量 AUC ∈ [0,1]，0.5=攻击失败。
    """
    N = covers.shape[0]
    if latents is None:
        xT_c = traj.invert_latents(covers, rec_steps)
        xT_s = traj.invert_latents(stegos, rec_steps)
    else:
        xT_c, xT_s = latents

    def _spec(x):
        f = torch.fft.fftshift(torch.fft.fft2(x, norm="ortho"), dim=(-2, -1))
        return f.mean(dim=1)                                   # 通道平均 → (N,H,W) complex

    deltas = (_spec(xT_s) - _spec(xT_c)).reshape(N, -1)        # (N, HW) complex
    HxW = deltas.shape[1]

    full = torch.stack([traj.full_bits(bits[i], key, nonces[i]) for i in range(N)])
    s = torch.where(full > 0.5, 1.0, -1.0).to(torch.complex64)   # (N, E)
    deltas = deltas.to(torch.complex64)
    D = (2.0 / N) * (s.t() @ deltas)                             # (E, HW) complex

    g = traj.n_pairs // traj.total_embed_bits
    aucs = []
    for j in range(traj.total_embed_bits):
        score = D[j].abs().cpu()
        pos = torch.zeros(HxW, dtype=torch.bool)
        for i in range(N):                                     # 跨图取并集（攻击者最优情形）
            bins = traj.params_for(key, nonces[i])["bins"][j * g:(j + 1) * g].cpu()
            pos[bins[:, 0] * traj.res + bins[:, 1]] = True
        if pos.sum() == 0:
            continue
        aucs.append(_rank_auc(score, pos))
    return float(np.mean(aucs)) if aucs else 0.5


# ---------------- 5. 盲水印检测（隐写分析视角） ----------------

@torch.no_grad()
def watermark_presence_auc(x_T_cover: torch.Tensor, x_T_stego: torch.Tensor,
                           n_train: int | None = None) -> dict:
    """盲检测 AUC：线性判别器能否区分"载密图 vs 原图"的潜变量（隐写分析视角）。

    攻击者只拿到两堆潜变量（等价于拿到图片 + 公开的模型与提取流程），
    不掌握密钥、也不需要定位具体频点。判别方向取训练集上的均值差分
    （两类等先验下的最优线性方向），再用其投影做 Mann-Whitney AUC。

    注意**方向归一化**：判别器的符号是任意的（+ 表示载密或 - 表示载密都可行），
    因此同时计算两个方向的分辨能力并取较大者。返回的 AUC ∈ [0.5, 1]，
    0.5 = 完全不可区分（安全），越接近 1 说明载密图越容易被测出。

    返回 dict：
      auc        —— 可分性（已做方向归一化）
      direction  —— 被采用的方向（+1 / -1）
      n_pos/n_neg, train_size, test_size
    """
    n = x_T_cover.shape[0]
    if n_train is None:
        n_train = max(1, n // 2)
    n_train = int(min(max(1, n_train), n - 1))
    tr_c, te_c = x_T_cover[:n_train], x_T_cover[n_train:]
    tr_s, te_s = x_T_stego[:n_train], x_T_stego[n_train:]

    template = (tr_s - tr_c).mean(dim=0).reshape(-1)          # (C*H*W,)
    scores_pos = (te_s - te_c).reshape(te_c.shape[0], -1) @ template   # 载密图投影
    scores_neg = torch.zeros(te_c.shape[0], device=te_c.device, dtype=template.dtype)
    pos = torch.cat([torch.ones(scores_pos.shape[0], dtype=torch.bool),
                     torch.zeros(scores_neg.shape[0], dtype=torch.bool)])
    scores = torch.cat([scores_pos, scores_neg]).cpu()
    auc_p = _rank_auc(scores, pos)
    auc_n = _rank_auc(-scores, pos)            # 反向判别器
    use_pos = auc_p >= auc_n
    return {"auc": float(auc_p if use_pos else auc_n),
            "direction": 1 if use_pos else -1,
            "n_pos": int(pos.sum()), "n_neg": int((~pos).sum()),
            "train_size": n_train, "test_size": int(te_c.shape[0])}


@torch.no_grad()
def residual_template_correlation(traj, covers: torch.Tensor, stegos: torch.Tensor,
                                  keys, nonces: list[str], rec_steps: int = 50,
                                  latents=None) -> dict:
    """跨图残差方向的一致性：攻击者把 N 张图的残差平均后还剩多少结构。

    做法：逐图计算去均值后的频域残差方向 r_i = ΔF_i / ||ΔF_i||，
    报告两两 |<r_i, r_j>| 的均值与最大值。

    - 同 nonce（图案固定）：残差方向高度一致 → |corr|→1，多图平均可累积；
    - 逐图 nonce（ours）：方向应接近正交 → |corr|→0，平均退化为随机游走。
    这给出了"逐图 nonce 到底起多大作用"的直接量化，比只报 AUC 更有解释力。
    """
    N = covers.shape[0]
    if latents is None:
        xT_c = traj.invert_latents(covers, rec_steps)
        xT_s = traj.invert_latents(stegos, rec_steps)
    else:
        xT_c, xT_s = latents

    def _spec(x):
        f = torch.fft.fftshift(torch.fft.fft2(x, norm="ortho"), dim=(-2, -1))
        return f.mean(dim=1)

    diffs = (_spec(xT_s) - _spec(xT_c)).reshape(N, -1)
    flat = torch.cat([diffs.real, diffs.imag], dim=1)
    flat = flat - flat.mean(dim=1, keepdim=True)
    flat = flat / flat.norm(dim=1, keepdim=True).clamp(min=1e-12)
    gram = (flat @ flat.t()).abs()
    off = gram[~torch.eye(N, dtype=torch.bool, device=gram.device)]
    return {"mean_abs_corr": float(off.mean()), "max_abs_corr": float(off.max()),
            "n": N}

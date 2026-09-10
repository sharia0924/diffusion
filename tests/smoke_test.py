"""冒烟测试（CPU/GPU 数秒）：验证调度公式、密钥图案、安全组件与整条流水线。

  python tests/smoke_test.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from krd import RingDecoder, Schedule, TrajStego, UNet, security
from krd.distortions import (STANDARD_ATTACKS, apply_attack, center_crop_resize, crop_fill,
                             diff_jpeg, rotate, scale_zoom, schedule_noise, translate)
from krd.metrics import psnr
from krd.pattern import inject_pattern, key_params, ring_features, template_features
from krd.utils import derive_nonce, derive_nonce_from_key, derive_nonces_from_keys


def check(name: str, cond: bool):
    status = "ok" if cond else "FAIL"
    print(f"[{status}] {name}")
    if not cond:
        raise AssertionError(name)


def main():
    # 中文 Windows 的 GBK 控制台无法编码部分字符，降级为替换而非崩溃
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    torch.manual_seed(0)

    # 1) 加噪/去噪公式自洽
    sched = Schedule(1000, device=device)
    x0 = torch.randn(2, 3, 32, 32, device=device).clamp(-1, 1)
    t = torch.randint(0, 1000, (2,), device=device)
    eps = torch.randn_like(x0)
    x_t = sched.add_noise(x0, t, eps)
    x0_hat = sched.predict_x0(x_t, t, eps, clip_denoised=False)
    check("add_noise/predict_x0 自洽", (x0_hat - x0).abs().max().item() < 1e-3)

    seq = sched.timestep_seq(50).tolist()
    check("timestep_seq 严格递增且首尾正确",
          seq[0] == 0 and seq[-1] == 999 and all(b > a for a, b in zip(seq, seq[1:])))

    # 2) 密钥图案: 注入后真密钥特征与模板高度相关, 错密钥接近 0
    bits = torch.tensor([0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0], device=device).float()
    n_pairs = 16 * 3 * 2
    p_true = key_params("key-123", n_pairs, res=32)
    p_wrong = key_params("evil-key", n_pairs, res=32)
    x_T = torch.randn(3, 32, 32, device=device)
    x_T2 = inject_pattern(x_T, bits, p_true, 1.0)

    def corr(feats, tmpl):
        feats = feats / feats.norm()
        tmpl = tmpl / tmpl.norm()
        return (feats * tmpl.to(feats.device)).sum().item()

    f_true = corr(ring_features(x_T2, p_true), template_features(bits, p_true))
    f_wrong = corr(ring_features(x_T2, p_wrong), template_features(bits, p_wrong))
    print(f"    corr(true-key)={f_true:.3f}, corr(wrong-key)={f_wrong:.3f}")
    check("真密钥相关≈1", f_true > 0.95)
    check("错密钥相关≈0（密钥门控生效）", f_wrong < 0.2)

    # 3) 整条流水线（随机权重 DDPM, 验证形状与流程）
    model = UNet(base=16, ch_mults=(1, 2, 2), attn_levels=(2,)).to(device)
    stego = TrajStego(model, sched, n_bits=16, ecc_reps=3, bins_per_bit=2,
                      n_check_bits=32, res=32, device=device)
    check("嵌入槽位 = ECC消息 + 校验比特", stego.total_embed_bits == 48 + 32)
    check("n_pairs = 槽位 × 每槽频点对", stego.n_pairs == stego.total_embed_bits * 2)

    cover = torch.rand(2, 3, 32, 32, device=device) * 2 - 1
    bits2 = torch.randint(0, 2, (2, 16), device=device).float()
    keys = ["alice", "bob"]
    nonces = derive_nonces_from_keys(keys, start=0)
    check("nonce 自包含且逐图不同",
          len(set(nonces)) == 2 and all(derive_nonce_from_key(k, i) == n
                                        for i, (k, n) in enumerate(zip(keys, nonces))))

    sg = stego.hide(cover, bits2, keys, hide_steps=4, strength=1.0, nonces=nonces)
    check("hide 输出形状与范围", sg.shape == cover.shape and -1.0 <= sg.min() and sg.max() <= 1.0)

    feats = stego.recover_features(sg, keys, rec_steps=4, nonces=nonces)
    check("recover_features 形状", feats.shape == (2, 2 * stego.n_pairs))

    dec = RingDecoder(feats.shape[1], stego.total_embed_bits).to(device)
    logits = stego.recover(sg, keys, rec_steps=4, decoder=dec, nonces=nonces)
    check("recover logits 形状(ECC 折叠后)", logits.shape == (2, 16))

    # 4) matched-filter 密钥校验（无需训练）
    xt_rand = torch.randn(3, 32, 32, device=device)
    b_single = torch.randint(0, 2, (16,), device=device).float()
    k, n = "k-test", "n-test"
    xt_inj = inject_pattern(xt_rand, stego.full_bits(b_single, k, n),
                            stego.params_for(k, n), 1.0)
    d_true = security.key_check_distances(stego, xt_inj[None], k, n)
    d_wrong = security.key_check_distances(stego, xt_inj[None], "wrong-key", n)
    print(f"    check-dist: true={int(d_true[0])}, wrong={int(d_wrong[0])}")
    check("真密钥校验距离=0（精确注入下）", int(d_true[0]) == 0)
    check("错密钥校验距离≈n/2（>8/32）", int(d_wrong[0]) > 8)

    # 5) nonce 逐图随机化: 共享频点上图案残差跨图去相关（跨图差分平均失效的根源）
    #    真实攻击中各图 cover 不同 -> 用两张不同基底谱, 残差 = 图案值 - 各自基底
    pA = stego.params_for("k1", "nA")
    pB = stego.params_for("k1", "nB")
    bA = pA["bins"].cpu().numpy()
    bB = pB["bins"].cpu().numpy()
    common = {(int(y), int(x)) for y, x in bA} & {(int(y), int(x)) for y, x in bB}
    base_A = torch.randn(3, 32, 32, device=device)
    base_B = torch.randn(3, 32, 32, device=device)
    F_base_A = torch.fft.fftshift(torch.fft.fft2(base_A, norm="ortho"), dim=(-2, -1))
    F_base_B = torch.fft.fftshift(torch.fft.fft2(base_B, norm="ortho"), dim=(-2, -1))
    F_A = torch.fft.fftshift(torch.fft.fft2(
        inject_pattern(base_A, stego.full_bits(b_single, "k1", "nA"), pA, 1.0),
        norm="ortho"), dim=(-2, -1))
    F_B = torch.fft.fftshift(torch.fft.fft2(
        inject_pattern(base_B, stego.full_bits(b_single, "k1", "nB"), pB, 1.0),
        norm="ortho"), dim=(-2, -1))
    da, db = [], []
    for y, x in common:
        da.append(F_A[:, y, x] - F_base_A[:, y, x])
        db.append(F_B[:, y, x] - F_base_B[:, y, x])
    da = torch.stack(da).flatten()
    db = torch.stack(db).flatten()
    # 相干度: |Σ da·conj(db)| / Σ(|da||db|)。图案同相=1, nonce 随机化后≈0
    cross = (da * torch.conj(db)).sum().abs() / (da.abs() * db.abs()).sum().clamp(min=1e-8)
    # 对照: 同 nonce 时应高度相干（攻击可累积）
    F_A2 = torch.fft.fftshift(torch.fft.fft2(
        inject_pattern(base_B, stego.full_bits(b_single, "k1", "nA"), pA, 1.0),
        norm="ortho"), dim=(-2, -1))
    da2 = torch.stack([F_A2[:, y, x] - F_base_B[:, y, x] for y, x in common]).flatten()
    cross_same = (da * torch.conj(da2)).sum().abs() / (da.abs() * da2.abs()).sum().clamp(min=1e-8)
    print(f"    nonce 共享频点 {len(common)}, 跨 nonce 相干度 {cross:.3f}, "
          f"同 nonce 相干度 {cross_same:.3f}")
    check("nonce 使跨图图案残差去相关（<0.4）", cross < 0.4)
    check("同 nonce 时图案相干（>0.5, 攻击可累积=脆弱基线）", cross_same > 0.5)

    # 6) 安全组件端到端（随机权重模型, 只验证可运行性）
    covers4 = torch.rand(4, 3, 32, 32, device=device) * 2 - 1
    bits4 = torch.randint(0, 2, (4, 16), device=device).float()
    keys4 = ["ka", "kb", "kc", "kd"]
    nonces4 = ["na", "nb", "nc", "nd"]
    sg4 = stego.hide(covers4, bits4, keys4, hide_steps=4, strength=1.0, nonces=nonces4)
    xT4 = stego.invert_latents(sg4, 4)
    bers = security.wrong_key_profile(stego, xT4, bits4, nonces4, dec,
                                      ["w1", "w2", "w3"])
    check("wrong_key_profile 可运行", bers.shape == (3,) and bool((bers >= 0).all()))
    auc = security.slot_detection_auc(stego, covers4, sg4, bits4, "ka", nonces4,
                                      rec_steps=4)
    print(f"    slot-detection AUC: {auc:.3f}")
    check("slot_detection_auc 可运行且在 [0,1]", 0.0 <= auc <= 1.0)
    ks = security.key_space_bounds(stego.n_pairs, stego.res)
    check("key_space_bounds 给出正的对数空间", ks["log2_total"] > 0)
    print(f"    log2(key space) ≈ {ks['log2_total']:.1f} bit")
    nk = security.near_keys("secret")
    check("near_keys 生成变体", len(nk) > 0 and "secret" not in nk)

    # 7) 调度坐标加噪（再生攻击代理）
    xn = schedule_noise(cover, sched)
    check("schedule_noise 形状/有限", xn.shape == cover.shape and bool(torch.isfinite(xn).all()))

    # 7b) v2 修复项：几何攻击必须是"真几何"而不是循环平移
    img = torch.arange(32, dtype=torch.float32, device=device).view(1, 1, 1, 32).expand(1, 1, 32, 32)
    cropped = crop_fill(img, 2)
    check("crop_fill 是裁剪而非循环平移（内容发生平移）",
          not torch.allclose(cropped, torch.roll(img, shifts=(2, 2), dims=(2, 3))))
    check("crop_fill 边缘用边缘像素回填（不是黑边）",
          bool(torch.allclose(cropped[:, :, :2, :], img[:, :, :2, :])))
    check("crop_fill(0) 恒等", bool(torch.equal(crop_fill(img, 0), img)))
    for fn, arg, name in [(rotate, 5.0, "rotate"), (translate, 2, "translate"),
                          (scale_zoom, 1.1, "scale_zoom"),
                          (center_crop_resize, 0.8, "center_crop_resize")]:
        out = fn(cover, arg)
        check(f"{name} 形状/范围正确",
              out.shape == cover.shape and -1.0 <= out.min() and out.max() <= 1.0)
    check("几何攻击集合已在 STANDARD_ATTACKS 中登记",
          {"crop2px", "rotate5", "translate2px", "zoom1.1", "cropresize0.8"}
          <= {n for n, _, _ in STANDARD_ATTACKS})

    # 7c) v2 修复项：nonce 不依赖 cover（换 cover 协议参数不变）
    k_test = "nonce-key"
    check("nonce 与 cover 无关",
          derive_nonce_from_key(k_test, 3) == derive_nonce_from_key(k_test, 3)
          and derive_nonce_from_key(k_test, 3) != derive_nonce_from_key(k_test, 4)
          and derive_nonce_from_key(k_test, 3) != derive_nonce_from_key("other", 3))
    check("旧的 cover-派生 nonce 仍保留但已弃用", len(derive_nonce(cover[0])) == 16)

    # 7d) v2 修复项：盲水印检测 AUC 必须能运行且给出合理值
    xT_clean = stego.invert_latents(cover, 4)
    pres = security.watermark_presence_auc(xT_clean, stego.invert_latents(sg, 4))
    print(f"    watermark-presence AUC: {pres['auc']:.3f} (direction {pres['direction']:+d})")
    check("watermark_presence_auc 已做方向归一化（≥0.5）",
          0.5 <= pres["auc"] <= 1.0)
    corr = security.residual_template_correlation(stego, cover, sg, keys, nonces, 4)
    check("residual_template_correlation 可运行", 0.0 <= corr["mean_abs_corr"] <= 1.0)
    check("effective_key_entropy_bits(16 hex)=64bit",
          security.effective_key_entropy_bits(None, 16) == 64.0)

    # 7e) v2 修复项：LPIPS 为可选依赖，缺失时返回 None 而不是崩
    from krd.perceptual import lpips, lpips_backend
    lp = lpips(cover, cover)
    print(f"    LPIPS backend: {lpips_backend() or 'unavailable'}, self-LPIPS={lp}")
    check("lpips 缺失时返回 None（不抛异常）", lp is None or lp >= 0.0)

    # 8) 可微 JPEG 数值范围与失真强度
    xj = diff_jpeg(cover, 50)
    check("diff_jpeg 输出范围", -1.0 <= xj.min() and xj.max() <= 1.0)
    q90 = psnr(cover, diff_jpeg(cover, 90))
    q30 = psnr(cover, diff_jpeg(cover, 30))
    print(f"    PSNR(jpeg90)={q90:.1f}dB > PSNR(jpeg30)={q30:.1f}dB")
    check("JPEG 质量越低失真越大", q90 > q30)

    print("\nsmoke test 全部通过 [OK]")


if __name__ == "__main__":
    main()

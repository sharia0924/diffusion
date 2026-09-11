"""strength 扫描：找出"载密图 PSNR vs 比特准确率"的真实工作点。

这是隐空间迁移的**关键闸门**：native VAE 的往返上限已到 36–38 dB，
但 strength=1.0 时载密图只有 ~9.6 dB（嵌入开销 ~28 dB）。
本脚本扫描 strength ∈ [0.5, 1e-4]，报告每个点的
  PSNR / SSIM / matched-filter 准确率 / 解码器准确率(clean, jpeg50)
从而回答：**能否在 35 dB 附近还保持解码可用**。

用法:
  python scripts/eval_strength_sweep.py --ddpm-ckpt checkpoints/ddpm_latent.pt \
      --decoder-ckpt checkpoints/decoder_latent_best.pt --out results/strip_latent.md
  python scripts/eval_strength_sweep.py --pixel-baseline --ddpm-ckpt checkpoints/ddpm_cifar.pt
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from krd import RingDecoder
from krd.distortions import apply_attack
from krd.metrics import bit_accuracy, psnr, ssim
from krd.perceptual import lpips
from krd.pattern import ecc_collapse
from krd.utils import derive_nonces_from_keys, seed_everything, token_key
from scripts.eval_common import StegoIO, cifar_loader, gather_covers
from scripts.train_decoder import load_stego


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddpm-ckpt", default="checkpoints/ddpm_latent.pt")
    ap.add_argument("--decoder-ckpt", default="checkpoints/decoder_latent_best.pt")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--hide-steps", type=int, default=50)
    ap.add_argument("--rec-steps", type=int, default=50)
    ap.add_argument("--strengths", default="0.5,0.2,0.1,0.05,0.02,0.01,0.005,0.001")
    ap.add_argument("--pixel-res", type=int, default=None)
    ap.add_argument("--inject-mode", default=None, help="覆盖注入模式；默认取解码器配置")
    ap.add_argument("--pixel-baseline", action="store_true")
    ap.add_argument("--out", default="results/strength_sweep.md")
    ap.add_argument("--seed", type=int, default=23)
    args = ap.parse_args()
    seed_everything(args.seed)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _m = torch.load(args.ddpm_ckpt, map_location="cpu", weights_only=False).get("args", {})
    pixel_res = args.pixel_res or _m.get("resize") or 32

    dec, cfg = None, None
    if os.path.exists(args.decoder_ckpt):
        ck = torch.load(args.decoder_ckpt, map_location=dev, weights_only=True)
        cfg = ck["config"]
    _c = cfg or {}
    stego = load_stego(args.ddpm_ckpt, dev, with_vae=not args.pixel_baseline,
                       n_bits=_c.get("n_bits", 16), ecc_reps=_c.get("ecc", 3),
                       bins_per_bit=_c.get("bpb", 2),
                       n_check_bits=_c.get("n_check_bits", 32),
                       inject_mode=args.inject_mode or _c.get("inject_mode", "replace"))
    io = StegoIO(stego, pixel_res=pixel_res)
    print(f"[space] {io.describe()} pixel_res={pixel_res}", flush=True)
    print(f"[capacity] {io.capacity_report(stego.total_embed_bits)}", flush=True)
    if cfg:
        dec = RingDecoder(2 * cfg["n_pairs"],
                          cfg["n_bits"] * cfg["ecc"] + cfg["n_check_bits"]).to(dev)
        dec.load_state_dict(ck["decoder"])
        dec.eval()

    loader = cifar_loader(args.data_root, train=False, batch_size=args.batch,
                          resize=None if pixel_res == 32 else pixel_res)
    covers = gather_covers(loader, args.n).to(dev)
    keys = [token_key() for _ in range(args.n)]
    nonces = derive_nonces_from_keys(keys, start=0)
    n_bits = stego.n_bits
    g = torch.Generator(device="cpu").manual_seed(0)
    bits = torch.randint(0, 2, (args.n, n_bits), generator=g).float().to(dev)

    if io.latent and pixel_res == 32:
        print("[warn] 隐空间模型通常配 --resize 64 训练；当前用 32×32 cover，"
              "若 VAE 是按 64 训练的请加 --pixel-res 64", flush=True)
    base = io.baseline_psnr(covers)
    print(f"[latent] 无嵌入往返上限 PSNR {base:.2f} dB", flush=True)

    def mf_acc(z):
        accs = []
        for i in range(z.shape[0]):
            p = stego.params_for(keys[i], nonces[i])
            feats = io.recover_features(z[i:i + 1], [keys[i]], args.rec_steps,
                                        nonces=[nonces[i]])[0]
            gg = stego.n_pairs // stego.total_embed_bits
            re = feats[:stego.n_pairs].view(stego.total_embed_bits, gg).mean(-1)
            lg = ecc_collapse(re[None, :stego.msg_embed_bits], n_bits, stego.ecc)
            accs.append(bit_accuracy(lg, bits[i][None]))
        return float(np.mean(accs))

    def dec_acc(z, atk=None):
        if dec is None:
            return float("nan")
        accs = []
        for i in range(0, args.n, args.batch):
            xin = z[i:i + args.batch]
            if atk is not None:
                xin = io.attack(xin, lambda t: apply_attack(t, atk[0], atk[1]))
            lg = io.recover(xin, keys[i:i + args.batch], args.rec_steps, dec,
                            nonces=nonces[i:i + args.batch])
            accs.append(bit_accuracy(lg, bits[i:i + args.batch]))
        return float(np.mean(accs))

    rows = []
    for st in [float(v) for v in args.strengths.split(",")]:
        sg = torch.cat([
            io.hide(covers[i:i + args.batch], bits[i:i + args.batch],
                    keys[i:i + args.batch], args.hide_steps, st,
                    nonces=nonces[i:i + args.batch])
            for i in range(0, args.n, args.batch)
        ])
        px = io.to_pixels(sg)
        row = {"strength": st, "psnr": psnr(px, covers), "ssim": ssim(px, covers),
               "lpips": lpips(px, covers), "mf": mf_acc(sg),
               "dec_clean": dec_acc(sg), "dec_jpeg50": dec_acc(sg, ("jpeg", 50))}
        rows.append(row)
        lp = row["lpips"]
        print(f"strength={st:<7}: PSNR {row['psnr']:6.2f} dB SSIM {row['ssim']:.4f} "
              f"LPIPS {'n/a' if lp is None else f'{lp:.4f}'} | mf {row['mf']:.3f} "
              f"dec {row['dec_clean']:.3f} jpeg50 {row['dec_jpeg50']:.3f}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# strength 扫描（载密图质量 vs 准确率）\n\n")
        f.write(f"- 空间: {io.describe()}，pixel_res={pixel_res}\n")
        f.write(f"- 无嵌入往返上限: PSNR **{base:.2f} dB**\n")
        f.write(f"- 容量: {io.capacity_report(stego.total_embed_bits)}\n")
        f.write(f"- 样本 n={args.n}, S_hide={args.hide_steps}, S_rec={args.rec_steps}\n\n")
        f.write("| strength | PSNR | SSIM | LPIPS | mf | dec clean | dec jpeg50 |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in rows:
            lp = r["lpips"]
            f.write(f"| {r['strength']} | {r['psnr']:.2f} | {r['ssim']:.4f} | "
                    f"{'n/a' if lp is None else f'{lp:.4f}'} | {r['mf']:.3f} | "
                    f"{r['dec_clean']:.3f} | {r['dec_jpeg50']:.3f} |\n")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()

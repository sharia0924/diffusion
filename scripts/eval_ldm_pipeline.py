"""LDM 迁移端到端验证：潜空间隐藏 -> 复原 -> 图像级指标。

回答一个问题：**把扩散搬到隐空间后，载密图的 PSNR 到底能到多少**，
以及相比像素空间基线提升了多少、瓶颈在哪里。

链路（与 train_decoder.py 的信道一致）：
    cover(像素) --VAE encode--> z --DDIM 反演--> z_T --注入密钥图案--> z_T'
        --DDIM 采样--> 载密 latent --VAE decode--> 载密图
    复原：载密图 --VAE encode--> z --DDIM 反演(S_rec)--> 特征 --解码器--> 比特

用法:
  python scripts/eval_ldm_pipeline.py --vae-ckpt checkpoints/vae_cifar.pt \
      --ddpm-ckpt checkpoints/ddpm_latent.pt --out results/ldm_pipeline.md
  python scripts/eval_ldm_pipeline.py --pixel-baseline --ddpm-ckpt checkpoints/ddpm_cifar.pt
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from krd import RingDecoder
from krd.distortions import apply_attack
from krd.metrics import bit_accuracy, psnr, ssim
from krd.perceptual import lpips
from krd.utils import derive_nonces_from_keys, seed_everything, token_key
from krd.vae import build_vae
from scripts.eval_common import StegoIO, cifar_loader, gather_covers
from scripts.train_decoder import load_stego, stego_to_images


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddpm-ckpt", default="checkpoints/ddpm_latent.pt")
    ap.add_argument("--vae-ckpt", default="checkpoints/vae_cifar.pt")
    ap.add_argument("--decoder-ckpt", default="checkpoints/decoder_latent_best.pt")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--hide-steps", type=int, default=50)
    ap.add_argument("--rec-steps", type=int, default=50)
    ap.add_argument("--strengths", default="0.25,0.5,1.0")
    ap.add_argument("--pixel-res", type=int, default=None,
                    help="cover 的像素尺寸；隐空间模型必须与 VAE 训练时的 --resize 一致"
                         "（默认从 checkpoint 的 args.resize 推断，否则 32）")
    ap.add_argument("--pixel-baseline", action="store_true",
                    help="用像素空间模型跑同一套指标，作为对照")
    ap.add_argument("--out", default="results/ldm_pipeline.md")
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()

    # 步数默认值从解码器 config 回填（训练/评测工作点必须一致）
    _explicit = {"--hide-steps", "--rec-steps"} & set(sys.argv)
    try:
        import torch as _t
        if os.path.exists(args.decoder_ckpt):
            _dcfg = _t.load(args.decoder_ckpt, map_location="cpu",
                            weights_only=True)["config"]
            if "--hide-steps" not in _explicit and _dcfg.get("hide_steps"):
                args.hide_steps = int(_dcfg["hide_steps"])
            if "--rec-steps" not in _explicit and _dcfg.get("rec_steps") \
                    and hasattr(args, "rec_steps"):
                args.rec_steps = int(_dcfg["rec_steps"])
            print(f"[steps] 从解码器回填: hide={getattr(args, 'hide_steps', None)} "
                  f"rec={getattr(args, 'rec_steps', None)}", flush=True)
    except Exception:
        pass
    seed_everything(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _ck = torch.load(args.ddpm_ckpt, map_location="cpu", weights_only=False)
    _margs = _ck.get("args", {})
    pixel_res = args.pixel_res or _margs.get("resize") or 32

    # 先读解码器配置，再用同一套容量参数构造 stego（否则 bins_per_bit / 槽位数不匹配）
    dec, cfg = None, None
    if os.path.exists(args.decoder_ckpt):
        ck = torch.load(args.decoder_ckpt, map_location=device, weights_only=True)
        cfg = ck["config"]
        print(f"[decoder] 载入 {args.decoder_ckpt}: n_bits={cfg['n_bits']} ecc={cfg['ecc']} "
              f"bpb={cfg['bpb']} check={cfg['n_check_bits']} res={cfg['res']}", flush=True)
    else:
        print(f"[decoder] 未找到 {args.decoder_ckpt}，只测 matched-filter 接收", flush=True)

    _cfg = cfg or {}
    stego = load_stego(args.ddpm_ckpt, device, with_vae=not args.pixel_baseline,
                       n_bits=_cfg.get("n_bits", 16), ecc_reps=_cfg.get("ecc", 3),
                       bins_per_bit=_cfg.get("bpb", 2),
                       n_check_bits=_cfg.get("n_check_bits", 32),
                       inject_mode=_cfg.get("inject_mode", "replace"))
    io = StegoIO(stego, pixel_res=pixel_res)
    print(f"[space] {io.describe()}  pixel_res={pixel_res}", flush=True)

    if cfg is not None:
        dec = RingDecoder(2 * cfg["n_pairs"],
                          cfg["n_bits"] * cfg["ecc"] + cfg["n_check_bits"]).to(device)
        dec.load_state_dict(ck["decoder"])
        dec.eval()

    loader = cifar_loader(args.data_root, train=False, batch_size=args.batch,
                          resize=None if pixel_res == 32 else pixel_res)
    covers = gather_covers(loader, args.n).to(device)
    keys = [token_key() for _ in range(args.n)]
    nonces = derive_nonces_from_keys(keys, start=0)
    n_bits = (cfg or {}).get("n_bits", stego.n_bits)
    g = torch.Generator(device="cpu").manual_seed(0)
    bits = torch.randint(0, 2, (args.n, n_bits), generator=g).float().to(device)

    report = {"space": io.describe(), "n": args.n, "n_bits": n_bits,
              "pixel_baseline": args.pixel_baseline, "rows": []}
    if io.latent:
        report["vae_roundtrip_psnr"] = io.baseline_psnr(covers)
        report["capacity"] = io.capacity_report(n_bits)
        print(f"[latent] VAE 往返上限（无嵌入）PSNR {report['vae_roundtrip_psnr']:.2f} dB",
              flush=True)
        print(f"[latent] 容量报告 {report['capacity']}", flush=True)

    def mf_accuracy(z, keys_):
        """matched-filter 接收（不依赖训练解码器）：去旋取同相分量。"""
        from krd.pattern import ecc_collapse
        accs = []
        for i in range(z.shape[0]):
            p = stego.params_for(keys_[i], nonces[i])
            feats = io.recover_features(z[i:i + 1], [keys_[i]], args.rec_steps,
                                        nonces=[nonces[i]])[0]
            npairs = stego.n_pairs
            total = stego.total_embed_bits
            gg = npairs // total
            re = feats[:npairs].view(total, gg).mean(-1)
            lg = ecc_collapse(re[None, :stego.msg_embed_bits], n_bits, stego.ecc)
            accs.append(bit_accuracy(lg, bits[i][:n_bits][None]))
        return float(np.mean(accs))

    for st in [float(v) for v in args.strengths.split(",")]:
        sg = torch.cat([
            io.hide(covers[i:i + args.batch], bits[i:i + args.batch],
                    keys[i:i + args.batch], args.hide_steps, st,
                    nonces=nonces[i:i + args.batch])
            for i in range(0, args.n, args.batch)
        ])
        sg_px = io.to_pixels(sg)
        row = {"strength": st, "psnr": psnr(sg_px, covers), "ssim": ssim(sg_px, covers),
               "lpips": lpips(sg_px, covers)}
        if dec is not None:
            accs = []
            for i in range(0, args.n, args.batch):
                logits = io.recover(sg[i:i + args.batch], keys[i:i + args.batch],
                                    args.rec_steps, dec, nonces=nonces[i:i + args.batch])
                accs.append(bit_accuracy(logits, bits[i:i + args.batch, :n_bits]
                                         if bits.shape[1] > n_bits else bits[i:i + args.batch]))
            row["dec_clean"] = float(np.mean(accs))
            sg_j = io.attack(sg, lambda t: apply_attack(t, "jpeg", 50))
            accs = []
            for i in range(0, args.n, args.batch):
                logits = io.recover(sg_j[i:i + args.batch], keys[i:i + args.batch],
                                    args.rec_steps, dec, nonces=nonces[i:i + args.batch])
                accs.append(bit_accuracy(logits, bits[i:i + args.batch, :n_bits]
                                         if bits.shape[1] > n_bits else bits[i:i + args.batch]))
            row["dec_jpeg50"] = float(np.mean(accs))
        row["mf_clean"] = mf_accuracy(sg, keys)
        report["rows"].append(row)
        lp = row["lpips"]
        print(f"strength={st}: PSNR {row['psnr']:.2f} dB SSIM {row['ssim']:.4f} "
              f"LPIPS {'n/a' if lp is None else f'{lp:.4f}'} | "
              f"mf {row['mf_clean']:.3f} "
              f"dec_clean {row.get('dec_clean', float('nan')):.3f} "
              f"dec_jpeg50 {row.get('dec_jpeg50', float('nan')):.3f}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# LDM 迁移端到端验证\n\n")
        f.write(f"- 空间: {report['space']}\n- 样本数: {args.n}, 容量 {n_bits} bits, "
                f"S_hide={args.hide_steps}, S_rec={args.rec_steps}\n")
        if "vae_roundtrip_psnr" in report:
            f.write(f"- **VAE 往返上限（无嵌入）: PSNR "
                    f"{report['vae_roundtrip_psnr']:.2f} dB**\n")
        if report.get("capacity"):
            f.write(f"- 容量报告: `{report['capacity']}`\n")
        f.write("\n| strength | PSNR | SSIM | LPIPS | mf clean | dec clean | dec jpeg50 |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in report["rows"]:
            lp = r["lpips"]
            f.write(f"| {r['strength']} | {r['psnr']:.2f} | {r['ssim']:.4f} | "
                    f"{'n/a' if lp is None else f'{lp:.4f}'} | {r['mf_clean']:.3f} | "
                    f"{r.get('dec_clean', float('nan')):.3f} | "
                    f"{r.get('dec_jpeg50', float('nan')):.3f} |\n")
    with open(args.out.replace(".md", ".json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()

"""P2 评测：复原步数失配鲁棒性 —— 解码器在 S_rec=T 训练，在 S_rec≠T 时评测。

结论形态：轻微失配下 BER 平滑退化（协议只需"近似已知步数"）。
对照：可分别在 --rec-jitter 0 / 10 下训练两个解码器再各自跑本脚本。

v2 修复：nonce 协议改为自包含的 H(key||counter)；取数改用 eval_common。

  python scripts/eval_step_mismatch.py --ddpm-ckpt checkpoints/ddpm_cifar.pt \
      --decoder-ckpt checkpoints/decoder_best.pt --out results/step_mismatch.md
"""

import argparse
import csv
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from krd import RingDecoder
from krd.distortions import diff_jpeg
from krd.metrics import psnr
from krd.utils import seed_everything
from scripts.eval_common import cifar_loader, make_eval_inputs
from scripts.eval_steps_grid import decode_acc
from scripts.train_decoder import load_stego


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddpm-ckpt", default="checkpoints/ddpm_cifar.pt")
    ap.add_argument("--decoder-ckpt", default="checkpoints/decoder_best.pt")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--rec-list", default="20,30,40,45,50,55,60,70,80")
    ap.add_argument("--hide-steps", type=int, default=50)
    ap.add_argument("--pixel-res", type=int, default=None,
                    help="cover 像素尺寸；隐空间模型由 eval_setup 自动匹配")
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--nonce-start", type=int, default=0)
    ap.add_argument("--out", default="results/step_mismatch.md")
    ap.add_argument("--seed", type=int, default=13)
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
    rec_list = sorted({int(v) for v in args.rec_list.split(",")})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from scripts.eval_setup import eval_setup
    dec_ckpt = torch.load(args.decoder_ckpt, map_location=device, weights_only=True)
    cfg = dec_ckpt["config"]
    trained_sr = cfg.get("rec_steps", "?")
    stego, io, loader, cfg, pixel_res = eval_setup(
        args.ddpm_ckpt, args.data_root, args.batch, device, args.decoder_ckpt)
    print(f"[space] {io.describe()}  pixel_res={pixel_res}")
    dec = RingDecoder(2 * cfg["n_pairs"],
                      cfg["n_bits"] * cfg["ecc"] + cfg["n_check_bits"]).to(device)
    dec.load_state_dict(dec_ckpt["decoder"])
    dec.eval()

    covers, bits, keys, nonces = make_eval_inputs(
        loader, args.n, cfg["n_bits"], device, nonce_start=args.nonce_start)

    sg = torch.cat([
        io.hide(covers[i:i + args.batch], bits[i:i + args.batch],
                   keys[i:i + args.batch], args.hide_steps, args.strength,
                   nonces=nonces[i:i + args.batch])
        for i in range(0, args.n, args.batch)
    ])
    # JPEG 定义在像素域：隐空间模型先解码再加失真、再编码回去
    sg_j = io.attack(sg, lambda t: diff_jpeg(t, 50))
    print(f"stego PSNR {psnr(io.to_pixels(sg), covers):.2f} dB; decoder trained @ S_rec={trained_sr}")

    rows = []
    for sr in rec_list:
        a_c = decode_acc(stego, sg, keys, nonces, sr, dec, bits, args.batch, io=io)
        a_j = decode_acc(stego, sg_j, keys, nonces, sr, dec, bits, args.batch, io=io)
        rows.append((sr, a_c, a_j))
        print(f"S_rec={sr}: clean {a_c:.3f} | jpeg50 {a_j:.3f}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(f"# 复原步数失配鲁棒性（解码器训练于 S_rec={trained_sr}, "
                f"S_hide={args.hide_steps}, strength={args.strength}）\n\n"
                f"- 载密图 PSNR(vs cover) = {psnr(io.to_pixels(sg), covers):.2f} dB\n"
                f"- nonce 协议: `H(key || nonce_start+i)`, nonce_start={args.nonce_start}\n\n"
                "| S_rec | clean | jpeg50 |\n|---|---|---|\n")
        for sr, a_c, a_j in rows:
            f.write(f"| {sr} | {a_c:.3f} | {a_j:.3f} |\n")
        f.write("\n> 解读提示：若准确率在很宽的 S_rec 范围内保持 1.000，需先排除"
                "“水印信噪比过高导致失配不可见”的解释，再主张几何不变性。\n")
    with open(args.out.replace(".md", ".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["s_rec", "clean_acc", "jpeg50_acc"])
        for sr, a_c, a_j in rows:
            w.writerow([sr, f"{a_c:.4f}", f"{a_j:.4f}"])
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()

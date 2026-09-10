"""命令行隐藏/复原工具。

隐藏:  python scripts/run_stego.py hide --input cover.png --output stego.png \
           --key my-secret --message "ZC" --ddpm-ckpt checkpoints/ddpm_cifar.pt
复原:  python scripts/run_stego.py recover --input stego.png --key my-secret \
           --ddpm-ckpt checkpoints/ddpm_cifar.pt --decoder-ckpt checkpoints/decoder_best.pt \
           [--counter 0] [--message "ZC"  # 提供则报告比特准确率]

nonce 协议（v2, 自包含、盲提取）：nonce = H(key || counter)。
  - 隐藏端与复原端用同一个 --counter 即可，**不需要**读取 cover；
  - counter 是公开参数（相当于图序号），可随图传输；写入 PNG 的 "krd_counter" 仅为方便；
  - 旧协议用 H(cover) 派生 nonce，复原端必须依赖侧信道传输，已弃用（见 utils.derive_nonce）。
注: CIFAR 32x32 原型容量为 16 比特消息 = 2 字节（约 2 个 ASCII 字符）。
"""

import argparse
import os
import sys

import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from krd import RingDecoder
from krd.metrics import bit_accuracy, psnr, ssim
from krd.perceptual import lpips
from krd.utils import (bits_to_str, derive_nonce_from_key, image_to_tensor, str_to_bits,
                       tensor_to_image)
from scripts.train_decoder import load_stego


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["hide", "recover"])
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="stego.png")
    ap.add_argument("--key", required=True)
    ap.add_argument("--message", default="")
    ap.add_argument("--counter", type=int, default=0,
                    help="nonce 计数器（公开参数）：nonce = H(key || counter)")
    ap.add_argument("--ddpm-ckpt", default="checkpoints/ddpm_cifar.pt")
    ap.add_argument("--decoder-ckpt", default="checkpoints/decoder_best.pt")
    ap.add_argument("--hide-steps", type=int, default=50)
    ap.add_argument("--rec-steps", type=int, default=50)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--size", type=int, default=32)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    nonce = derive_nonce_from_key(args.key, args.counter)

    if args.mode == "hide":
        if not args.message:
            raise SystemExit("hide 需要 --message")
        stego = load_stego(args.ddpm_ckpt, device)
        cover = image_to_tensor(Image.open(args.input), size=args.size).to(device)
        bits = str_to_bits(args.message, stego.n_bits)[None].to(device)
        sg = stego.hide(cover, bits, [args.key], args.hide_steps, args.strength,
                        nonces=[nonce])
        meta = PngInfo()
        meta.add_text("krd_counter", str(args.counter))   # 仅为方便复原端取回计数器
        tensor_to_image(sg).save(args.output, pnginfo=meta)
        lp = lpips(sg, cover)
        lp_s = "n/a" if lp is None else f"{lp:.4f}"
        print(f"stego -> {args.output}  PSNR={psnr(sg, cover):.2f}dB "
              f"SSIM={ssim(sg, cover):.4f} LPIPS={lp_s}  counter={args.counter}")
    else:
        img = Image.open(args.input)
        counter = args.counter
        meta_counter = (img.text or {}).get("krd_counter")
        if meta_counter is not None and "--counter" not in sys.argv:
            counter = int(meta_counter)
            nonce = derive_nonce_from_key(args.key, counter)
            print(f"（从 PNG 元数据读取 counter={counter}）")
        dec_ckpt = torch.load(args.decoder_ckpt, map_location=device, weights_only=True)
        cfg = dec_ckpt["config"]
        stego = load_stego(args.ddpm_ckpt, device,
                           n_bits=cfg["n_bits"], ecc_reps=cfg["ecc"],
                           bins_per_bit=cfg["bpb"], n_check_bits=cfg["n_check_bits"])
        dec = RingDecoder(2 * cfg["n_pairs"],
                          cfg["n_bits"] * cfg["ecc"] + cfg["n_check_bits"]).to(device)
        dec.load_state_dict(dec_ckpt["decoder"])
        dec.eval()
        stego_img = image_to_tensor(img, size=args.size).to(device)
        logits = stego.recover(stego_img, [args.key], args.rec_steps, dec,
                               nonces=[nonce])[0]
        rec = bits_to_str((logits > 0).float())
        raw = "".join(str(int(b)) for b in (logits > 0).int().tolist())
        print(f"decoded: {rec!r}  (raw bits {raw}, counter={counter})")
        if args.message:
            ref = str_to_bits(args.message, cfg["n_bits"]).to(device)
            print(f"bit accuracy vs '{args.message}': "
                  f"{bit_accuracy(logits[None], ref[None]):.3f}")


if __name__ == "__main__":
    main()

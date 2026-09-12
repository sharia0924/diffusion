"""命令行隐藏/复原工具（支持像素空间与隐空间模型）。

隐藏:  python scripts/run_stego.py hide --input cover.png --output stego.png \
           --key my-secret --message "ZC" --ddpm-ckpt checkpoints/ddpm_cifar.pt
复原:  python scripts/run_stego.py recover --input stego.png --key my-secret \
           --ddpm-ckpt checkpoints/ddpm_cifar.pt --decoder-ckpt checkpoints/decoder_best.pt \
           [--counter 0] [--message "ZC"]

隐空间模型（--ddpm-ckpt 指向 latent 模型时）：
  - 像素尺寸由 VAE 的 --resize 自动决定（**不要手写 --size**，写错会得到尺寸不符的
    潜变量，图案频率网格与张量尺寸不匹配）；
  - 隐藏走 StegoIO 桥接：cover -> VAE encode -> 注入 -> 采样 -> VAE decode；
  - 复原同理：图片 -> encode -> 反演 -> 特征 -> 解码器。
  （历史上本脚本硬编码 32×32 且绕过 StegoIO 直接调 stego.hide/recover，
    隐空间模型下必然出错。）

nonce 协议（v2, 自包含、盲提取）：nonce = H(key || counter)。
  - 隐藏端与复原端用同一个 --counter 即可，**不需要**读取 cover；
  - counter 是公开参数，写入 PNG 的 "krd_counter" 仅为方便；
注: 容量由模型与解码器配置决定（像素空间原型为 16 比特 = 2 字节）。
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
from scripts.eval_setup import eval_setup, load_decoder_cfg


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
    ap.add_argument("--size", type=int, default=None,
                    help="覆盖像素尺寸；默认由 checkpoint 自动推断（隐空间=VAE 的 resize）")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    nonce = derive_nonce_from_key(args.key, args.counter)

    # 统一入口：自动匹配像素尺寸、容量参数、注入模式与是否走 VAE 桥接
    stego, io, _, cfg, pixel_res = eval_setup(
        args.ddpm_ckpt, "./data", 1, device, args.decoder_ckpt)
    size = args.size or pixel_res
    print(f"[space] {io.describe()}  pixel_res={size}", flush=True)

    def load_image(path):
        img = Image.open(path)
        if not io.latent and size != 32:
            print(f"[warn] 像素空间模型通常用 32×32，当前 --size={size}", flush=True)
        return img, image_to_tensor(img, size=size).to(device)

    if args.mode == "hide":
        if not args.message:
            raise SystemExit("hide 需要 --message")
        _, cover = load_image(args.input)
        bits = str_to_bits(args.message, stego.n_bits)[None].to(device)
        sg = io.hide(cover, bits, [args.key], args.hide_steps, args.strength,
                     nonces=[nonce])
        sg_px = io.to_pixels(sg)
        meta = PngInfo()
        meta.add_text("krd_counter", str(args.counter))   # 仅为方便复原端取回计数器
        tensor_to_image(sg_px).save(args.output, pnginfo=meta)
        lp = lpips(sg_px, cover)
        lp_s = "n/a" if lp is None else f"{lp:.4f}"
        print(f"stego -> {args.output}  PSNR={psnr(sg_px, cover):.2f}dB "
              f"SSIM={ssim(sg_px, cover):.4f} LPIPS={lp_s}  counter={args.counter}")
    else:
        img = Image.open(args.input)
        counter = args.counter
        meta_counter = (img.text or {}).get("krd_counter")
        if meta_counter is not None and "--counter" not in sys.argv:
            counter = int(meta_counter)
            nonce = derive_nonce_from_key(args.key, counter)
            print(f"（从 PNG 元数据读取 counter={counter}）")
        if cfg is None:
            raise SystemExit(f"recover 需要解码器 checkpoint：{args.decoder_ckpt} 不存在")
        dec = RingDecoder(2 * cfg["n_pairs"],
                          cfg["n_bits"] * cfg["ecc"] + cfg["n_check_bits"]).to(device)
        dec.load_state_dict(torch.load(args.decoder_ckpt, map_location=device,
                                       weights_only=True)["decoder"])
        dec.eval()
        _, stego_img = load_image(args.input)
        logits = io.recover(stego_img, [args.key], args.rec_steps, dec,
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

"""阶段 2：训练密钥条件复原解码器（冻结 DDPM，只训 MLP 头）。

训练循环：cover -> hide(随机比特/密钥/强度/nonce) -> 随机失真 -> 固定步数 DDIM 反演
          -> 密钥门控环带特征 -> RingDecoder -> BCE。
嵌入向量 = ECC(消息) + 密钥校验比特；解码器同时学习两者。
失真下的隐式训练 + 密钥随机化 = 鲁棒性与密钥安全性的来源；
--sched-noise-prob 开启"调度坐标加噪"（扩散再生攻击的训练代理）。

用法:
  python scripts/train_decoder.py --ddpm-ckpt checkpoints/ddpm_cifar.pt --steps 3000
  python scripts/train_decoder.py --tiny   # 冒烟
"""

import argparse
import os
import random
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from krd import RingDecoder, Schedule, TrajStego, UNet
from krd.distortions import apply_attack, random_distortion
from krd.metrics import bit_accuracy
from krd.utils import (derive_nonces_from_keys, resolve_num_workers, seed_everything,
                       token_key)


def load_stego(ckpt_path: str, device: str, n_bits: int = 16, ecc_reps: int = 3,
               bins_per_bit: int = 2, n_check_bits: int = 32) -> TrajStego:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    margs = ckpt.get("args", {})
    unet = UNet(base=margs.get("base", 64)).to(device)
    unet.load_state_dict(ckpt.get("ema", ckpt["model"]))
    sched = Schedule(margs.get("timesteps", 1000), device=device)
    return TrajStego(unet, sched, n_bits=n_bits, ecc_reps=ecc_reps,
                     bins_per_bit=bins_per_bit, n_check_bits=n_check_bits, device=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddpm-ckpt", default="checkpoints/ddpm_cifar.pt")
    ap.add_argument("--out", default="checkpoints/decoder.pt")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hide-steps", type=int, default=50)
    ap.add_argument("--rec-steps", type=int, default=50)
    ap.add_argument("--rec-jitter", type=int, default=0,
                    help=">0 时复原步数在 [rec-jitter, +jitter] 随机（步数灵活性鲁棒）")
    ap.add_argument("--strength-min", type=float, default=0.6)
    ap.add_argument("--strength-max", type=float, default=1.4)
    ap.add_argument("--distort-prob", type=float, default=0.8)
    ap.add_argument("--geom-prob", type=float, default=0.25,
                    help="随机失真中选中几何攻击（真裁剪/旋转/缩放/平移）的条件概率，"
                         "与 eval_robustness 的评测口径对齐")
    ap.add_argument("--sched-noise-prob", type=float, default=0.0,
                    help=">0 时以该概率把失真换成调度坐标加噪（扩散再生攻击的训练代理）")
    ap.add_argument("--n-check-bits", type=int, default=32)
    ap.add_argument("--eval-size", type=int, default=128)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--num-workers", type=int, default=-1,
                    help="DataLoader worker 数；-1 = 自动探测（受限沙箱下自动回退 0）")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    seed_everything(args.seed)
    rng = random.Random(args.seed)
    if args.tiny:
        args.steps = min(args.steps, 30)
        args.batch_size = min(args.batch_size, 4)
        args.eval_size = 16

    device = "cuda" if torch.cuda.is_available() else "cpu"
    stego = load_stego(args.ddpm_ckpt, device, n_check_bits=args.n_check_bits)
    decoder = RingDecoder(2 * stego.n_pairs, stego.total_embed_bits).to(device)
    opt = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=1e-4)

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    ds = datasets.CIFAR10(args.data_root, train=True, download=True, transform=tf)
    if args.tiny:
        ds = Subset(ds, range(64))
    num_workers = resolve_num_workers(args.num_workers)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=num_workers, pin_memory=(device == "cuda"),
                        persistent_workers=num_workers > 0)
    it = iter(loader)

    # 固定评测批（nonce 协议与部署一致：nonce = H(key || counter)，不依赖 cover）
    eval_ds = datasets.CIFAR10(args.data_root, train=False, download=True, transform=tf)
    eval_x = torch.stack([eval_ds[i][0] for i in range(args.eval_size)]).to(device)
    eval_keys = [token_key(rng) for _ in range(args.eval_size)]
    eval_nonces = derive_nonces_from_keys(eval_keys, start=0)

    def evaluate() -> dict:
        decoder.eval()
        n = args.eval_size
        bits = torch.randint(0, 2, (n, stego.n_bits), device=device).float()
        keys = eval_keys
        sg = stego.hide(eval_x, bits, keys, args.hide_steps, strength=1.0,
                        nonces=eval_nonces)
        out = {}
        for name, atk in [("clean", "clean"), ("jpeg50", ("jpeg", 50)),
                          ("noise05", ("noise", 0.05))]:
            x_in = sg if atk == "clean" else apply_attack(sg, atk[0], atk[1])
            rs = args.rec_steps
            if args.rec_jitter > 0:
                rs += rng.randint(-args.rec_jitter, args.rec_jitter)
            logits = stego.recover(x_in, keys, rs, decoder, nonces=eval_nonces)
            out[name] = bit_accuracy(logits, bits)
        decoder.train()
        return out

    best = -1.0
    config = {"n_bits": stego.n_bits, "ecc": stego.ecc, "bpb": stego.bpb,
              "res": stego.res, "n_check_bits": stego.n_check_bits,
              "n_pairs": stego.n_pairs, "hide_steps": args.hide_steps,
              "rec_steps": args.rec_steps, "nonce_protocol": "keyed-v2",
              "geom_prob": args.geom_prob}
    for step in range(1, args.steps + 1):
        try:
            x0, _ = next(it)
        except StopIteration:
            it = iter(loader)
            x0, _ = next(it)
        x0 = x0.to(device)
        B = x0.shape[0]
        bits = torch.randint(0, 2, (B, stego.n_bits), device=device).float()
        keys = [token_key(rng) for _ in range(B)]
        nonces = derive_nonces_from_keys(keys, start=step * 1000)
        strength = torch.empty(B, device=device).uniform_(args.strength_min, args.strength_max)

        sg = stego.hide(x0, bits, keys, args.hide_steps, strength=strength, nonces=nonces)
        x_in = torch.stack([
            random_distortion(sg[i:i + 1], rng, schedule=stego.sched,
                              sched_noise_prob=args.sched_noise_prob,
                              geom_prob=args.geom_prob)[0]
            if rng.random() < args.distort_prob else sg[i]
            for i in range(B)
        ])
        rs = args.rec_steps
        if args.rec_jitter > 0:
            rs = max(8, rs + rng.randint(-args.rec_jitter, args.rec_jitter))

        feats = stego.recover_features(x_in, keys, rs, nonces=nonces)
        logits_all = decoder(feats)
        bits_full = torch.stack([stego.full_bits(bits[i], keys[i], nonces[i])
                                 for i in range(B)])
        loss = F.binary_cross_entropy_with_logits(logits_all, bits_full)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 50 == 0:
            msg_acc = bit_accuracy(stego.collapse(logits_all), bits)
            print(f"step {step}/{args.steps} loss {loss.item():.4f} train-bitacc {msg_acc:.3f}",
                  flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            res = evaluate()
            print(f"[eval] step {step} " +
                  " ".join(f"{k}={v:.3f}" for k, v in res.items()), flush=True)
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            torch.save({"decoder": decoder.state_dict(), "config": config}, args.out)
            if res.get("jpeg50", 0) >= best:
                best = res.get("jpeg50", 0)
                torch.save({"decoder": decoder.state_dict(), "config": config},
                           args.out.replace(".pt", "_best.pt"))
    print(f"saved -> {args.out} (best jpeg50 acc {best:.3f})")


if __name__ == "__main__":
    main()

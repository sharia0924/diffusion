"""阶段 1a：训练隐空间 VAE（LDM 迁移的第一步）。

把图像压到隐空间 (B, z_ch, H/down, W/down)，供 `train_latent_ddpm.py` 在其上训练
扩散模型。支持两种来源：

  1. 本项目原生 VAE（默认，无额外依赖）—— 用本脚本从头训练；
  2. 预训练 SD VAE —— 不需要本脚本，直接在 train_latent_ddpm.py 里
     用 --vae-backend sd 编码。

用法:
  python scripts/train_vae.py --epochs 30 --batch-size 128 --downsample 3 --z-ch 4
  python scripts/train_vae.py --tiny          # 冒烟

输出: checkpoints/vae_cifar.pt （含 args，供 VAEWrapper 重建结构）
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from krd.metrics import psnr
from krd.utils import resolve_num_workers, seed_everything
from krd.vae import NativeVAE


def build_dataset(data_root: str, tiny: bool, size: int | None = None):
    tf = [transforms.ToTensor(),
          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    if size is not None:
        tf.insert(0, transforms.Resize(size, antialias=True))
    ds = datasets.CIFAR10(data_root, train=True, download=True,
                          transform=transforms.Compose(tf))
    if tiny:
        ds = Subset(ds, range(1024))
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--out", default="checkpoints/vae_cifar.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--z-ch", type=int, default=4, help="隐变量通道数")
    ap.add_argument("--downsample", type=int, default=3, help="空间压缩 2^downsample")
    ap.add_argument("--ch-mults", default="1,2,4,8")
    ap.add_argument("--kl-weight", type=float, default=1e-4,
                    help="KL 项权重（小值 -> 接近确定性自编码器，重建更保真）")
    ap.add_argument("--resize", type=int, default=None, help="先把图像缩放到该尺寸")
    ap.add_argument("--num-workers", type=int, default=-1)
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    seed_everything(args.seed)

    if args.tiny:
        args.epochs = min(args.epochs, 2)
        args.base = min(args.base, 16)
        args.ch_mults = "1,2,4"
        args.downsample = 2
        args.batch_size = min(args.batch_size, 32)
    ch_mults = tuple(int(v) for v in args.ch_mults.split(","))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = build_dataset(args.data_root, args.tiny, args.resize)
    num_workers = resolve_num_workers(args.num_workers)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=num_workers, pin_memory=(device == "cuda"),
                        persistent_workers=num_workers > 0)

    model = NativeVAE(in_ch=3, base=args.base, z_ch=args.z_ch,
                      downsample=args.downsample, ch_mults=ch_mults).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[setup] device={device} params={n_par / 1e6:.2f}M z_ch={args.z_ch} "
          f"down={args.downsample} ch_mults={ch_mults} kl={args.kl_weight}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    resume_path = args.out.replace(".pt", ".last.pt")
    start_epoch = 0
    if os.path.exists(resume_path):
        st = torch.load(resume_path, map_location=device, weights_only=False)
        if int(st.get("epoch", 0)) < args.epochs:
            model.load_state_dict(st["model"])
            if "opt" in st:
                opt.load_state_dict(st["opt"])
            start_epoch = int(st["epoch"])
            print(f"[resume] 从 {resume_path} 恢复: 已完成 {start_epoch} epoch", flush=True)

    def save_state(ep: int):
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep,
                    "args": vars(args)}, resume_path)

    step = 0
    for epoch in range(start_epoch, args.epochs):
        model.train()
        agg = {"rec": 0.0, "kl": 0.0, "psnr": 0.0}
        nb = 0
        for x, _ in loader:
            x = x.to(device)
            y, mu, logvar = model(x)
            rec = F.mse_loss(y, x)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = rec + args.kl_weight * kl

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            with torch.no_grad():
                agg["rec"] += rec.item()
                agg["kl"] += kl.item()
                agg["psnr"] += psnr(y.clamp(-1, 1), x)
            nb += 1
            step += 1
            if step % args.log_every == 0:
                print(f"epoch {epoch} step {step} rec {agg['rec'] / nb:.4f} "
                      f"kl {agg['kl'] / nb:.4f} psnr {agg['psnr'] / nb:.2f} dB", flush=True)
        print(f"[epoch {epoch}] rec_mse {agg['rec'] / max(nb, 1):.4f} "
              f"kl {agg['kl'] / max(nb, 1):.4f} 重建 PSNR {agg['psnr'] / max(nb, 1):.2f} dB",
              flush=True)
        if args.save_every and (epoch + 1) % args.save_every == 0:
            save_state(epoch + 1)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"model": model.state_dict(), "args": vars(args), "epochs_done": args.epochs},
               args.out)
    save_state(args.epochs)
    print(f"saved -> {args.out}")

    # 收尾：报告该 VAE 的往返保真度（决定整条 LDM 链路的 PSNR 上限）
    model.eval()
    with torch.no_grad():
        xs = torch.stack([ds[i][0] for i in range(min(32, len(ds)))]).to(device)
        z = model.encode(xs)
        y = model.decode(z, target_hw=(xs.shape[-2], xs.shape[-1]))
        print(f"[summary] latent {tuple(z.shape)}  往返 PSNR {psnr(y.clamp(-1, 1), xs):.2f} dB "
              f"(这是纯 VAE 上限；扩散模型还会再引入误差)", flush=True)


if __name__ == "__main__":
    main()

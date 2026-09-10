"""阶段 1：在 CIFAR-10 上预训练基础 DDPM（U-Net + eps 预测）。

用法:
  python scripts/train_ddpm.py --epochs 60 --batch-size 128
  python scripts/train_ddpm.py --tiny          # 冒烟: 少量数据/轮数, 验证流程
"""

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from krd import Schedule, UNet
from krd.utils import seed_everything


def build_dataset(data_root: str, tiny: bool):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    ds = datasets.CIFAR10(data_root, train=True, download=True, transform=tf)
    if tiny:
        ds = Subset(ds, range(2048))
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--out", default="checkpoints/ddpm_cifar.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--ema", type=float, default=0.9995)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    seed_everything(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = build_dataset(args.data_root, args.tiny)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=2, pin_memory=(device == "cuda"), drop_last=True)
    if args.tiny:
        args.epochs = min(args.epochs, 2)
        args.base = min(args.base, 32)

    model = UNet(base=args.base).to(device)
    sched = Schedule(args.timesteps, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}

    step = 0
    for epoch in range(args.epochs):
        running = 0.0
        for x, _ in loader:
            x = x.to(device)
            t = torch.randint(0, sched.T, (x.shape[0],), device=device)
            noise = torch.randn_like(x)
            x_t = sched.add_noise(x, t, noise)
            pred = model(x_t, t)
            loss = torch.nn.functional.mse_loss(pred, noise)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    ema[k].mul_(args.ema).add_(v.detach(), alpha=1 - args.ema)
            running += loss.item()
            step += 1
            if step % 100 == 0:
                print(f"epoch {epoch} step {step} loss {running / 100:.4f}", flush=True)
                running = 0.0

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "ema": {k: v for k, v in ema.items()},
        "args": vars(args),
    }, args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()

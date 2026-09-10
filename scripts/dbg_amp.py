"""定位 AMP(fp16 autocast) 下 U-Net 输出变 nan 的具体层。"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from krd import Schedule, UNet
from krd.unet import ResBlock, SelfAttention
from krd.utils import seed_everything


def rep(tag, h):
    ok = bool(torch.isfinite(h).all())
    print(f"  {tag:34s} dtype={str(h.dtype):15s} finite={ok} "
          f"absmax={h.abs().max().item():10.4g}", flush=True)
    return ok


def main():
    seed_everything(0)
    dev = "cuda"
    base = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    fp32_rb = len(sys.argv) > 2 and sys.argv[2] == "fp32rb"
    model = UNet(base=base).to(dev).eval()
    model.fp32_resblocks = fp32_rb
    print(f"fp32_resblocks={fp32_rb}", flush=True)
    sched = Schedule(1000, device=dev)
    x = torch.randn(8, 3, 32, 32, device=dev)
    t = torch.randint(0, 1000, (8,), device=dev)
    xt = sched.add_noise(x, t, torch.randn_like(x))

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        out = model(xt, t)
    rep("UNet.forward (autocast)", out)

    # 训练一步（含反向），检查梯度是否溢出
    model.train()
    o = torch.optim.AdamW(model.parameters(), lr=2e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    xt2 = sched.add_noise(x, t, torch.randn_like(x))
    noise = torch.randn_like(x)
    with torch.autocast("cuda", dtype=torch.float16):
        pred = model(xt2, t)
    loss = torch.nn.functional.mse_loss(pred.float(), noise)
    o.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9).item()
    print(f"  train step: loss={loss.item():.4f} grad_norm={gn:.4g}", flush=True)


if __name__ == "__main__":
    main()

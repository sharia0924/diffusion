"""定位 train_ddpm 每步耗时（本机 GTX 1650 4GB）的瓶颈：前向/反向/优化器/EMA。"""

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from krd import Schedule, UNet
from krd.utils import seed_everything


def timeit(fn, n=20, warmup=5, label=""):
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = (time.time() - t0) / n
    print(f"{label:42s} {dt * 1000:8.1f} ms/step", flush=True)
    return dt


def main():
    seed_everything(0)
    dev = "cuda"
    bs = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    print(f"device={dev} batch={bs}", flush=True)
    print("cudnn.benchmark:", torch.backends.cudnn.benchmark,
          "| cudnn.enabled:", torch.backends.cudnn.enabled,
          "| tf32:", torch.backends.cuda.matmul.allow_tf32, flush=True)

    model = UNet(base=64).to(dev)
    sched = Schedule(1000, device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    x = torch.randn(bs, 3, 32, 32, device=dev)
    t = torch.randint(0, 1000, (bs,), device=dev)
    noise = torch.randn_like(x)
    xt = sched.add_noise(x, t, noise)

    # 1) 纯前向
    model.eval()
    with torch.no_grad():
        timeit(lambda: model(xt, t), label="forward only (no_grad)")

    # 2) 前向 + 反向
    model.train()
    def fwd_bwd():
        loss = torch.nn.functional.mse_loss(model(xt, t), noise)
        opt.zero_grad(set_to_none=True)
        loss.backward()
    timeit(fwd_bwd, label="forward + backward (no opt.step)")

    # 3) 完整一步
    def full():
        loss = torch.nn.functional.mse_loss(model(xt, t), noise)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    timeit(full, label="forward + backward + AdamW")

    # 4) 只测优化器
    timeit(lambda: opt.step(), n=50, label="AdamW only")

    # 5) 只测 EMA（旧 train_ddpm 的 state_dict 循环写法）
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    def ema_loop():
        with torch.no_grad():
            for k, v in model.state_dict().items():
                ema[k].mul_(0.9995).add_(v.detach(), alpha=1 - 0.9995)
    timeit(ema_loop, n=50, label="EMA via state_dict loop")

    # 6) EMA 的 foreach 写法（快很多）
    ema_items = list(ema.items())
    params = dict(model.named_parameters())
    def ema_foreach():
        with torch.no_grad():
            for k, v in ema_items:
                src = params.get(k.split(".")[-1], None)
                v.mul_(0.9995).add_(v, alpha=0.0)  # 占位，避免优化器影响
    timeit(lambda: None, n=50, label="(noop 基线)")

    # 7) 合并：full + ema_loop（复现 train_ddpm 的真实每步）
    def full_with_ema():
        loss = torch.nn.functional.mse_loss(model(xt, t), noise)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        ema_loop()
    timeit(full_with_ema, label="train_ddpm 真实每步 (full + EMA)")


if __name__ == "__main__":
    main()

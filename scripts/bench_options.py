"""实测本机可用的加速项：batch 大小 / channels_last / cudnn.benchmark。

AMP(fp16) 在本 U-Net 上会溢出（up-path ResBlock，即使选择性 fp32 仍 nan），
因此这里不测 AMP，只量化"放大 batch + 内存格式 + cudnn autotune"的收益。
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from krd import Schedule, UNet
from krd.utils import seed_everything

CIFAR_TRAIN = 50000


def bench(base, batch, channels_last, cudnn_bench, steps, warmup, lr):
    dev = "cuda"
    model = UNet(base=base).to(dev)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    sched = Schedule(1000, device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    x = torch.randn(batch, 3, 32, 32, device=dev)
    if channels_last:
        x = x.to(memory_format=torch.channels_last)
    t = torch.randint(0, 1000, (batch,), device=dev)
    noise = torch.randn_like(x)
    xt = sched.add_noise(x, t, noise)

    def step():
        loss = torch.nn.functional.mse_loss(model(xt, t), noise)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    try:
        for _ in range(warmup):
            step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for _ in range(steps):
            step()
        torch.cuda.synchronize()
        dt = (time.time() - t0) / steps
        peak = torch.cuda.max_memory_allocated() / 2**30
        return dt, peak, "ok"
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return float("nan"), float("nan"), "OOM"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--batches", default="128,256,512,1024")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--out", default="results/bench_options.json")
    args = ap.parse_args()
    seed_everything(0)
    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA")
    p = torch.cuda.get_device_properties(0)
    print(f"gpu={p.name} mem={p.total_memory / 2**30:.1f}GiB torch={torch.__version__}",
          flush=True)

    rows = []
    for cl in (False, True):
        for cb in (False, True):
            torch.backends.cudnn.benchmark = cb
            for bs in [int(v) for v in args.batches.split(",")]:
                dt, peak, status = bench(args.base, bs, cl, cb, args.steps, args.warmup,
                                         args.lr)
                ips = (bs / dt) if status == "ok" else float("nan")
                ep_min = (CIFAR_TRAIN / ips / 60) if status == "ok" else float("nan")
                rows.append({"batch": bs, "channels_last": cl, "cudnn_benchmark": cb,
                             "status": status, "ms_step": dt * 1000, "img_per_s": ips,
                             "peak_gib": peak, "epoch_min": ep_min,
                             "300ep_h": ep_min * 300 / 60 if status == "ok" else float("nan")})
                print(f"  bs={bs:<5} ch_last={str(cl):<5} cudnn_bench={str(cb):<5} "
                      f"{status:<4} {dt * 1000:8.1f} ms/step {ips:7.1f} img/s "
                      f"peak {peak:5.2f}GiB 300ep {ep_min * 300 / 60:5.2f}h", flush=True)
                if status == "OOM" and bs >= 512:
                    break  # 更大只会更 OOM
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    ok = [r for r in rows if r["status"] == "ok"]
    if ok:
        best = max(ok, key=lambda r: r["img_per_s"])
        print(f"\n最优: batch={best['batch']} channels_last={best['channels_last']} "
              f"cudnn_benchmark={best['cudnn_benchmark']} -> {best['img_per_s']:.1f} img/s, "
              f"300 epoch ≈ {best['300ep_h']:.1f} h")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()

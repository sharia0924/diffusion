"""本机/远程吞吐基准与显存探测：为 train_ddpm / train_decoder 选 batch 提供依据。

两种用法：

1) 单点基准（默认）—— 报告一步训练/推理耗时与外推时间：
     python scripts/bench_local.py --batch 128 --decoder-batch 16

2) batch 扫描 + AMP 开关 —— 找出目标显卡上"内存装得下、吞吐最高"的配置：
     python scripts/bench_local.py --probe-batches 128,256,512,1024 --amp both
   每个组合报告：OOM 与否、峰值显存、step/s、images/s、时间外推。

说明：`train_ddpm` 的一步 = 前向 + 反向 + AdamW + EMA；本项目实测瓶颈在显存带宽，
所以 batch 增大后 images/s 会显著提升（小 batch 时 GPU 利用率很低）。
"""

import argparse
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from krd import Schedule, UNet
from krd.metrics import psnr
from krd.utils import seed_everything

CIFAR_TRAIN = 50000


def _timeit(fn, n, warmup, device):
    for _ in range(warmup):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(n):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) / n
    peak = (torch.cuda.max_memory_allocated() / 2**30) if device == "cuda" else 0.0
    return dt, peak


def probe_batches(batches, amps, base, steps, warmup, lr):
    """扫描 (batch, amp) 组合，报告是否 OOM、峰值显存与吞吐。"""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev != "cuda":
        print("无 CUDA，batch 扫描无意义")
        return []
    p = torch.cuda.get_device_properties(0)
    print(f"gpu={p.name} mem={p.total_memory / 2**30:.1f}GiB  torch={torch.__version__}")
    rows = []
    for amp in amps:
        for bs in batches:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            model = UNet(base=base).to(dev)
            sched = Schedule(1000, device=dev)
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            scaler = torch.cuda.amp.GradScaler(enabled=amp)
            x = torch.randn(bs, 3, 32, 32, device=dev)
            t = torch.randint(0, 1000, (bs,), device=dev)
            noise = torch.randn_like(x)
            xt = sched.add_noise(x, t, noise)

            def one_step():
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                    loss = torch.nn.functional.mse_loss(model(xt, t), noise)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                with torch.no_grad():
                    for k, v in model.state_dict().items():
                        v.mul_(1.0)  # EMA 占比极小，这里不模拟

            status = "ok"
            try:
                dt, peak = _timeit(one_step, steps, warmup, dev)
            except torch.cuda.OutOfMemoryError:
                status, dt, peak = "OOM", float("nan"), float("nan")
                torch.cuda.empty_cache()
            except RuntimeError as e:
                status = "ERR:" + str(e)[:40]
                dt, peak = float("nan"), float("nan")
                torch.cuda.empty_cache()

            ips = (bs / dt) if status == "ok" else float("nan")
            epoch_min = (CIFAR_TRAIN / ips / 60) if status == "ok" else float("nan")
            rows.append({"batch": bs, "amp": amp, "status": status, "ms_step": dt * 1000,
                         "img_per_s": ips, "peak_gib": peak, "epoch_min": epoch_min,
                         "300ep_h": epoch_min * 300 / 60 if status == "ok" else float("nan")})
            print(f"  batch={bs:<5} amp={str(amp):<5} {status:<6} "
                  f"{dt * 1000:8.1f} ms/step  {ips:7.1f} img/s  "
                  f"peak {peak:5.2f} GiB  epoch {epoch_min:5.2f} min  "
                  f"300ep {epoch_min * 300 / 60:5.2f} h", flush=True)
            del model, opt, scaler, x, t, noise, xt
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--decoder-batch", type=int, default=16)
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--roundtrip-batch", type=int, default=16)
    ap.add_argument("--roundtrip-steps", type=int, default=50)
    ap.add_argument("--probe-batches", default=None,
                    help="逗号分隔的 batch 列表；给定时进入扫描模式")
    ap.add_argument("--amp", choices=["both", "on", "off"], default="both",
                    help="扫描模式下要试的精度")
    args = ap.parse_args()
    seed_everything(0)

    if args.probe_batches:
        batches = [int(v) for v in args.probe_batches.split(",")]
        amps = {"both": [False, True], "on": [True], "off": [False]}[args.amp]
        if args.amp == "both" and not torch.cuda.is_available():
            amps = [False]
        rows = probe_batches(batches, amps, args.base, args.steps, args.warmup, args.lr)
        os.makedirs("results", exist_ok=True)
        with open("results/bench_batches.json", "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        ok = [r for r in rows if r["status"] == "ok"]
        if ok:
            best = max(ok, key=lambda r: r["img_per_s"])
            print(f"\n最优: batch={best['batch']} amp={best['amp']} "
                  f"-> {best['img_per_s']:.1f} img/s, epoch {best['epoch_min']:.2f} min, "
                  f"300 epoch ≈ {best['300ep_h']:.1f} h")
            print(f"建议命令: --batch-size {best['batch']} "
                  f"{'--amp on' if best['amp'] else '--amp off'} "
                  f"--cudnn-benchmark --channels-last")
        print("saved -> results/bench_batches.json")
        return

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}", flush=True)
    if dev == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)} "
              f"mem={torch.cuda.get_device_properties(0).total_memory / 2**30:.1f}GiB",
              flush=True)

    model = UNet(base=args.base).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    sched = Schedule(1000, device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    x = torch.randn(args.batch, 3, 32, 32, device=dev)
    t = torch.randint(0, 1000, (args.batch,), device=dev)
    noise = torch.randn_like(x)
    xt = sched.add_noise(x, t, noise)

    def fwd_bwd():
        loss = torch.nn.functional.mse_loss(model(xt, t), noise)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    dt, peak = _timeit(fwd_bwd, args.steps, args.warmup, dev)
    step_s = dt
    epoch_s = step_s * (CIFAR_TRAIN / args.batch)
    print(f"[train_ddpm] {1 / step_s:.1f} step/s (bs={args.batch}, {peak:.2f} GiB peak) "
          f"-> {epoch_s / 60:.1f} min/epoch, 60 epoch = {epoch_s * 60 / 3600:.1f} h, "
          f"300 epoch = {epoch_s * 300 / 3600:.1f} h", flush=True)

    model.eval()
    xb = torch.randn(args.roundtrip_batch, 3, 32, 32, device=dev)
    with torch.no_grad():
        sched.ddim_invert(model, xb, args.roundtrip_steps)
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        xT = sched.ddim_invert(model, xb, args.roundtrip_steps)
        inv_s = time.time() - t0
        t0 = time.time()
        rec = sched.ddim_sample(model, xT, args.roundtrip_steps)
        smp_s = time.time() - t0
        rt_psnr = psnr(rec, xb)

    per_pass = inv_s / args.roundtrip_batch
    print(f"[inference] 反演 {inv_s:.2f}s / 采样 {smp_s:.2f}s @bs={args.roundtrip_batch}, "
          f"{args.roundtrip_steps} 步 -> {per_pass * 1000:.1f} ms/图/趟", flush=True)
    print(f"[inference] 随机图往返 PSNR={rt_psnr:.1f}dB（仅计时用, 无意义）", flush=True)

    from krd import TrajStego
    from krd.distortions import random_distortion
    import random
    stego = TrajStego(model, sched, n_bits=16, ecc_reps=3, bins_per_bit=2,
                      n_check_bits=32, device=dev)
    rng = random.Random(0)
    xd = torch.randn(args.decoder_batch, 3, 32, 32, device=dev).clamp(-1, 1)
    bits = torch.randint(0, 2, (args.decoder_batch, 16), device=dev).float()
    keys = ["k%d" % i for i in range(args.decoder_batch)]
    with torch.no_grad():
        stego.hide(xd, bits, keys, 50, 1.0)
    if dev == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        sg = stego.hide(xd, bits, keys, 50, 1.0)
        x_in = torch.stack([random_distortion(sg[i:i + 1], rng)[0]
                            for i in range(args.decoder_batch)])
        stego.recover_features(x_in, keys, 50)
    if dev == "cuda":
        torch.cuda.synchronize()
    dec_t = time.time() - t0
    print(f"[train_decoder] {dec_t:.2f}s / step (bs={args.decoder_batch}) "
          f"-> 1000 step = {dec_t * 1000 / 3600:.2f} h", flush=True)

    os.makedirs("results", exist_ok=True)
    out = {"device": dev, "train_step_s": step_s, "epoch_min": epoch_s / 60,
           "60ep_h": epoch_s * 60 / 3600, "300ep_h": epoch_s * 300 / 3600,
           "peak_gib": peak, "decoder_step_s": dec_t,
           "decoder_1000_h": dec_t * 1000 / 3600,
           "infer_ms_per_img_pass": per_pass * 1000, "n_params": n_par}
    with open("results/bench_local.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("saved -> results/bench_local.json")


if __name__ == "__main__":
    main()

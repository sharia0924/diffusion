"""阶段 1：在 CIFAR-10 上预训练基础 DDPM（U-Net + eps 预测）。

支持**断点续训**：每 `--save-every` 个 epoch 把完整训练状态（model/ema/optimizer/epoch）
写入 `<out>.last.pt`；下次启动若发现该文件且已完成 epoch 数不足，则自动恢复继续训练。
这对 300+ epoch 的长时间训练是必需的（本机 GTX 1650 上 300 epoch ≈ 9.5 小时）。

用法:
  python scripts/train_ddpm.py --epochs 60 --batch-size 128
  python scripts/train_ddpm.py --epochs 300 --save-every 10   # 若已有 60 epoch 断点则自动续训
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

CIFAR_TRAIN_SIZE = 50000


def make_grad_scaler(enabled: bool):
    """兼容新旧 torch 的 GradScaler 构造（新 API 无弃用警告）。"""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _mp_ok() -> bool:
    """探测能否使用多进程队列（DataLoader num_workers>0 依赖它）。

    受限沙箱下 multiprocessing.Pipe() 会抛 PermissionError(WinError 5)，
    必须回退到 num_workers=0，否则脚本在 DataLoader 构造阶段直接失败。
    """
    try:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        q.close()
        q.join_thread()
        return True
    except Exception as e:
        print(f"[data] 多进程队列不可用（{type(e).__name__}: {e}）→ num_workers=0",
              flush=True)
        return False


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
    ap.add_argument("--save-every", type=int, default=10,
                    help="每 N 个 epoch 写一次断点续训状态（0 = 只在结束时保存）")
    ap.add_argument("--no-resume", action="store_true",
                    help="忽略已有断点，从第 0 epoch 重新训练")
    ap.add_argument("--log-every", type=int, default=100, help="每 N step 打印一次 loss")
    ap.add_argument("--num-workers", type=int, default=-1,
                    help="DataLoader worker 数；-1 = 自动探测（受限沙箱下自动回退 0）")
    ap.add_argument("--amp", choices=["auto", "on", "off"], default="off",
                    help="混合精度(fp16 autocast)。默认 off：本 U-Net 的 up-path ResBlock "
                         "在 fp16 下会溢出成 inf/nan（即使把 ResBlock 强制 fp32 仍复现），"
                         "因此未启用。--amp on 仅供实验。")
    ap.add_argument("--grad-accum", type=int, default=1,
                    help="梯度累积步数，等效 batch = batch-size × grad-accum")
    ap.add_argument("--channels-last", action="store_true",
                    help="使用 channels_last 内存格式（部分显卡上卷积更快）")
    ap.add_argument("--cudnn-benchmark", action="store_true",
                    help="开启 cudnn.benchmark（输入尺寸固定时更快）")
    ap.add_argument("--oom-retry", action="store_true", default=True,
                    help="显存不足时自动减半 batch 重试（默认开）")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    seed_everything(args.seed)

    if args.num_workers < 0:
        args.num_workers = 2 if _mp_ok() else 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (args.amp == "on") and device == "cuda"
    if args.amp == "on" and device != "cuda":
        print("[setup] 非 CUDA 环境，忽略 --amp on", flush=True)
    if use_amp:
        print("[setup][WARN] 实验性开关：本 U-Net 在 fp16 下有溢出风险（实测 loss→nan），"
              "训练结果不可用，请检查 loss 是否有限。", flush=True)
    if args.cudnn_benchmark and device == "cuda":
        torch.backends.cudnn.benchmark = True

    ds = build_dataset(args.data_root, args.tiny)

    def make_loader(batch_size: int) -> DataLoader:
        return DataLoader(ds, batch_size=batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=(device == "cuda"),
                          drop_last=True, persistent_workers=args.num_workers > 0)

    batch_size = args.batch_size
    if args.tiny:
        args.epochs = min(args.epochs, 2)
        args.base = min(args.base, 32)
        batch_size = min(batch_size, 64)

    model = UNet(base=args.base).to(device)
    # fp16 下 up-path ResBlock 会溢出（实测 inf/nan），默认让 ResBlock 走 fp32
    model.fp32_resblocks = use_amp
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    sched = Schedule(args.timesteps, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    scaler = make_grad_scaler(use_amp)

    print(f"[setup] device={device} amp={use_amp} batch={batch_size} "
          f"grad_accum={args.grad_accum} -> effective_batch={batch_size * args.grad_accum} "
          f"num_workers={args.num_workers} channels_last={args.channels_last}", flush=True)
    if device == "cuda":
        p = torch.cuda.get_device_properties(0)
        print(f"[setup] gpu={p.name} mem={p.total_memory / 2**30:.1f}GiB", flush=True)

    # ---------------- 断点续训 ----------------
    resume_path = args.out.replace(".pt", ".last.pt")
    start_epoch = 0
    if os.path.exists(resume_path) and not args.no_resume:
        st = torch.load(resume_path, map_location=device, weights_only=False)
        done = int(st.get("epoch", 0))
        if done < args.epochs:
            model.load_state_dict(st["model"])
            ema = {k: v.to(device) for k, v in st["ema"].items()}
            if "opt" in st:
                opt.load_state_dict(st["opt"])
            else:
                # 兼容旧格式 checkpoint（无优化器状态）：AdamW 动量从零重新累积，
                # 前几十步相当于热重启，对 300 epoch 级别的续训影响可忽略。
                print("[resume] 断点不含优化器状态，AdamW 动量将重新累积", flush=True)
            start_epoch = done
            print(f"[resume] 从 {resume_path} 恢复: 已完成 {done} epoch, "
                  f"继续训练到 {args.epochs}", flush=True)
        else:
            print(f"[resume] {resume_path} 已完成 {done} >= {args.epochs} epoch, 无需训练",
                  flush=True)

    def save_state(epoch_done: int):
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        torch.save({"model": model.state_dict(),
                    "ema": {k: v.cpu() for k, v in ema.items()},
                    "opt": opt.state_dict(),
                    "epoch": epoch_done,
                    "args": vars(args)}, resume_path)

    step = start_epoch * ((CIFAR_TRAIN_SIZE if not args.tiny else 2048) // batch_size)

    def train_epoch(loader_: DataLoader, epoch: int, running_ref: list) -> None:
        """跑一个 epoch（AMP + 梯度累积 + EMA），把累计 loss 写回 running_ref[0]。"""
        opt.zero_grad(set_to_none=True)
        for i, (x, _) in enumerate(loader_):
            x = x.to(device, non_blocking=True)
            if args.channels_last:
                x = x.to(memory_format=torch.channels_last)
            t = torch.randint(0, sched.T, (x.shape[0],), device=device)
            noise = torch.randn_like(x)
            x_t = sched.add_noise(x, t, noise)
            # autocast 只包住前向：eps 预测在 fp16 下算，MSE 回到 fp32 统计，
            # 否则 fp16 平方和容易溢出成 inf/nan（实测 amp=on 时 loss 直接变 nan）。
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                pred = model(x_t, t)
            loss = torch.nn.functional.mse_loss(pred.float(), noise.float())
            loss_scaled = loss / args.grad_accum
            scaler.scale(loss_scaled).backward()
            if (i + 1) % args.grad_accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                with torch.no_grad():
                    for k, v in model.state_dict().items():
                        ema[k].mul_(args.ema).add_(v.detach(), alpha=1 - args.ema)
            running_ref[0] += loss.item()
        # 处理不满一个累积窗口的尾部梯度
        if len(loader_) % args.grad_accum != 0:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    ema[k].mul_(args.ema).add_(v.detach(), alpha=1 - args.ema)

    for epoch in range(start_epoch, args.epochs):
        running = [0.0]
        try:
            loader = make_loader(batch_size)
            train_epoch(loader, epoch, running)
            step += len(loader)
        except torch.cuda.OutOfMemoryError:
            if not args.oom_retry or batch_size <= 8:
                raise
            torch.cuda.empty_cache()
            new_bs = max(8, batch_size // 2)
            print(f"[oom] batch {batch_size} 显存不足 → 自动降为 {new_bs} 并重跑本 epoch",
                  flush=True)
            batch_size = new_bs
            running = [0.0]
            loader = make_loader(batch_size)
            train_epoch(loader, epoch, running)
            step += len(loader)

        if step and step % args.log_every < len(loader):
            print(f"epoch {epoch} done, step {step}, avg loss (epoch) "
                  f"{running[0] / max(len(loader), 1):.4f}, "
                  f"peak_mem {torch.cuda.max_memory_allocated() / 2**30:.1f}GiB"
                  if device == "cuda" else
                  f"epoch {epoch} done, step {step}, avg loss "
                  f"{running[0] / max(len(loader), 1):.4f}", flush=True)

        if args.save_every and (epoch + 1) % args.save_every == 0:
            save_state(epoch + 1)
            print(f"[ckpt] epoch {epoch + 1}/{args.epochs} 已写入 {resume_path}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "ema": {k: v for k, v in ema.items()},
        "args": vars(args),
        "epochs_done": args.epochs,
    }, args.out)
    save_state(args.epochs)
    print(f"saved -> {args.out} ({args.epochs} epoch; 续训状态 {resume_path})")


if __name__ == "__main__":
    main()

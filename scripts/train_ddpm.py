"""阶段 1：训练基础扩散模型（U-Net + eps 预测）。

支持两种空间：
  - **像素空间**（默认）：直接在 CIFAR-10 32×32 上训练；
  - **隐空间（LDM 迁移）**：`--vae-backend native|sd`，先用 VAE 把图编码成潜变量
    （离线缓存到磁盘），再在潜空间上训练扩散模型。隐空间的频点预算 ∝ 分辨率²，
    是突破像素空间 PSNR 上限的关键。

支持**断点续训**：每 `--save-every` 个 epoch 把完整训练状态（model/ema/optimizer/epoch）
写入 `<out>.last.pt`；下次启动若发现该文件且已完成 epoch 数不足，则自动恢复继续训练。

用法:
  python scripts/train_ddpm.py --epochs 60 --batch-size 128
  python scripts/train_ddpm.py --epochs 300 --save-every 10   # 自动续训
  # LDM: 原生 VAE（先跑 scripts/train_vae.py）
  python scripts/train_ddpm.py --epochs 200 --vae-backend native \
      --vae-ckpt checkpoints/vae_cifar.pt --out checkpoints/ddpm_latent.pt
  # LDM: SD VAE（需要 pip install diffusers）
  python scripts/train_ddpm.py --epochs 200 --vae-backend sd \
      --out checkpoints/ddpm_latent_sd.pt
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
from krd.latent import LatentTensorDataset, build_latent_cache, cache_key, cache_path
from krd.utils import resolve_num_workers, seed_everything

CIFAR_TRAIN_SIZE = 50000


def make_grad_scaler(enabled: bool):
    """兼容新旧 torch 的 GradScaler 构造（新 API 无弃用警告）。"""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def build_dataset(data_root: str, tiny: bool, size: int | None = None):
    tf = [transforms.ToTensor(),
          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    if size is not None:
        tf.insert(0, transforms.Resize(size, antialias=True))
    ds = datasets.CIFAR10(data_root, train=True, download=True,
                          transform=transforms.Compose(tf))
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
    # ---- LDM（隐空间）相关 ----
    ap.add_argument("--vae-backend", choices=["none", "native", "sd"], default="none",
                    help="none = 像素空间训练；native/sd = 在 VAE 隐空间训练（LDM 迁移）")
    ap.add_argument("--vae-ckpt", default="checkpoints/vae_cifar.pt",
                    help="--vae-backend native 时的 VAE checkpoint（train_vae.py 产物）")
    ap.add_argument("--vae-model-id", default="stabilityai/sd-vae-ft-mse",
                    help="--vae-backend sd 时的 HuggingFace 模型 id")
    ap.add_argument("--latent-cache-dir", default="cache/latents",
                    help="潜变量缓存目录（离线编码一次，之后复用）")
    ap.add_argument("--rebuild-latent-cache", action="store_true",
                    help="强制重建潜变量缓存")
    ap.add_argument("--resize", type=int, default=None,
                    help="先把图像缩放到该尺寸再编码（如 256，配合 SD VAE 使用）")
    ap.add_argument("--in-ch", type=int, default=None,
                    help="U-Net 输入通道；默认像素空间 3、隐空间取 VAE 的 latent_channels")
    ap.add_argument("--clip-denoised", choices=["auto", "on", "off"], default="auto",
                    help="是否把预测 x0 截断到 [-1,1]。auto = 像素空间开、隐空间**关**；"
                         "潜变量幅值可达 ±6，截断到 [-1,1] 会毁掉往返"
                         "（实测 latent 往返 33.6dB -> 13.1dB）")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    seed_everything(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 隐空间默认关闭 x0 截断（见 --clip-denoised 的帮助）：潜变量幅值远超 [-1,1]
    if args.clip_denoised == "auto":
        args.clip_denoised = (args.vae_backend == "none")
    else:
        args.clip_denoised = (args.clip_denoised == "on")
    use_amp = (args.amp == "on") and device == "cuda"
    if args.amp == "on" and device != "cuda":
        print("[setup] 非 CUDA 环境，忽略 --amp on", flush=True)
    if use_amp:
        print("[setup][WARN] 实验性开关：本 U-Net 在 fp16 下有溢出风险（实测 loss→nan），"
              "训练结果不可用，请检查 loss 是否有限。", flush=True)
    if args.cudnn_benchmark and device == "cuda":
        torch.backends.cudnn.benchmark = True

    args.num_workers = resolve_num_workers(args.num_workers)
    ds = build_dataset(args.data_root, args.tiny, args.resize)

    # ---------------- 隐空间：编码 + 缓存 ----------------
    vae = None
    latent_meta = {}
    if args.vae_backend == "none":
        args.in_ch = args.in_ch or 3
    else:
        from krd.vae import build_vae
        vae = build_vae(args.vae_backend, ckpt=args.vae_ckpt, device=device,
                        model_id=args.vae_model_id)
        args.in_ch = args.in_ch or vae.latent_channels
        n = len(ds)
        key = cache_key(vae.describe(), f"cifar-train-{n}", args.resize, n)
        path = cache_path(args.latent_cache_dir, key)
        if args.tiny:
            path = path.replace(".pt", "_tiny.pt")
        if os.path.exists(path) and not args.rebuild_latent_cache:
            from krd.latent import load_latent_cache
            latents, labels, latent_meta = load_latent_cache(path)
            print(f"[latent] 复用缓存 {path} {tuple(latents.shape)}", flush=True)
        else:
            print(f"[latent] 编码数据集 -> {path}（{vae.describe()}）", flush=True)
            latent_meta = build_latent_cache(vae, ds, path, batch_size=64, device=device)
            from krd.latent import load_latent_cache
            latents, labels, latent_meta = load_latent_cache(path)
        ds = LatentTensorDataset(latents, labels)
        from krd.latent import latent_capacity_report
        args.latent_shape = [int(v) for v in latents.shape[1:]]
        print(f"[latent] 几何 {tuple(args.latent_shape)}  {latent_meta}", flush=True)

    def make_loader(batch_size: int) -> DataLoader:
        return DataLoader(ds, batch_size=batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=(device == "cuda"),
                          drop_last=True, persistent_workers=args.num_workers > 0)

    batch_size = args.batch_size
    if args.tiny:
        args.epochs = min(args.epochs, 2)
        args.base = min(args.base, 32)
        batch_size = min(batch_size, 64)

    model = UNet(in_ch=args.in_ch, base=args.base).to(device)
    # fp16 下 up-path ResBlock 会溢出（实测 inf/nan），默认让 ResBlock 走 fp32
    model.fp32_resblocks = use_amp
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    sched = Schedule(args.timesteps, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    scaler = make_grad_scaler(use_amp)

    print(f"[setup] device={device} amp={use_amp} batch={batch_size} in_ch={args.in_ch} "
          f"vae={args.vae_backend} grad_accum={args.grad_accum} -> "
          f"effective_batch={batch_size * args.grad_accum} "
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
        # **架构校验**：断点必须与当前参数同构，否则 load_state_dict 会因形状不匹配崩溃
        # （典型场景：先用 --tiny 试跑过，之后用正式参数训练时残留 .last.pt 直接把训练打挂）。
        pa = st.get("args", {})
        mism = []
        for k in ("base", "in_ch", "vae_backend", "vae_ckpt", "latent_shape", "resize"):
            old, cur = pa.get(k), getattr(args, k, None)
            if old is not None and str(old) != str(cur):
                mism.append(f"{k}: 断点={old} 当前={cur}")
        if mism:
            print(f"[resume][跳过] {resume_path} 的配置与当前参数不一致，"
                  f"将从第 0 epoch 重新训练：\n    " + "\n    ".join(mism), flush=True)
        elif done < args.epochs:
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

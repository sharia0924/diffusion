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


def build_unet_from_args(margs: dict, device: str, check_compat: bool = False) -> UNet:
    """按 checkpoint 中记录的配置重建 U-Net（支持隐空间 in_ch）。"""
    unet = UNet(in_ch=margs.get("in_ch", 3), base=margs.get("base", 64)).to(device)
    if check_compat:
        lv = margs.get("latent_shape")
        if lv is not None:
            print(f"  [model] 隐空间模型: latent_shape={lv} in_ch={margs.get('in_ch')}")
    return unet


def auto_bins_per_bit(res: int, n_slots: int, requested: int = 2) -> int:
    """按空间的实际频点预算决定每槽占用多少对频点。

    容量 ∝ 分辨率²：像素 32² 单通道有 342 对，2 对/槽可放 160 对；
    而 16×16 的 latent 单通道只有 70 对，若仍用 2 对/槽 需要 160 对 → 超出预算。
    这里从 requested 向下找到第一个放得下的值（至少 1）。
    """
    from krd.pattern import ring_capacity
    avail = ring_capacity(res)
    for bpb in range(requested, 0, -1):
        if n_slots * bpb <= avail:
            if bpb != requested:
                print(f"  [capacity] res={res} 单通道可用 {avail} 对，"
                      f"bins_per_bit {requested} -> {bpb}（槽位 {n_slots}）")
            return bpb
    raise ValueError(f"res={res} 连每槽 1 对都放不下 {n_slots} 个槽位（可用 {avail} 对）")


def fit_capacity(res: int, n_bits: int, ecc: int, n_check: int,
                 requested_bpb: int = 2) -> tuple[int, int, int, int]:
    """把 (n_bits, ecc, n_check, bpb) 调整到该分辨率的频点预算之内。

    优先保持 n_bits 与 ecc，先降 bpb；仍放不下则缩减 n_check，最后才降 n_bits。
    返回调整后的四元组，并在发生调整时打印说明。
    （隐空间 16×16 的预算只有 70 对，而默认配置需要 160 对，
      不做自适应会让评测脚本在没有解码器时直接崩掉。）
    """
    from krd.pattern import ring_capacity
    avail = ring_capacity(res)
    bpb, check = requested_bpb, n_check
    bpb = min(bpb, max(1, avail // max(1, n_bits * ecc + check)))
    while n_bits * ecc + check > avail // max(bpb, 1) and check > 0:
        check -= 1
    while n_bits > 1 and n_bits * ecc + check > avail // max(bpb, 1):
        n_bits -= 1
    slots = n_bits * ecc + check
    if (n_bits, check, bpb) != (n_bits, n_check, requested_bpb):
        print(f"  [capacity] res={res} 可用 {avail} 对 -> "
              f"自适应为 n_bits={n_bits} ecc={ecc} check={check} bpb={bpb} "
              f"(slots={slots}, pairs={slots * bpb})")
    return n_bits, ecc, check, bpb


def load_stego(ckpt_path: str, device: str, n_bits: int = 16, ecc_reps: int = 3,
               bins_per_bit: int = 2, n_check_bits: int = 32,
               with_vae: bool = False, res: int | None = None,
               inject_mode: str | None = None) -> TrajStego:
    """加载 DDPM（可选 VAE）并构造 TrajStego。

    with_vae=True 且 checkpoint 是隐空间模型时，会一并加载 VAE 并挂到
    `stego.vae` 上，供"潜变量 -> 图像"的变换（train_decoder.py 的输入必须是
    像素空间，因为 JPEG/噪声等失真是定义在像素上的）。
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    margs = ckpt.get("args", {})
    unet = build_unet_from_args(margs, device)
    unet.load_state_dict(ckpt.get("ema", ckpt["model"]))
    # clip_denoised 必须与训练时一致：隐空间为 False（潜变量幅值远超 [-1,1]，
    # 截断会把往返 PSNR 从 ~33.6dB 打到 ~13.1dB —— 见 krd/schedule.py 注释）
    _clip = margs.get("clip_denoised")
    if _clip is None:
        _clip = (margs.get("vae_backend", "none") == "none")
    sched = Schedule(margs.get("timesteps", 1000), device=device, clip_denoised=bool(_clip))
    # res（图案的频率网格尺寸）必须与模型实际空间一致，否则 bins 越界：
    # 优先级 = 显式参数 > checkpoint 的 latent_res > latent_shape 的 H > 32（像素空间）
    lshape = margs.get("latent_shape")
    _res = res or margs.get("latent_res") or (lshape[1] if lshape else 32)
    # 频点预算不足时自适应（像素空间 res=32 预算充足，参数不变）
    n_bits, ecc_reps, n_check_bits, bins_per_bit = fit_capacity(
        int(_res), n_bits, ecc_reps, n_check_bits, bins_per_bit)
    stego = TrajStego(unet, sched, n_bits=n_bits, ecc_reps=ecc_reps,
                      bins_per_bit=bins_per_bit, n_check_bits=n_check_bits,
                      res=int(_res), inject_mode=inject_mode or "replace",
                      device=device)
    vae = None
    if with_vae:
        backend = margs.get("vae_backend", "none")
        if backend != "none":
            from krd.vae import build_vae
            vae = build_vae(backend, ckpt=margs.get("vae_ckpt"), device=device,
                            model_id=margs.get("vae_model_id", "stabilityai/sd-vae-ft-mse"))
            print(f"  [vae] {vae.describe()}")
    stego.vae = vae
    stego.latent_shape = margs.get("latent_shape")
    stego.pixel_res = margs.get("resize") or 32
    return stego


def stego_to_images(stego, z: torch.Tensor) -> torch.Tensor:
    """潜变量 -> 像素图（[-1,1]）；像素空间模型则原样返回。"""
    vae = getattr(stego, "vae", None)
    if vae is None:
        return z
    px = getattr(stego, "pixel_res", 32)
    return vae.decode(z, target_hw=(px, px))


def images_to_stego_space(stego, x: torch.Tensor) -> torch.Tensor:
    """像素图 -> 模型所在空间（隐空间则编码）；像素空间模型则原样返回。"""
    vae = getattr(stego, "vae", None)
    if vae is None:
        return x
    return vae.encode(x, sample=False)


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
    ap.add_argument("--n-bits", type=int, default=16,
                    help="消息比特数（容量）；隐空间 16×16 频点预算有限时需下调")
    ap.add_argument("--ecc-reps", type=int, default=3,
                    help="重复码次数；槽位数 = n_bits×reps + n_check_bits")
    ap.add_argument("--bins-per-bit", type=int, default=2,
                    help="每槽占用的频点对数；超预算时会自动下调并打印提示")
    ap.add_argument("--inject-mode", choices=["replace", "add"], default="add",
                    help="频谱注入方式：add = 加性（保留原系数，质量随 strength 平滑变化，"
                         "隐空间必备）；replace = 历史行为（覆盖系数，实测一加注入"
                         "PSNR 就从 35dB 掉到 15dB）")
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
    # with_vae=True 仅对隐空间 checkpoint 生效（像素空间 checkpoint 的 vae_backend=none）
    stego = load_stego(args.ddpm_ckpt, device, n_bits=args.n_bits, ecc_reps=args.ecc_reps,
                       bins_per_bit=args.bins_per_bit, n_check_bits=args.n_check_bits,
                       with_vae=True, inject_mode=args.inject_mode)
    if getattr(stego, "vae", None) is not None:
        print(f"[latent] 隐空间解码器训练: latent_shape={stego.latent_shape} "
              f"容量={stego.n_bits}bit ecc={stego.ecc} bpb={stego.bpb} "
              f"slots={stego.total_embed_bits} n_pairs={stego.n_pairs}"
              f"（失真在像素空间施加，再编码回隐空间）", flush=True)
    decoder = RingDecoder(2 * stego.n_pairs, stego.total_embed_bits).to(device)
    opt = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=1e-4)

    # **像素尺寸必须与 VAE 训练时一致**：否则 VAE 编码出的 latent 分辨率会偏小
    # （例：VAE 按 64² 训练 -> 32×32 latent；若这里仍用 32² CIFAR，会得到 16×16 latent，
    #  而图案频率网格按 res=32 生成 -> gather 越界 / 指标无效）。
    _vae_resize = None
    if getattr(stego, "vae", None) is not None:
        try:
            _vck = torch.load(args.ddpm_ckpt, map_location="cpu",
                              weights_only=False).get("args", {}).get("vae_ckpt")
            _vae_resize = torch.load(_vck, map_location="cpu",
                                     weights_only=False).get("args", {}).get("resize")
        except Exception:
            _vae_resize = None
        if _vae_resize:
            print(f"[latent] 像素输入将缩放到 {_vae_resize}×{_vae_resize}"
                  f"（与 VAE 训练一致）", flush=True)

    tf_list = []
    if _vae_resize:
        tf_list.append(transforms.Resize(int(_vae_resize), antialias=True))
    tf_list += [transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    tf = transforms.Compose(tf_list)
    ds = datasets.CIFAR10(args.data_root, train=True, download=True, transform=tf)
    eval_ds = datasets.CIFAR10(args.data_root, train=False, download=True, transform=tf)
    if args.tiny:
        ds = Subset(ds, range(64))
    num_workers = resolve_num_workers(args.num_workers)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=num_workers, pin_memory=(device == "cuda"),
                        persistent_workers=num_workers > 0)
    it = iter(loader)

    # 固定评测批（nonce 协议与部署一致：nonce = H(key || counter)，不依赖 cover）
    eval_x = torch.stack([eval_ds[i][0] for i in range(args.eval_size)]).to(device)
    eval_keys = [token_key(rng) for _ in range(args.eval_size)]
    eval_nonces = derive_nonces_from_keys(eval_keys, start=0)

    # ---------------- 空间适配（像素空间直通；隐空间做 编码/解码 桥接） ----------------
    #
    # 关键点：JPEG/噪声等失真定义在**像素**上，而扩散在前向/隐藏在**潜变量**上。
    # 因此隐空间模型的信道是：latent -> decode 到像素 -> 加失真 -> encode 回 latent。
    # 这个"解码-再编码"本身就是 VAE 引入的额外信道损耗，必须在训练时就模拟。
    def hide_space(x_pix, bits, keys, hide_steps, strength, nonces):
        z = images_to_stego_space(stego, x_pix)
        return stego.hide(z, bits, keys, hide_steps, strength=strength, nonces=nonces)

    def to_space(x_img):
        return images_to_stego_space(stego, x_img)

    def to_images(z_or_img):
        return stego_to_images(stego, z_or_img)

    def distort(z, fn):
        """在像素空间施加失真，再回到模型空间。"""
        return to_space(fn(to_images(z)))

    def recover_logits(z, keys, rs, dec, nonces):
        return stego.recover(z, keys, rs, dec, nonces=nonces)

    def evaluate() -> dict:
        decoder.eval()
        n = args.eval_size
        bits = torch.randint(0, 2, (n, stego.n_bits), device=device).float()
        keys = eval_keys
        sg = hide_space(eval_x, bits, keys, args.hide_steps, 1.0, eval_nonces)
        out = {}
        for name, atk in [("clean", "clean"), ("jpeg50", ("jpeg", 50)),
                          ("noise05", ("noise", 0.05))]:
            x_in = sg if atk == "clean" else distort(sg, lambda t, a=atk: apply_attack(t, a[0], a[1]))
            rs = args.rec_steps
            if args.rec_jitter > 0:
                rs += rng.randint(-args.rec_jitter, args.rec_jitter)
            logits = recover_logits(x_in, keys, rs, decoder, eval_nonces)
            out[name] = bit_accuracy(logits, bits)
        decoder.train()
        return out

    best = -1.0
    config = {"n_bits": stego.n_bits, "ecc": stego.ecc, "bpb": stego.bpb,
              "res": stego.res, "n_check_bits": stego.n_check_bits,
              "n_pairs": stego.n_pairs, "hide_steps": args.hide_steps,
              "rec_steps": args.rec_steps, "nonce_protocol": "keyed-v2",
              "geom_prob": args.geom_prob, "inject_mode": args.inject_mode}
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

        if step == 1 and os.environ.get("KRD_DEBUG_STEP"):
            # 崩溃定位用：打印注入前的真实形状与索引范围
            from krd.pattern import _lin_index
            z_dbg = images_to_stego_space(stego, x0)
            xT_dbg = stego.sched.ddim_invert(stego.model, z_dbg, args.hide_steps)
            p_dbg = stego.params_for(keys[0], nonces[0])
            b_dbg = p_dbg["bins"]
            Cd, Hd, Wd = xT_dbg[0].shape
            idx_dbg = _lin_index(b_dbg, Wd)
            mir_dbg = torch.stack([(-b_dbg[:, 0]) % Hd, (-b_dbg[:, 1]) % Wd], dim=1)
            print(f"[DEBUG] x0={tuple(x0.shape)} z={tuple(z_dbg.shape)} "
                  f"xT={tuple(xT_dbg.shape)} B={B}", flush=True)
            print(f"[DEBUG] res={stego.res} bpb={stego.bpb} n_pairs={stego.n_pairs} "
                  f"mode={stego.inject_mode} strength={strength[:3].tolist()}", flush=True)
            print(f"[DEBUG] bins row[{int(b_dbg[:,0].min())},{int(b_dbg[:,0].max())}] "
                  f"col[{int(b_dbg[:,1].min())},{int(b_dbg[:,1].max())}]", flush=True)
            print(f"[DEBUG] idx.max={int(idx_dbg.max())} mirror.max={int(_lin_index(mir_dbg, Wd).max())} "
                  f"limit={Cd * Hd * Wd}", flush=True)

        sg = hide_space(x0, bits, keys, args.hide_steps, strength, nonces)
        x_in = torch.stack([
            distort(sg[i:i + 1], lambda t: random_distortion(
                t, rng, schedule=stego.sched,
                sched_noise_prob=args.sched_noise_prob,
                geom_prob=args.geom_prob))[0]
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

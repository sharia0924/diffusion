"""Diagnostic probe: separate DDIM round-trip error from pattern-injection error.

Two things are measured:

  1. The DDIM round-trip PSNR (no embedding at all) -- the ceiling of the pipeline.
  2. The injection variant comparison at EQUAL perturbation energy:
       replace : current code, F <- template*strength*A0   (flattens magnitudes)
       delta   : F <- F + d*s_j*e^{j phi_j}                (keeps magnitude structure)
       ratio   : F <- F*(1 + strength*s_j)                 (multiplicative)

It also demonstrates the bin-indexing bug in the current krd/pattern.py:
`F[:, bins[:,0], bins[:,1]]` on a (C,H,W) tensor indexes (channels, rows, cols)
only when the first advanced index is broadcast against the channel axis --
PyTorch treats a full 3D advanced index as selecting along ALL THREE dims, so the
row indices are applied to the channel axis.  The library's own call therefore
either raises IndexError (CPU) / device-side assert (CUDA) or silently reads the
wrong bins.  This probe uses the unambiguous flat gather instead:
    F.reshape(C, H*W).gather(1, row*W + col)

Usage:
  python scripts/diag_psnr_probe.py --n 8 --device cpu --out results/diag_psnr_probe.json
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from krd import Schedule, UNet
from krd.distortions import apply_attack
from krd.metrics import bit_accuracy, psnr, ssim
from krd.pattern import ecc_collapse, ecc_encode, key_params
from krd.utils import seed_everything, token_key


def load_unet(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    margs = ckpt.get("args", {})
    unet = UNet(base=margs.get("base", 64)).to(device)
    unet.load_state_dict(ckpt.get("ema", ckpt["model"]))
    unet.eval()
    for p in unet.parameters():
        p.requires_grad_(False)
    return unet, Schedule(margs.get("timesteps", 1000), device=device)


# ---- unambiguous, resolution-agnostic spectral access (works CPU + CUDA) ----

def lin_index(bins, W, device):
    """(n_pairs,) linear index into the flattened (C, H*W) spectrum."""
    return (bins[:, 0] * W + bins[:, 1]).to(dtype=torch.long, device=device)


def spec_gather(x_T, bins, W):
    """x_T (C,H,W) -> spectrum values at bins, shape (C, n_pairs), complex."""
    F = torch.fft.fftshift(torch.fft.fft2(x_T, norm="ortho"), dim=(-2, -1))
    C = x_T.shape[0]
    idx = lin_index(bins, W, x_T.device).view(1, -1).expand(C, -1)
    return F.reshape(C, -1).gather(1, idx)


def inject_replace(x_T, bits, params, strength):
    """Current code semantics: flat-magnitude replacement."""
    L = int(bits.numel())
    bins, phases = params["bins"], params["phases"]
    g = bins.shape[0] // L
    sign = torch.where(bits > 0.5, 1.0, -1.0).repeat_interleave(g)
    C, H, W = x_T.shape
    F = torch.fft.fftshift(torch.fft.fft2(x_T, norm="ortho"), dim=(-2, -1))
    A0 = spec_gather(x_T, bins, W).abs().median()
    tmpl = params["base_mag"] * torch.exp(1j * phases) * sign
    F = F.reshape(C, -1).clone()
    idx = lin_index(bins, W, x_T.device)
    F[:, idx] = tmpl * strength * A0
    F[:, (bins[:, 0] * (-1)) % H * W + (bins[:, 1] * (-1)) % W] = \
        torch.conj(tmpl) * strength * A0
    F = F.view(C, H, W)
    return torch.fft.ifft2(torch.fft.ifftshift(F, dim=(-2, -1)), norm="ortho").real


def inject_delta(x_T, bits, params, strength):
    """Magnitude-structure preserving additive injection, energy-matched to replace."""
    L = int(bits.numel())
    bins, phases = params["bins"], params["phases"]
    g = bins.shape[0] // L
    sign = torch.where(bits > 0.5, 1.0, -1.0).repeat_interleave(g)
    C, H, W = x_T.shape
    A0 = spec_gather(x_T, bins, W).abs().median()
    d = strength * A0 * (2.0 / np.sqrt(3.0))
    F = torch.fft.fftshift(torch.fft.fft2(x_T, norm="ortho"), dim=(-2, -1))
    F = F.reshape(C, -1).clone()
    idx = lin_index(bins, W, x_T.device)
    midx = ((bins[:, 0] * (-1)) % H) * W + (bins[:, 1] * (-1)) % W
    delta = d * sign * torch.exp(1j * phases)
    F[:, idx] = F[:, idx] + delta
    F[:, midx] = F[:, midx] + torch.conj(delta)
    F = F.view(C, H, W)
    return torch.fft.ifft2(torch.fft.ifftshift(F, dim=(-2, -1)), norm="ortho").real


def inject_ratio(x_T, bits, params, strength):
    """Multiplicative magnitude modulation."""
    L = int(bits.numel())
    bins, phases = params["bins"], params["phases"]
    g = bins.shape[0] // L
    sign = torch.where(bits > 0.5, 1.0, -1.0).repeat_interleave(g)
    C, H, W = x_T.shape
    F = torch.fft.fftshift(torch.fft.fft2(x_T, norm="ortho"), dim=(-2, -1))
    F = F.reshape(C, -1).clone()
    idx = lin_index(bins, W, x_T.device)
    midx = ((bins[:, 0] * (-1)) % H) * W + (bins[:, 1] * (-1)) % W
    F[:, idx] = F[:, idx] * (1.0 + strength * sign)
    F[:, midx] = F[:, midx] * (1.0 + strength * sign)
    F = F.view(C, H, W)
    return torch.fft.ifft2(torch.fft.ifftshift(F, dim=(-2, -1)), norm="ortho").real


def cyclic_shift(x_T, shift: int = 2):
    """旧版 `crop` 的真实语义：torch.roll 循环平移（频谱幅度不变，只旋转相位）。"""
    return torch.roll(x_T, shifts=(shift, shift), dims=(-2, -1))


VARIANTS = {"replace": inject_replace, "delta": inject_delta, "ratio": inject_ratio}


def mf_logits(x_T, params, n_msg_bits, ecc, n_check):
    """Matched filter: de-rotate by key phase, average in-phase component per slot."""
    bins, phases = params["bins"], params["phases"]
    C, H, W = x_T.shape
    vals = spec_gather(x_T, bins, W) * torch.exp(-1j * phases)[None, :]
    norm = vals.abs().median().clamp(min=1e-8)
    re = vals.real.mean(0) / norm
    total = n_msg_bits * ecc + n_check
    g = bins.shape[0] // total
    return re.view(total, g).mean(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddpm-ckpt", default="checkpoints/ddpm_cifar.pt")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--hide-steps", type=int, default=50)
    ap.add_argument("--rec-steps", type=int, default=50)
    ap.add_argument("--n-bits", type=int, default=16)
    ap.add_argument("--ecc", type=int, default=3)
    ap.add_argument("--bpb", type=int, default=2)
    ap.add_argument("--n-check", type=int, default=32)
    ap.add_argument("--strengths", default="0.2,0.5,1.0")
    ap.add_argument("--out", default="results/diag_psnr_probe.json")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    seed_everything(args.seed)

    device = args.device
    unet, sched = load_unet(args.ddpm_ckpt, device)

    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
    ds = datasets.CIFAR10(args.data_root, train=False, download=True, transform=tf)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False)
    covers = torch.cat([x for x, _ in loader])[:args.n].to(device)
    n = covers.shape[0]

    total_bits = args.n_bits * args.ecc + args.n_check
    n_pairs = total_bits * args.bpb
    key = token_key()
    params = key_params(key, n_pairs, res=32, nonce="")
    params = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in params.items()}
    bits = torch.randint(0, 2, (n, args.n_bits), device=device).float()
    bits_full = ecc_encode(bits, args.ecc)
    check = torch.randint(0, 2, (args.n_check,), device=device).float()
    full = torch.cat([bits_full, check.expand(n, -1)], dim=1)

    report = {"n": n, "device": device, "hide_steps": args.hide_steps,
              "rec_steps": args.rec_steps, "n_pairs": n_pairs, "total_bits": total_bits}

    x_T_all = sched.ddim_invert(unet, covers, args.hide_steps)
    rt = sched.ddim_sample(unet, x_T_all, args.hide_steps)
    report["ddim_roundtrip_psnr"] = psnr(rt, covers)
    report["ddim_roundtrip_ssim"] = ssim(rt, covers)
    print(f"[Q1] DDIM roundtrip (S={args.hide_steps}) PSNR "
          f"{report['ddim_roundtrip_psnr']:.2f} dB SSIM "
          f"{report['ddim_roundtrip_ssim']:.4f}", flush=True)

    bins = params["bins"]
    A0 = spec_gather(x_T_all[0], bins, 32).abs()
    report["xT_total_energy"] = float((x_T_all ** 2).sum(1).mean())
    report["A0_median"] = float(A0.median())
    report["A0_over_rms"] = float(A0.median() / x_T_all.pow(2).mean().sqrt())
    report["ring_energy_share"] = float(
        (2 * A0 ** 2).sum() / (x_T_all[0] ** 2).sum())
    print(f"[Q2] A0(median)={report['A0_median']:.4f} A0/rms={report['A0_over_rms']:.3f} "
          f"ring-energy-share={report['ring_energy_share']:.4f}", flush=True)

    rows = []
    for sname, fn in VARIANTS.items():
        for st in [float(v) for v in args.strengths.split(",")]:
            xe = torch.stack([fn(x_T_all[i], full[i], params, st) for i in range(n)])
            d_energy = float(((xe - x_T_all) ** 2).sum(1).mean() / report["xT_total_energy"])
            stego = sched.ddim_sample(unet, xe, args.hide_steps)
            row = {"variant": sname, "strength": st, "rel_perturb_energy": d_energy,
                   "psnr": psnr(stego, covers), "ssim": ssim(stego, covers)}
            for atk_name, atk in [("clean", None), ("jpeg50", ("jpeg", 50)),
                                  ("cyclic2", "cyclic2"), ("translate2", ("translate", 2)),
                                  ("crop2", ("crop", 2))]:
                if atk == "cyclic2":
                    xin = cyclic_shift(stego, 2)
                else:
                    xin = stego if atk is None else apply_attack(stego, atk[0], atk[1])
                xr = sched.ddim_invert(unet, xin, args.rec_steps)
                accs = []
                for i in range(n):
                    lg = mf_logits(xr[i], params, args.n_bits, args.ecc, args.n_check)
                    lg = ecc_collapse(lg[:args.n_bits * args.ecc], args.n_bits, args.ecc)
                    accs.append(bit_accuracy(lg[None], bits[i][None]))
                row[f"mf_{atk_name}"] = float(np.mean(accs))
            rows.append(row)
            print(f"[Q3] {sname:8s} s={st:<4} pertE={d_energy:.4f} PSNR {row['psnr']:5.2f} "
                  f"SSIM {row['ssim']:.3f} | mf clean {row['mf_clean']:.3f} "
                  f"jpeg50 {row['mf_jpeg50']:.3f} cyclic2 {row['mf_cyclic2']:.3f} "
                  f"translate2 {row['mf_translate2']:.3f} crop2 {row['mf_crop2']:.3f}",
                  flush=True)

    report["rows"] = rows
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()

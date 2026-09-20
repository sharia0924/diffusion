import sys
sys.path.insert(0, ".")
import torch
from krd.metrics import bit_accuracy, psnr
from krd.pattern import ecc_collapse
from krd.utils import derive_nonces_from_keys, seed_everything
from scripts.eval_common import StegoIO, cifar_loader, gather_covers
from scripts.train_decoder import load_stego
seed_everything(0)
dev = "cuda"
s = load_stego("checkpoints/ddpm_latent32.pt", dev, with_vae=True, inject_mode="add")
io = StegoIO(s, pixel_res=64)
covers = gather_covers(cifar_loader("./data", train=False, batch_size=4, resize=64), 4).to(dev)
keys = ["k1","k2","k3","k4"]; nonces = derive_nonces_from_keys(keys, start=0)
bits = torch.randint(0,2,(4, s.n_bits), device=dev).float()
z = io.to_space(covers)
def mf(zx):
    o=[]
    for i in range(4):
        f = s.recover_features(zx[i:i+1],[keys[i]],150,nonces=[nonces[i]])[0]
        g = s.n_pairs // s.total_embed_bits
        re = f[:s.n_pairs].view(s.total_embed_bits,g).mean(-1)
        lg = ecc_collapse(re[None,:s.msg_embed_bits], s.n_bits, s.ecc)
        o.append(bit_accuracy(lg, bits[i][None]))
    return sum(o)/len(o)
print(f"  {'总能量E':>8s} {'n':>3s} {'单点s':>8s} {'PSNR':>9s} {'mf':>7s}")
for E in (0.1, 0.2, 0.3):
    for n in (1, 2, 4, 8):
        st = E / n
        sg = s.hide(z, bits, keys, 150, st, nonces=nonces, inject_at=0.35, n_inject=n)
        print(f"  {E:>8.2f} {n:>3d} {st:>8.4f} {psnr(io.to_pixels(sg), covers):>8.2f}dB {mf(sg):>7.3f}", flush=True)

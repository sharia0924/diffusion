"""逐层打印 NativeVAE 编解码形状，定位形状错配。"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from krd.vae import NativeVAE


def main():
    m = NativeVAE(in_ch=3, base=64, z_ch=4, downsample=3, ch_mults=(1, 2, 2, 2))
    print("enc_levels:", [(l[0].norm1.num_groups, l[1].out_channels, l[2].out_channels)
                          for l in m.enc_levels])
    print("dec_levels:", [(l[0].out_channels, l[1].norm1.num_groups, l[2].out_channels)
                          for l in m.dec_levels])
    print("dec_out final conv in_ch:", m.dec_out[-1].in_channels,
          "out_ch:", m.dec_out[-1].out_channels)
    x = torch.randn(1, 3, 32, 32)
    h = m.enc_in(x)
    print("enc_in     ", tuple(h.shape))
    for i, (res, conv, down) in enumerate(m.enc_levels):
        h = conv(res(h))
        print(f"  enc{i} conv", tuple(h.shape))
        h = down(torch.nn.functional.silu(h))
        print(f"  enc{i} down", tuple(h.shape))
    h = m.enc_mid(h)
    print("enc_mid    ", tuple(h.shape))
    z = m.to_mu(h)
    print("mu (z)     ", tuple(z.shape))

    h = m.dec_mid(m.from_z(z))
    print("dec_mid    ", tuple(h.shape))
    levels = len(m.dec_levels)
    for i, (up_conv, res, conv) in enumerate(m.dec_levels):
        tgt = (32 // (2 ** (levels - 1 - i)), 32 // (2 ** (levels - 1 - i)))
        h = torch.nn.functional.interpolate(up_conv(h), size=tgt, mode="nearest")
        print(f"  dec{i} up  ", tuple(h.shape), "target", tgt)
        h = conv(res(h))
        print(f"  dec{i} conv", tuple(h.shape))
    out = m.dec_out(h)
    print("output     ", tuple(out.shape))


if __name__ == "__main__":
    main()

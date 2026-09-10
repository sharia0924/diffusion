"""小型 DDPM U-Net（CIFAR 32x32 量级），时间条件 + 自注意力。

结构：base=64, ch_mults=(1,2,2,2)（分辨率 32/16/8/4），8x8 与 4x4 处自注意力。
可通过 base/ch_mults/attn_levels 缩放（--tiny 冒烟测试用 base=16）。

混合精度注意：本实现在 fp16 autocast 下**会溢出**（实测 up path 的 ResBlock
出现 inf/nan）：up 路径 concat 后激活幅值变大，fp16 卷积 + GroupNorm 的
平方和累加超出 fp16 范围。因此 ResBlock 内部的前向可要求 fp32
（`ResBlock.forward_fp32`），由 UNet.forward(fp32_resblocks=True) 控制；
混合精度开关见 train_ddpm.py 的 --amp。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float()[:, None] * freqs[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class TimeMLP(nn.Module):
    def __init__(self, dim_in: int = 256, dim_out: int = 512):
        super().__init__()
        self.dim_in = dim_in
        self.net = nn.Sequential(
            nn.Linear(dim_in, dim_out), nn.SiLU(), nn.Linear(dim_out, dim_out)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(timestep_embedding(t, self.dim_in))


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(temb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)

    def forward_fp32(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        """强制 fp32 计算（autocast 下用于避免 fp16 溢出）。

        注意：autocast 只自动转激活，**不转权重**——在 autocast 上下文里调用
        `self.conv1(...)` 时权重已被缓存成 fp16。因此这里必须显式把参数
        cast 成 fp32 再算，否则 fp16 权重仍会溢出。
        """
        with torch.autocast(device_type=x.device.type, enabled=False):
            w1 = self.conv1.weight.float()
            w2 = self.conv2.weight.float()
            wt = self.time_proj.weight.float()
            b1 = None if self.conv1.bias is None else self.conv1.bias.float()
            b2 = None if self.conv2.bias is None else self.conv2.bias.float()
            bt = None if self.time_proj.bias is None else self.time_proj.bias.float()
            h = F.conv2d(F.silu(self.norm1(x.float())), w1, b1, padding=1)
            h = h + F.linear(F.silu(temb.float()), wt, bt)[:, :, None, None]
            h = F.conv2d(F.silu(self.norm2(h)), w2, b2, padding=1)
            if isinstance(self.skip, nn.Identity):
                skip = x.float()
            else:
                skip = F.conv2d(x.float(), self.skip.weight.float(),
                                None if self.skip.bias is None else self.skip.bias.float())
            return h + skip


class SelfAttention(nn.Module):
    def __init__(self, ch: int, heads: int = 4):
        super().__init__()
        assert ch % heads == 0
        self.heads = heads
        self.norm = nn.GroupNorm(8, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, h * w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]          # (b, heads, d, hw)
        attn = torch.softmax(q.transpose(-1, -2) @ k / math.sqrt(q.shape[1]), dim=-1)
        out = (v @ attn.transpose(-1, -2)).reshape(b, c, h, w)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class UNet(nn.Module):
    def __init__(self, in_ch: int = 3, base: int = 64, ch_mults=(1, 2, 2, 2),
                 attn_levels=(2, 3), time_dim: int = 512):
        super().__init__()
        chans = [base * m for m in ch_mults]
        self.time = TimeMLP(256, time_dim)
        self.init_conv = nn.Conv2d(in_ch, base, 3, padding=1)

        # -------- down path --------
        self.down_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        prev = base
        for level, ch in enumerate(chans):
            self.down_blocks.append(nn.ModuleList([
                ResBlock(prev, ch, time_dim),
                SelfAttention(ch) if level in attn_levels else nn.Identity(),
            ]))
            prev = ch
            if level < len(chans) - 1:
                self.downs.append(Downsample(ch))

        # -------- middle --------
        mid_ch = chans[-1]
        self.mid = nn.ModuleList([
            ResBlock(mid_ch, mid_ch, time_dim),
            SelfAttention(mid_ch),
            ResBlock(mid_ch, mid_ch, time_dim),
        ])

        # -------- up path --------
        self.up_blocks = nn.ModuleList()
        self.ups = nn.ModuleList()
        for level in reversed(range(len(chans))):
            ch = chans[level]
            skip_ch = chans[level]  # down path 在该 level 输出的通道数
            self.up_blocks.append(nn.ModuleList([
                ResBlock(prev + skip_ch, ch, time_dim),
                SelfAttention(ch) if level in attn_levels else nn.Identity(),
            ]))
            prev = ch
            if level > 0:
                self.ups.append(Upsample(ch))

        self.out_norm = nn.GroupNorm(8, prev)
        self.out_conv = nn.Conv2d(prev, in_ch, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                fp32_resblocks: bool | None = None) -> torch.Tensor:
        """fp32_resblocks 控制 ResBlock 是否强制 fp32（autocast 下防 fp16 溢出）。

        为 None 时使用实例属性 `self.fp32_resblocks`（默认 False）。
        任何调用方（schedule / stego / eval）都无需显式传参即可生效。
        """
        if fp32_resblocks is None:
            fp32_resblocks = getattr(self, "fp32_resblocks", False)

        def rb(blk, h, temb):
            return blk.forward_fp32(h, temb) if fp32_resblocks else blk(h, temb)

        temb = self.time(t)
        h = self.init_conv(x)
        h_last = h  # 保留 init_conv 输出作为 level0 的额外 skip（近似对称）

        skips = []
        for i, (blk, attn) in enumerate(self.down_blocks):
            h = rb(blk, h, temb)
            h = attn(h)
            skips.append(h)
            if i < len(self.downs):
                h = self.downs[i](h)

        for m in self.mid:
            h = rb(m, h, temb) if isinstance(m, ResBlock) else m(h)

        for i, (blk, attn) in enumerate(self.up_blocks):
            h = torch.cat([h, skips.pop()], dim=1)
            h = rb(blk, h, temb)
            h = attn(h)
            if i < len(self.ups):
                h = self.ups[i](h)

        # 最终 skip 与 init_conv 对齐（分辨率/通道一致时融合，增强细节重建）
        if h.shape == h_last.shape:
            h = h + h_last
        return self.out_conv(F.silu(self.out_norm(h)))

# LDM 迁移端到端验证

- 空间: 空间=隐空间 (4, 16, 16)，VAE=VAE[native] ch=4 down=2x (2^1) scale=1.00000 src=checkpoints/vae16.last.pt
- 样本数: 4, 容量 16 bits, S_hide=50, S_rec=50
- **VAE 往返上限（无嵌入）: PSNR 38.31 dB**
- 容量报告: `{'z_shape': (4, 16, 16), 'avail_bins_single_channel': 70, 'avail_pairs_all_channels': 280, 'r_min': 2, 'r_max': 7, 'slots': 16, 'obs_per_bit_ceiling': 17.5}`

| strength | PSNR | SSIM | LPIPS | mf clean | dec clean | dec jpeg50 |
|---|---|---|---|---|---|---|
| 1.0 | 9.61 | 0.1665 | n/a | 1.000 | nan | nan |

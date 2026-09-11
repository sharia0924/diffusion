# LDM 迁移端到端验证

- 空间: 空间=隐空间 (4, 8, 8)，VAE=VAE[native] ch=4 down=4x (2^2) scale=1.00000 src=D:\homework\face recognition\deeplearning\diffusion\checkpoints_tiny_ldm\vae32.pt
- 样本数: 4, 容量 4 bits, S_hide=25, S_rec=25
- **VAE 往返上限（无嵌入）: PSNR 16.96 dB**
- 容量报告: `{'z_shape': (4, 8, 8), 'avail_bins_single_channel': 14, 'avail_pairs_all_channels': 56, 'r_min': 1, 'r_max': 3, 'slots': 4, 'obs_per_bit_ceiling': 14.0}`

| strength | PSNR | SSIM | LPIPS | mf clean | dec clean | dec jpeg50 |
|---|---|---|---|---|---|---|
| 0.02 | 14.27 | 0.2401 | n/a | 0.375 | 0.438 | 0.438 |
| 0.05 | 13.62 | 0.2089 | n/a | 0.500 | 0.438 | 0.438 |
| 0.1 | 12.30 | 0.1509 | n/a | 0.625 | 0.438 | 0.438 |
| 0.3 | 9.52 | 0.0826 | n/a | 0.750 | 0.438 | 0.438 |

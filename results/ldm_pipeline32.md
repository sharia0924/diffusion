# LDM 迁移端到端验证

- 空间: 空间=隐空间 (4, 32, 32)，VAE=VAE[native] ch=4 down=2x (2^1) scale=1.00000 src=D:\homework\face recognition\deeplearning\diffusion\checkpoints\vae32.pt
- 样本数: 8, 容量 16 bits, S_hide=150, S_rec=150
- **VAE 往返上限（无嵌入）: PSNR 51.53 dB**
- 容量报告: `{'z_shape': (4, 32, 32), 'avail_bins_single_channel': 342, 'avail_pairs_all_channels': 1368, 'r_min': 3, 'r_max': 15, 'slots': 16, 'obs_per_bit_ceiling': 85.5}`

| strength | PSNR | SSIM | LPIPS | mf clean | dec clean | dec jpeg50 |
|---|---|---|---|---|---|---|
| 0.02 | 38.81 | 0.9798 | n/a | 0.523 | 0.508 | 0.523 |
| 0.05 | 38.26 | 0.9764 | n/a | 0.562 | 0.562 | 0.531 |
| 0.1 | 36.81 | 0.9674 | n/a | 0.664 | 0.641 | 0.641 |
| 0.3 | 28.10 | 0.8654 | n/a | 0.930 | 0.883 | 0.734 |

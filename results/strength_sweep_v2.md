# strength 扫描（载密图质量 vs 准确率）

- 空间: 空间=隐空间 (4, 32, 32)，VAE=VAE[native] ch=4 down=2x (2^1) scale=1.00000 src=D:\homework\face recognition\deeplearning\diffusion\checkpoints\vae32.pt，pixel_res=64
- 无嵌入往返上限: PSNR **51.40 dB**
- 容量: {'z_shape': (4, 32, 32), 'avail_bins_single_channel': 342, 'avail_pairs_all_channels': 1368, 'r_min': 3, 'r_max': 15, 'slots': 80, 'obs_per_bit_ceiling': 17.1}
- 样本 n=32, S_hide=150, S_rec=150

| strength | PSNR | SSIM | LPIPS | mf | dec clean | dec jpeg50 |
|---|---|---|---|---|---|---|
| 0.3 | 26.05 | 0.8082 | n/a | 0.984 | 0.916 | 0.758 |
| 0.2 | 31.07 | 0.9077 | n/a | 0.906 | 0.840 | 0.676 |
| 0.15 | 33.95 | 0.9428 | n/a | 0.818 | 0.768 | 0.633 |
| 0.1 | 36.73 | 0.9662 | n/a | 0.727 | 0.695 | 0.596 |
| 0.05 | 38.77 | 0.9789 | n/a | 0.615 | 0.604 | 0.549 |
| 0.02 | 39.32 | 0.9818 | n/a | 0.545 | 0.523 | 0.512 |

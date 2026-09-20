# strength 扫描（载密图质量 vs 准确率）

- 空间: 空间=隐空间 (4, 32, 32)，VAE=VAE[native] ch=4 down=2x (2^1) scale=1.00000 src=D:\homework\face recognition\deeplearning\diffusion\checkpoints\vae32.pt，pixel_res=64
- 无嵌入往返上限: PSNR **51.53 dB**
- 容量: {'z_shape': (4, 32, 32), 'avail_bins_single_channel': 342, 'avail_pairs_all_channels': 1368, 'r_min': 3, 'r_max': 15, 'slots': 56, 'obs_per_bit_ceiling': 24.428571428571427}
- 样本 n=16, S_hide=150, S_rec=150

| strength | PSNR | SSIM | LPIPS | mf | dec clean | dec jpeg50 |
|---|---|---|---|---|---|---|
| 0.5 | 19.43 | 0.5822 | n/a | 1.000 | 1.000 | 0.984 |
| 0.3 | 27.10 | 0.8403 | n/a | 0.992 | 0.984 | 0.836 |
| 0.2 | 31.81 | 0.9188 | n/a | 0.922 | 0.891 | 0.727 |
| 0.15 | 34.23 | 0.9449 | n/a | 0.820 | 0.797 | 0.695 |
| 0.1 | 36.45 | 0.9632 | n/a | 0.734 | 0.688 | 0.633 |

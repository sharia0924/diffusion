# strength 扫描（载密图质量 vs 准确率）

- 空间: 空间=隐空间 (4, 32, 32)，VAE=VAE[native] ch=4 down=2x (2^1) scale=1.00000 src=D:\homework\face recognition\deeplearning\diffusion\checkpoints\vae32.pt，pixel_res=64
- 无嵌入往返上限: PSNR **51.40 dB**
- 容量: {'z_shape': (4, 32, 32), 'avail_bins_single_channel': 342, 'avail_pairs_all_channels': 1368, 'r_min': 3, 'r_max': 15, 'slots': 56, 'obs_per_bit_ceiling': 24.428571428571427}
- 样本 n=32, S_hide=150, S_rec=150

| strength | PSNR | SSIM | LPIPS | mf | dec clean | dec jpeg50 |
|---|---|---|---|---|---|---|
| 0.3 | 26.13 | 0.8076 | n/a | 0.996 | 0.980 | 0.777 |
| 0.2 | 31.30 | 0.9091 | n/a | 0.957 | 0.914 | 0.699 |
| 0.15 | 34.09 | 0.9420 | n/a | 0.906 | 0.859 | 0.641 |
| 0.1 | 36.64 | 0.9633 | n/a | 0.812 | 0.781 | 0.613 |
| 0.05 | 38.56 | 0.9763 | n/a | 0.684 | 0.645 | 0.574 |
| 0.02 | 39.22 | 0.9807 | n/a | 0.594 | 0.570 | 0.551 |

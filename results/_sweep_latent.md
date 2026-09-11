# strength 扫描（载密图质量 vs 准确率）

- 空间: 空间=隐空间 (4, 16, 16)，VAE=VAE[native] ch=4 down=2x (2^1) scale=1.00000 src=checkpoints/vae16.last.pt，pixel_res=32
- 无嵌入往返上限: PSNR **37.87 dB**
- 容量: {'z_shape': (4, 16, 16), 'avail_bins_single_channel': 70, 'avail_pairs_all_channels': 280, 'r_min': 2, 'r_max': 7, 'slots': 70, 'obs_per_bit_ceiling': 4.0}
- 样本 n=8, S_hide=50, S_rec=50

| strength | PSNR | SSIM | LPIPS | mf | dec clean | dec jpeg50 |
|---|---|---|---|---|---|---|
| 0.0 | 35.11 | 0.9721 | n/a | 0.539 | nan | nan |
| 0.001 | 14.56 | 0.1976 | n/a | 0.461 | nan | nan |
| 0.01 | 14.56 | 0.1979 | n/a | 0.500 | nan | nan |
| 0.05 | 14.56 | 0.1993 | n/a | 0.656 | nan | nan |
| 0.2 | 14.61 | 0.2035 | n/a | 0.898 | nan | nan |
| 1.0 | 8.79 | 0.1383 | n/a | 1.000 | nan | nan |

# strength 扫描（载密图质量 vs 准确率）

- 空间: 空间=隐空间 (4, 8, 8)，VAE=VAE[native] ch=4 down=4x (2^2) scale=1.00000 src=D:\homework\face recognition\deeplearning\diffusion\checkpoints_tiny_ldm\vae32.pt，pixel_res=32
- 无嵌入往返上限: PSNR **16.96 dB**
- 容量: {'z_shape': (4, 8, 8), 'avail_bins_single_channel': 14, 'avail_pairs_all_channels': 56, 'r_min': 1, 'r_max': 3, 'slots': 12, 'obs_per_bit_ceiling': 4.666666666666667}
- 样本 n=4, S_hide=25, S_rec=25

| strength | PSNR | SSIM | LPIPS | mf | dec clean | dec jpeg50 |
|---|---|---|---|---|---|---|
| 0.005 | 14.41 | 0.2431 | n/a | 0.438 | 0.375 | 0.438 |
| 0.02 | 14.26 | 0.2309 | n/a | 0.562 | 0.375 | 0.375 |
| 0.05 | 13.66 | 0.1944 | n/a | 0.625 | 0.375 | 0.438 |
| 0.1 | 12.60 | 0.1456 | n/a | 0.688 | 0.438 | 0.438 |
| 0.3 | 10.44 | 0.0865 | n/a | 0.875 | 0.625 | 0.438 |
| 1.0 | 9.63 | 0.0735 | n/a | 1.000 | 0.750 | 0.375 |

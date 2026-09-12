# strength 扫描（载密图质量 vs 准确率）

- 空间: 空间=隐空间 (4, 32, 32)，VAE=VAE[native] ch=4 down=2x (2^1) scale=1.00000 src=D:\homework\face recognition\deeplearning\diffusion\checkpoints\vae32.pt，pixel_res=64
- 无嵌入往返上限: PSNR **51.95 dB**
- 容量: {'z_shape': (4, 32, 32), 'avail_bins_single_channel': 342, 'avail_pairs_all_channels': 1368, 'r_min': 3, 'r_max': 15, 'slots': 80, 'obs_per_bit_ceiling': 17.1}
- 样本 n=4, S_hide=50, S_rec=50

| strength | PSNR | SSIM | LPIPS | mf | dec clean | dec jpeg50 |
|---|---|---|---|---|---|---|
| 0.0 | 29.40 | 0.8974 | n/a | 0.578 | nan | nan |
| 0.005 | 15.95 | 0.3434 | n/a | 0.578 | nan | nan |
| 0.02 | 15.94 | 0.3430 | n/a | 0.578 | nan | nan |
| 0.05 | 15.94 | 0.3423 | n/a | 0.609 | nan | nan |
| 0.1 | 15.94 | 0.3413 | n/a | 0.641 | nan | nan |
| 0.3 | 16.04 | 0.3399 | n/a | 0.703 | nan | nan |

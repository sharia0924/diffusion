# strength 扫描（载密图质量 vs 准确率）

- 空间: 空间=隐空间 (4, 32, 32)，VAE=VAE[native] ch=4 down=2x (2^1) scale=1.00000 src=D:\homework\face recognition\deeplearning\diffusion\checkpoints\vae32.pt，pixel_res=64
- 无嵌入往返上限: PSNR **51.45 dB**（全局口径）
- 容量: {'z_shape': (4, 32, 32), 'avail_bins_single_channel': 342, 'avail_pairs_all_channels': 1368, 'r_min': 3, 'r_max': 15, 'slots': 80, 'obs_per_bit_ceiling': 17.1}
- 样本 n=24, S_hide=150, S_rec=150
- 注入口径: inject_at=0.35, n_inject=8, inject_mode=add, r_max=None
- PSNR 口径: `PSNR` = 批内全局 MSE（历史口径，随 n 漂移）；`PSNR_img` = 逐图 PSNR 再平均（文献通行口径，通常更高）

| strength | PSNR | PSNR_img | SSIM | LPIPS | mf | mf@jpeg50 | dec clean | dec jpeg50 |
|---|---|---|---|---|---|---|---|---|
| 0.2 | 31.19 | 31.63 | 0.9230 | n/a | 1.000 | 0.833 | 0.990 | 0.802 |
| 0.25 | 30.84 | 31.36 | 0.9164 | n/a | 1.000 | 0.875 | 1.000 | 0.833 |
| 0.3 | 29.73 | 30.43 | 0.8953 | n/a | 1.000 | 0.948 | 1.000 | 0.906 |
| 0.35 | 28.06 | 28.83 | 0.8584 | n/a | 1.000 | 0.969 | 1.000 | 0.969 |
| 0.4 | 26.10 | 26.85 | 0.8063 | n/a | 1.000 | 0.969 | 1.000 | 0.969 |

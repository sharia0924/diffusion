# strength 扫描（载密图质量 vs 准确率）

- 空间: 空间=隐空间 (4, 32, 32)，VAE=VAE[native] ch=4 down=2x (2^1) scale=1.00000 src=D:\homework\face recognition\deeplearning\diffusion\checkpoints\vae32.pt，pixel_res=64
- 无嵌入往返上限: PSNR **51.45 dB**
- 容量: {'z_shape': (4, 32, 32), 'avail_bins_single_channel': 342, 'avail_pairs_all_channels': 1368, 'r_min': 3, 'r_max': 15, 'slots': 56, 'obs_per_bit_ceiling': 24.428571428571427}
- 样本 n=24, S_hide=150, S_rec=150
- 注入口径: inject_at=0.35, n_inject=8, inject_mode=add, r_max=13

| strength | PSNR | SSIM | LPIPS | mf | dec clean | dec jpeg50 |
|---|---|---|---|---|---|---|
| 0.15 | 30.69 | 0.9140 | n/a | 0.802 | 0.792 | 0.615 |
| 0.2 | 30.84 | 0.9161 | n/a | 0.891 | 0.859 | 0.651 |
| 0.25 | 30.73 | 0.9133 | n/a | 0.917 | 0.875 | 0.688 |
| 0.3 | 30.11 | 0.9019 | n/a | 0.969 | 0.932 | 0.766 |
| 0.35 | 28.88 | 0.8780 | n/a | 1.000 | 0.948 | 0.823 |
| 0.4 | 27.18 | 0.8394 | n/a | 1.000 | 0.964 | 0.823 |

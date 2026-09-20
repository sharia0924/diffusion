# KRD-Steg 鲁棒性评测

- 样本数 n=16, 隐藏步数=150, 复原步数=150, 容量=8 bits, strength=0.3
- 空间=隐空间 (4, 32, 32)，VAE=VAE[native] ch=4 down=2x (2^1) scale=1.00000 src=D:\homework\face recognition\deeplearning\diffusion\checkpoints\vae32.pt
- nonce 协议: `nonce_i = H(key || nonce_start+i)`（自包含, 不依赖 cover）, nonce_start=0
- **VAE 往返上限（无嵌入）: PSNR 51.53 dB** —— 这是整条链路的天花板
- 容量报告: {'z_shape': (4, 32, 32), 'avail_bins_single_channel': 342, 'avail_pairs_all_channels': 1368, 'r_min': 3, 'r_max': 15, 'slots': 8, 'obs_per_bit_ceiling': 171.0}
- stego 质量: **PSNR 27.71 dB / SSIM 0.8436 / LPIPS n/a**（LPIPS backend: unavailable）

| 攻击 | 比特准确率 |
|---|---|
| clean | 0.930 |
| jpeg30 | 0.789 |
| jpeg50 | 0.789 |
| jpeg75 | 0.797 |
| noise0.05 | 0.820 |
| noise0.10 | 0.742 |
| blur3x3 | 0.891 |
| blur5x5 | 0.906 |
| resize0.5 | 0.891 |
| resize0.7 | 0.852 |
| bright+0.1 | 0.906 |
| contrast1.2 | 0.875 |
| crop2px | 0.906 |
| crop4px | 0.906 |
| cropresize0.8 | 0.508 |
| rotate5 | 0.711 |
| rotate15 | 0.523 |
| translate2px | 0.516 |
| zoom1.1 | 0.633 |
| regen(t=400) | 0.727 |
| wrong-key | 0.492 |

注: `crop*` 为**真裁剪**（裁边+边缘回填）；`translate*` 为零填充平移；
旧版 `crop` 是 `torch.roll` 循环平移，不具备裁剪语义。
隐空间模型的所有失真都在**像素空间**施加后再编码回隐空间。

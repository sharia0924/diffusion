# KRD-Steg 鲁棒性评测

- 样本数 n=4, 隐藏步数=25, 复原步数=25, 容量=4 bits, strength=0.4
- 空间=隐空间 (4, 8, 8)，VAE=VAE[native] ch=4 down=4x (2^2) scale=1.00000 src=D:\homework\face recognition\deeplearning\diffusion\checkpoints\vae32.pt
- nonce 协议: `nonce_i = H(key || nonce_start+i)`（自包含, 不依赖 cover）, nonce_start=0
- **VAE 往返上限（无嵌入）: PSNR 16.96 dB** —— 这是整条链路的天花板
- 容量报告: {'z_shape': (4, 8, 8), 'avail_bins_single_channel': 14, 'avail_pairs_all_channels': 56, 'r_min': 1, 'r_max': 3, 'slots': 4, 'obs_per_bit_ceiling': 14.0}
- stego 质量: **PSNR 9.75 dB / SSIM 0.0566 / LPIPS n/a**（LPIPS backend: unavailable）

| 攻击 | 比特准确率 |
|---|---|
| clean | 0.688 |
| jpeg30 | 0.438 |
| jpeg50 | 0.438 |
| jpeg75 | 0.438 |
| noise0.05 | 0.438 |
| noise0.10 | 0.438 |
| blur3x3 | 0.438 |
| blur5x5 | 0.438 |
| resize0.5 | 0.438 |
| resize0.7 | 0.500 |
| bright+0.1 | 0.500 |
| contrast1.2 | 0.438 |
| crop2px | 0.438 |
| crop4px | 0.438 |
| cropresize0.8 | 0.500 |
| rotate5 | 0.500 |
| rotate15 | 0.562 |
| translate2px | 0.312 |
| zoom1.1 | 0.562 |
| regen(t=400) | 0.750 |
| wrong-key | 0.500 |

注: `crop*` 为**真裁剪**（裁边+边缘回填）；`translate*` 为零填充平移；
旧版 `crop` 是 `torch.roll` 循环平移，不具备裁剪语义。
隐空间模型的所有失真都在**像素空间**施加后再编码回隐空间。

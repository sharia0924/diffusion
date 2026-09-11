# KRD-Steg 鲁棒性评测

- 样本数 n=4, 隐藏步数=50, 复原步数=50, 容量=16 bits, strength=1.0
- 空间=像素（无 VAE）
- nonce 协议: `nonce_i = H(key || nonce_start+i)`（自包含, 不依赖 cover）, nonce_start=0
- stego 质量: **PSNR 11.33 dB / SSIM 0.1690 / LPIPS n/a**（LPIPS backend: unavailable）

| 攻击 | 比特准确率 |
|---|---|
| clean | 0.984 |
| jpeg30 | 1.000 |
| jpeg50 | 0.984 |
| jpeg75 | 0.984 |
| noise0.05 | 0.984 |
| noise0.10 | 0.984 |
| blur3x3 | 0.969 |
| blur5x5 | 0.969 |
| resize0.5 | 0.938 |
| resize0.7 | 0.969 |
| bright+0.1 | 0.984 |
| contrast1.2 | 0.984 |
| crop2px | 0.984 |
| crop4px | 0.984 |
| cropresize0.8 | 0.469 |
| rotate5 | 0.844 |
| rotate15 | 0.531 |
| translate2px | 0.359 |
| zoom1.1 | 0.828 |
| regen(t=400) | 0.797 |
| wrong-key | 0.562 |

注: `crop*` 为**真裁剪**（裁边+边缘回填）；`translate*` 为零填充平移；
旧版 `crop` 是 `torch.roll` 循环平移，不具备裁剪语义。
隐空间模型的所有失真都在**像素空间**施加后再编码回隐空间。

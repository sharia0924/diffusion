# KRD-Steg 鲁棒性评测

- 样本数 n=32, 隐藏步数=50, 复原步数=50, 容量=16 bits, strength=1.0
- nonce 协议: `nonce_i = H(key || nonce_start+i)`（自包含, 不依赖 cover）, nonce_start=0
- stego 质量: **PSNR 9.55 dB / SSIM 0.1094 / LPIPS n/a**（LPIPS backend: unavailable）

| 攻击 | 比特准确率 |
|---|---|
| clean | 0.994 |
| jpeg30 | 0.992 |
| jpeg50 | 0.994 |
| jpeg75 | 0.998 |
| noise0.05 | 0.998 |
| noise0.10 | 0.992 |
| blur3x3 | 0.992 |
| blur5x5 | 0.984 |
| resize0.5 | 0.961 |
| resize0.7 | 0.992 |
| bright+0.1 | 0.994 |
| contrast1.2 | 0.998 |
| crop2px | 0.994 |
| crop4px | 0.994 |
| cropresize0.8 | 0.465 |
| rotate5 | 0.887 |
| rotate15 | 0.479 |
| translate2px | 0.377 |
| zoom1.1 | 0.857 |
| regen(t=400) | 0.898 |
| wrong-key | 0.490 |

注: `crop*` 为**真裁剪**（裁边+边缘回填）；`translate*` 为零填充平移；
旧版 `crop` 是 `torch.roll` 循环平移，不具备裁剪语义。

# KRD-Steg 鲁棒性评测

- 样本数 n=16, 隐藏步数=50, 复原步数=50, 容量=16 bits, strength=1.0
- nonce 协议: `nonce_i = H(key || nonce_start+i)`（自包含, 不依赖 cover）, nonce_start=0
- stego 质量: **PSNR 9.86 dB / SSIM 0.1153 / LPIPS n/a**（LPIPS backend: unavailable）

| 攻击 | 比特准确率 |
|---|---|
| clean | 0.992 |
| jpeg30 | 0.988 |
| jpeg50 | 0.992 |
| jpeg75 | 1.000 |
| noise0.05 | 0.996 |
| noise0.10 | 0.992 |
| blur3x3 | 0.992 |
| blur5x5 | 0.984 |
| resize0.5 | 0.965 |
| resize0.7 | 0.988 |
| bright+0.1 | 0.992 |
| contrast1.2 | 0.996 |
| crop2px | 0.992 |
| crop4px | 0.992 |
| cropresize0.8 | 0.457 |
| rotate5 | 0.887 |
| rotate15 | 0.457 |
| translate2px | 0.355 |
| zoom1.1 | 0.855 |
| regen(t=400) | 0.871 |
| wrong-key | 0.539 |

注: `crop*` 为**真裁剪**（裁边+边缘回填）；`translate*` 为零填充平移；
旧版 `crop` 是 `torch.roll` 循环平移，不具备裁剪语义。

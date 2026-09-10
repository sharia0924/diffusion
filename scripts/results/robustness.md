# KRD-Steg 评测

- 样本数 n=64, 隐藏步数=50, 复原步数=50, 容量=16 bits, strength=1.0
- stego 质量: **PSNR 9.51 dB / SSIM 0.1022**

| 攻击 | 比特准确率 |
|---|---|
| clean | 0.482 |
| jpeg30 | 0.486 |
| jpeg50 | 0.488 |
| jpeg75 | 0.497 |
| noise0.05 | 0.486 |
| noise0.10 | 0.490 |
| blur3x3 | 0.493 |
| blur5x5 | 0.498 |
| resize0.5 | 0.496 |
| resize0.7 | 0.496 |
| bright+0.1 | 0.497 |
| contrast1.2 | 0.489 |
| crop+2px | 0.494 |
| regen(t=400) | 0.483 |
| wrong-key | 0.484 |

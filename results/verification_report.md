# 结果复核报告（evaluation-only）

- checkpoint: checkpoints/ddpm_latent32.pt + checkpoints/decoder_latent32_best.pt
- n=8, wrong keys=24, strength=0.3
- config: bpb=2, S=150/150, inject_mode=add

| 复核项 | 声明值 | 复测值 |
|---|---|---|
| R1 VAE 往返上限 | 51.53 dB | 51.53 dB |
| R2 无嵌入往返 @S=50 | 29.40 dB | 30.07 dB |
| R2 无嵌入往返 @S=150 | 38.00 dB | 39.00 dB |
| R3 端到端 s=0.3: PSNR | 28.10 dB | 28.10 dB |
| R3 端到端 s=0.3: dec clean | 0.883 | 0.883 |
| R3 端到端 s=0.3: dec jpeg50 | 0.734 | 0.734 |
| R3 端到端 s=0.1: PSNR | 36.81 dB | 36.81 dB |
| R3 端到端 s=0.1: dec clean | 0.641 | 0.641 |
| R4 错密钥 BER mean±std | 0.4977±0.0403 | 0.4997±0.0426 |
| R4 真密钥解码 acc | 0.8984 | 0.8438 |
| R4 mf 距离 真/错 | 6.88 / 15.98 | 8.00 / 16.14 |
| R4 FAR@tau=12 | 0.037 | 0.1375 |
| R4 定位 AUC @N=8 (无nonce/ours) | 0.964 / 0.662 | 0.965 / 0.658 |
| R5 S=50: 往返基线 | 29.40 (像素口径) | 30.07 dB（隐空间口径） |
| R5 S=50: s=0.05 vs 0.1 | 应几乎相同（抹除） | 30.16 / 30.34 dB |
| R5 S=50: s=0.3 应显著下降 | — | 28.68 dB |

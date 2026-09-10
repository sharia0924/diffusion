# 扩散再生攻击（预算化：攻击代价以 **cover** 为参照）

- 样本数 n=16, S_hide=50, S_rec=50, strength=1.0
- 载密图基线: PSNR(vs cover) **9.65 dB**
- LPIPS backend: unavailable

| t_reg | regen steps | 攻击代价 PSNR vs cover | PSNR vs stego | LPIPS vs cover | BER |
|---|---|---|---|---|---|
| 400 | 25 | 9.01 | 9.46 | n/a | 0.090 |
| 800 | 25 | 9.73 | 8.71 | n/a | 0.465 |

注: `PSNR vs cover` 才是攻击者付出的图像质量代价；`vs stego` 仅表示攻击对载密图的改动幅度。t_reg=999 相当于完全重生成，是攻击者上限。
只有 BER 与 (PSNR/LPIPS vs cover) 的联合曲线才支持"攻击者困境"的结论。

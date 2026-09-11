# 复原步数失配鲁棒性（解码器训练于 S_rec=50, S_hide=50, strength=1.0）

- 载密图 PSNR(vs cover) = 9.90 dB
- nonce 协议: `H(key || nonce_start+i)`, nonce_start=0

| S_rec | clean | jpeg50 |
|---|---|---|
| 20 | 1.000 | 1.000 |
| 50 | 1.000 | 1.000 |

> 解读提示：若准确率在很宽的 S_rec 范围内保持 1.000，需先排除“水印信噪比过高导致失配不可见”的解释，再主张几何不变性。

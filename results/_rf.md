# 鲁棒性前沿（strength 扫描, S_hide=50, S_rec=50）

- nonce 协议: `H(key || nonce_start+i)`, nonce_start=0
- 几何攻击为真实实现（crop 真裁剪 / rotate 旋转），LPIPS 为感知距离

| strength | psnr | lpips | clean | jpeg50 | noise0.05 | blur3x3 | crop2px | rotate5 |
|---|---|---|---|---|---|---|---|---|---|
| 0.50 | 15.49 | n/a | 0.516 | 0.625 | 0.688 | 0.578 | 0.516 | 0.656 |
| 1.00 | 10.39 | n/a | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.844 |

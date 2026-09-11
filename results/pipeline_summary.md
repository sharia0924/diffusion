# 流水线运行汇总

- 阶段: ddpm -> decoder -> robust -> p1 -> p2 -> p3
- tiny: False
- 解释器: `D:\anaconda3\envs\yolo\python.exe`
- DDPM 目标 epoch: 300（当前 checkpoint 已完成 300）
- nonce 协议: `H(key || nonce-start+i)`（自包含, 不依赖 cover）, nonce-start=0
- 几何攻击: 真裁剪/旋转/缩放/平移（crop 不再是 torch.roll 循环平移）
- 再生攻击代价: 同时报告 PSNR vs cover 与 vs stego

| 阶段 | 状态 | 用时(min) | 命令 |
|---|---|---|---|
| ddpm | done | 289.1 | `--data-root ./data --out D:\homework\face recognition\deeplearning\diffusion\checkpoints\ddpm_cifar.pt --epochs 300 --batch-size 128 --lr 0.0002 --base 64 --amp off --grad-accum 1 --save-every 10` |

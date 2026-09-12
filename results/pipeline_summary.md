# 流水线运行汇总

- 模式: **LDM 隐空间**
- 阶段: vae -> latent_ddpm -> latent_decoder -> ldm_eval -> strength_sweep -> latent_robust -> latent_p1 -> latent_p2 -> latent_p3
- tiny: False
- 解释器: `D:\anaconda3\envs\yolo\python.exe`
- VAE: resize=64 downsample=1 (latent 32×…) epochs=25
- 隐空间 DDPM: epochs=100, batch=64
- 隐空间解码器: steps=1000, inject_mode=**add**, strength∈[0.05,0.4]
- nonce 协议: `H(key || nonce-start+i)`（自包含, 不依赖 cover）, nonce-start=0
- 几何攻击: 真裁剪/旋转/缩放/平移（crop 不再是 torch.roll 循环平移）
- 再生攻击代价: 同时报告 PSNR vs cover 与 vs stego

| 阶段 | 状态 | 用时(min) | 命令 |
|---|---|---|---|

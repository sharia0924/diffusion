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
| vae | done | 87.5 | `--data-root ./data --out D:\homework\face recognition\deeplearning\diffusion\checkpoints\vae32.pt --epochs 25 --batch-size 64 --base 64 --z-ch 4 --downsample 1 --ch-mults 1,2 --kl-weight 0.0001 --resize 64 --save-every 2 --log-every 200` |
| latent_ddpm | done | 89.0 | `--data-root ./data --out D:\homework\face recognition\deeplearning\diffusion\checkpoints\ddpm_latent32.pt --epochs 100 --batch-size 64 --save-every 10 --log-every 200 --vae-backend native --vae-ckpt D:\homework\face recognition\deeplearning\diffusion\checkpoints\vae32.pt --rebuild-latent-cache` |
| latent_decoder | done | 35.8 | `--ddpm-ckpt D:\homework\face recognition\deeplearning\diffusion\checkpoints\ddpm_latent32.pt --out D:\homework\face recognition\deeplearning\diffusion\checkpoints\decoder_latent32.pt --data-root ./data --steps 1000 --batch-size 16 --inject-mode add --strength-min 0.05 --strength-max 0.4 --sched-noise-prob 0.3 --geom-prob 0.25 --hide-steps 50 --rec-steps 50 --eval-every 250` |

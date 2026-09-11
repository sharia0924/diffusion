# 流水线运行汇总

- 模式: **LDM 隐空间**
- 阶段: vae -> latent_ddpm -> latent_decoder -> ldm_eval -> strength_sweep -> latent_robust -> latent_p1 -> latent_p2 -> latent_p3
- tiny: True
- 解释器: `D:\anaconda3\envs\yolo\python.exe`
- VAE: resize=32 downsample=1 (latent 16×…) epochs=2
- 隐空间 DDPM: epochs=2, batch=64
- 隐空间解码器: steps=30, inject_mode=**add**, strength∈[0.05,0.4]
- nonce 协议: `H(key || nonce-start+i)`（自包含, 不依赖 cover）, nonce-start=0
- 几何攻击: 真裁剪/旋转/缩放/平移（crop 不再是 torch.roll 循环平移）
- 再生攻击代价: 同时报告 PSNR vs cover 与 vs stego

| 阶段 | 状态 | 用时(min) | 命令 |
|---|---|---|---|
| vae | done | 0.5 | `--data-root ./data --out D:\homework\face recognition\deeplearning\diffusion\checkpoints\vae32.pt --epochs 2 --batch-size 64 --base 32 --z-ch 4 --downsample 1 --ch-mults 1,2 --kl-weight 0.0001 --resize 32 --save-every 2 --log-every 200 --tiny` |
| latent_ddpm | done | 0.5 | `--data-root ./data --out D:\homework\face recognition\deeplearning\diffusion\checkpoints\ddpm_latent32.pt --epochs 2 --batch-size 64 --save-every 10 --log-every 200 --vae-backend native --vae-ckpt D:\homework\face recognition\deeplearning\diffusion\checkpoints\vae32.pt --rebuild-latent-cache --tiny` |
| latent_decoder | done | 1.0 | `--ddpm-ckpt D:\homework\face recognition\deeplearning\diffusion\checkpoints\ddpm_latent32.pt --out D:\homework\face recognition\deeplearning\diffusion\checkpoints\decoder_latent32.pt --data-root ./data --steps 30 --batch-size 4 --inject-mode add --strength-min 0.05 --strength-max 0.4 --sched-noise-prob 0.3 --geom-prob 0.25 --hide-steps 25 --rec-steps 25 --eval-every 250 --tiny` |
| ldm_eval | done | 0.4 | `--ddpm-ckpt D:\homework\face recognition\deeplearning\diffusion\checkpoints\ddpm_latent32.pt --data-root ./data --pixel-res 32 --decoder-ckpt D:\homework\face recognition\deeplearning\diffusion\checkpoints\decoder_latent32_best.pt --out D:\homework\face recognition\deeplearning\diffusion\results\ldm_pipeline32.md --n 4 --batch 2 --hide-steps 25 --rec-steps 25 --strengths 0.02,0.05,0.1,0.3` |
| strength_sweep | done | 0.6 | `--ddpm-ckpt D:\homework\face recognition\deeplearning\diffusion\checkpoints\ddpm_latent32.pt --data-root ./data --pixel-res 32 --decoder-ckpt D:\homework\face recognition\deeplearning\diffusion\checkpoints\decoder_latent32_best.pt --out D:\homework\face recognition\deeplearning\diffusion\results\strip32.md --n 4 --batch 2 --hide-steps 25 --rec-steps 25 --strengths 0.005,0.02,0.05,0.1,0.3,1.0` |
| latent_robust | done | 0.4 | `--ddpm-ckpt D:\homework\face recognition\deeplearning\diffusion\checkpoints\ddpm_latent32.pt --data-root ./data --pixel-res 32 --decoder-ckpt D:\homework\face recognition\deeplearning\diffusion\checkpoints\decoder_latent32_best.pt --out D:\homework\face recognition\deeplearning\diffusion\results\robustness_l32.md --n 4 --batch 2 --hide-steps 25 --rec-steps 25 --strength 0.4` |
| latent_p1 | done | 0.2 | `--ddpm-ckpt D:\homework\face recognition\deeplearning\diffusion\checkpoints\ddpm_latent32.pt --data-root ./data --pixel-res 32 --decoder-ckpt D:\homework\face recognition\deeplearning\diffusion\checkpoints\decoder_latent32_best.pt --out D:\homework\face recognition\deeplearning\diffusion\results\key_security_l32.md --n 4 --batch 2 --n-wrong 20 --hide-steps 25 --rec-steps 25 --strength 0.4` |
| latent_p2-grid | done | 0.2 | `--ddpm-ckpt D:\homework\face recognition\deeplearning\diffusion\checkpoints\ddpm_latent32.pt --data-root ./data --pixel-res 32 --decoder-ckpt D:\homework\face recognition\deeplearning\diffusion\checkpoints\decoder_latent32_best.pt --out D:\homework\face recognition\deeplearning\diffusion\results\steps_grid_l32.md --n 4 --batch 2 --hide-list 4,10 --rec-list 4,10 --strength 0.4` |
| latent_p2-mismatch | done | 0.2 | `--ddpm-ckpt D:\homework\face recognition\deeplearning\diffusion\checkpoints\ddpm_latent32.pt --data-root ./data --pixel-res 32 --decoder-ckpt D:\homework\face recognition\deeplearning\diffusion\checkpoints\decoder_latent32_best.pt --out D:\homework\face recognition\deeplearning\diffusion\results\step_mismatch_l32.md --n 4 --batch 2 --rec-list 4,10,25 --hide-steps 25 --strength 0.4` |
| latent_p3-regen | done | 0.2 | `--ddpm-ckpt D:\homework\face recognition\deeplearning\diffusion\checkpoints\ddpm_latent32.pt --data-root ./data --pixel-res 32 --decoder-ckpt D:\homework\face recognition\deeplearning\diffusion\checkpoints\decoder_latent32_best.pt --out D:\homework\face recognition\deeplearning\diffusion\results\regen_l32.md --n 4 --batch 2 --rec-steps 25 --t-regs 400 --regen-steps-list 25 --hide-steps 25 --strength 0.4` |

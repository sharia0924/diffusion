# 流水线运行汇总

- 阶段: ddpm -> decoder -> robust -> p1 -> p2 -> p3
- tiny: False
- nonce 协议: `H(key || nonce-start+i)`（自包含, 不依赖 cover）, nonce-start=0
- 几何攻击: 真裁剪/旋转/缩放/平移（crop 不再是 torch.roll 循环平移）
- 再生攻击代价: 同时报告 PSNR vs cover 与 vs stego

| 阶段 | 状态 | 用时(min) | 命令 |
|---|---|---|---|
| ddpm | done | 29.6 | `--data-root ./data --out /root/diffusion/checkpoints/ddpm_cifar.pt --epochs 60 --batch-size 128` |
| decoder[main] | done | 25.7 | `--ddpm-ckpt /root/diffusion/checkpoints/ddpm_cifar.pt --out /root/diffusion/checkpoints/decoder.pt --data-root ./data --steps 1000 --batch-size 16 --sched-noise-prob 0.3 --geom-prob 0.25 --hide-steps 50 --rec-steps 50` |
| robust[main] | done | 0.9 | `--ddpm-ckpt /root/diffusion/checkpoints/ddpm_cifar.pt --data-root ./data --nonce-start 0 --decoder-ckpt /root/diffusion/checkpoints/decoder_best.pt --out /root/diffusion/results/robustness.md --n 64 --batch 16 --hide-steps 50 --rec-steps 50 --strength 1.0 --regen-t 400` |
| p1 | done | 0.6 | `--ddpm-ckpt /root/diffusion/checkpoints/ddpm_cifar.pt --data-root ./data --nonce-start 0 --decoder-ckpt /root/diffusion/checkpoints/decoder_best.pt --out /root/diffusion/results/key_security.md --n 32 --batch 16 --n-wrong 200 --hide-steps 50 --rec-steps 50 --strength 1.0` |
| p2-grid | done | 0.7 | `--ddpm-ckpt /root/diffusion/checkpoints/ddpm_cifar.pt --data-root ./data --nonce-start 0 --decoder-ckpt /root/diffusion/checkpoints/decoder_best.pt --out /root/diffusion/results/steps_grid.md --n 24 --batch 16 --hide-list 10,25,50,100 --rec-list 10,25,50,100 --strength 1.0` |
| p2-mismatch | done | 0.4 | `--ddpm-ckpt /root/diffusion/checkpoints/ddpm_cifar.pt --data-root ./data --nonce-start 0 --decoder-ckpt /root/diffusion/checkpoints/decoder_best.pt --out /root/diffusion/results/step_mismatch.md --n 24 --batch 16 --rec-list 20,30,40,45,50,55,60,70,80 --hide-steps 50 --strength 1.0` |
| p3-regen[main] | done | 0.4 | `--ddpm-ckpt /root/diffusion/checkpoints/ddpm_cifar.pt --data-root ./data --nonce-start 0 --decoder-ckpt /root/diffusion/checkpoints/decoder_best.pt --out /root/diffusion/results/regen.md --n 32 --batch 16 --rec-steps 50 --t-regs 200,400,600,800 --regen-steps-list 25,50 --hide-steps 50 --strength 1.0` |
| p3-frontier | done | 1.0 | `--ddpm-ckpt /root/diffusion/checkpoints/ddpm_cifar.pt --data-root ./data --nonce-start 0 --decoder-ckpt /root/diffusion/checkpoints/decoder_best.pt --out /root/diffusion/results/frontier.md --n 32 --batch 16 --rec-steps 50 --hide-steps 50 --strengths 0.5,0.75,1.0,1.25,1.5,2.0` |

# 远程 2080（22GB）训练计划：把 latent 从 32×32 扩到 64×64

> 目的：本地 32×32 latent 的观测数天花板已被吃满（§16：N=32 时 jpeg50 只到
> 0.879@30.1dB）。**N 的上限 ∝ latent 边长²**，所以结构性提升只能靠更大的 latent。
> 本文件给出可直接复制执行的命令、资源估算与判定门槛。

## 0. 为什么是"扩大 latent"而不是别的

| 已实测否定的杠杆 | 结论 | 出处 |
|---|---|---|
| S: 150→250 | 同准确率 PSNR 反而 -0.6 dB（扰动被更长的轨迹放大） | §12.3 |
| 接收端加权/软判决 | 所有 MRC/逆方差/幅度加权都不如等权平均 | §13.1 |
| r_max 下移（搬到低频） | 等 PSNR 下 jpeg50/jpeg75 无稳定收益 | §15 |
| ecc/bpb/check 重分配（N 顶满） | 有效但饱和：N=18→32 只换来 jpeg50 +0.043 | §16 |
| 提高 S / 更强基座 DDPM | 只能抬"往返地板"，收益被投影放大效应抵消 | §12.2/§12.3 |

唯一没被否定、且天花板随 latent 面积线性增长的就是 **N 本身**。
当前 res=32 的频点预算是 342 对；res=64 时（r_min=3，r_max=Nyquist）约
**1600 对**，理论 N 可到 160（现为 32）。

## 1. 三阶段命令（按顺序跑，每阶段有判定门槛）

### 阶段 1：训练 64×64 latent 的 VAE（约 1.5–2h）

```bash
# 关键：--downsample 0 表示不下采样，latent 与像素同尺寸 64×64。
# 注意 NativeVAE 有断言 downsample == len(ch_mults)-1，所以 downsample=0 必须配 --ch-mults 1
python scripts/train_vae.py --data-root ./data --out checkpoints/vae64.pt \
  --epochs 30 --batch-size 64 --base 64 --z-ch 4 --downsample 0 \
  --ch-mults 1 --kl-weight 1e-4 --resize 64 --save-every 5 --log-every 200
```

判定门槛：`python scripts/diag_latent_roundtrip.py`（或 train_vae 的日志）
报告 encode→decode 往返 PSNR **≥ 51 dB**（当前 32×32 latent 的 VAE 是 51.5 dB）。
低于 48 dB 就加大 `--base` 或延长 epoch，别往下走。

### 阶段 2：训练 64×64 latent 的 DDPM（约 6–8h，22GB 卡可加大 batch）

```bash
# latent 尺寸由 VAE 决定并自动记录（train_ddpm 会把 latent_shape 写进 ckpt，
# 下游 train_decoder 用它当 res；没有 --latent-res 这个参数）
python scripts/train_ddpm.py --data-root ./data --out checkpoints/ddpm_latent64.pt \
  --vae-backend native --vae-ckpt checkpoints/vae64.pt \
  --resize 64 --epochs 120 --batch-size 64 --base 96 --save-every 10 \
  --grad-accum 1 --amp auto --channels-last --cudnn-benchmark
```

- latent 尺寸 ×4 → 单步显存约 ×4；`--base 96` 让 UNet 也更大一些；
  22GB 下 batch 64 + channels-last + AMP 应可跑；OOM 就把 batch 降到 32。
- 判定门槛：往返 PSNR（无嵌入）**≥ 42 dB @ S=150**（当前 32×32 是 38.00 dB）。
  这就是目标 35 dB 时的"往返地板"——地板低于 40 dB，35 dB 目标就没有余量（§12.2）。

### 阶段 3：在新 latent 上训解码器（约 2–3h）

```bash
python scripts/train_decoder.py --ddpm-ckpt checkpoints/ddpm_latent64.pt \
  --out checkpoints/decoder_latent64_v1.pt --data-root ./data \
  --steps 1200 --batch-size 16 \
  --n-bits 8 --ecc-reps 8 --n-check-bits 16 --inject-mode add \
  --inject-at 0.35 --n-inject 8 --strength-min 0.15 --strength-max 0.4 \
  --sched-noise-prob 0.3 --geom-prob 0.25 --hide-steps 150 --rec-steps 150 \
  --eval-strength 0.3 --eval-every 200 --mf-residual
```

注意 `fit_capacity` 会自动把 bpb 顶到 `1600//80 = 20`（N=bpb×ecc=160）。
若显存/时间吃紧，可用 `--ecc-reps 4 --n-check-bits 16`（槽位 48，bpb 33，N=132）。

### 阶段 4：等 PSNR 判定（半小时，可先于阶段 3 用 mf 接收做）

```bash
python scripts/diag_ecc_frontier.py --n 32 --target-psnr 30 \
  --configs "3/32,8/16,16/16,32/16" --out results/ecc_frontier64.json
```

判定门槛（决定是否值得写进论文）：
- 在同一 PSNR（如 30 dB）下，jpeg50 从当前的 0.836 提到 **≥ 0.92**，
  或 jpeg50=0.95 所需的 PSNR 从 ~25 dB 提到 **≥ 32 dB** → 值得；
- 若提升仍 < 0.03，说明瓶颈不在观测数，而在**信道误差的空间相关性**
  （JPEG 8×8 块结构）——此时应转向"跨块交织"或降低载荷，而不是继续加 latent。

## 2. 资源与时间估算（22GB 2080）

| 阶段 | 本地 1650 4GB | 2080 22GB（估） |
|---|---|---|
| VAE 64×64 30ep | 会 OOM | 1.5–2 h |
| DDPM 64×64 latent 120ep | 不可行 | 6–8 h |
| 解码器 1200 步（N=160） | 会 OOM | 2–3 h |
| 等 PSNR 扫描 | — | 0.5 h |

## 3. 顺带的两个必做项（与本计划无关，但论文前必须补）

1. **报数口径**：所有 PSNR 必须注明"批内全局 MSE"还是"逐图平均"（§14 实测两者
   差 1.3 dB、子集抖动差 3 倍），且 n ≥ 32。
2. **同环境对比基线**：Tree-Ring / ZoDiac / MDDM 需要在同一 VAE+DDPM 上重跑，
   否则"我们更好"没有说服力（见 §5 的创新点自评）。

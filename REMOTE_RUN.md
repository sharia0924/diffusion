# 远程算力平台训练手册（22GB 2080 级）

> 目标：把 latent 从 32×32 扩到 **64×64**，把每比特观测数 N 从 **32 提到 144**，
> 从而同时抬高 PSNR 上限与鲁棒性。此前所有结论见 [ITERATION_LOG.md](ITERATION_LOG.md)
> §12–§22；门槛与差距见 §12/§18。
>
> ⚠️ 开工前先看 §22：**不要启用几何同步（`--mag-profile` / `--sync`）**，
> 已实测否定（定位命中 0/6，且 clean 从 1.000 掉到 0.562）。

## 0. 环境

```bash
# 依赖：torch>=2.1 + torchvision，CUDA 可用
python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
pip install -r requirements.txt 2>/dev/null || pip install torch torchvision numpy pillow
```

数据根目录默认 `./data`（CIFAR-10 会自动下载）。

## 1. 主线路：64×64 latent（推荐，一条命令跑完训练三段）

```bash
python scripts/run_pipeline.py --ldm --stages vae latent_ddpm latent_decoder --finish \
  --vae-resize 64 --vae-downsample 0 --vae-ch-mults 1 --vae-base 64 --vae-epochs 30 \
  --vae-batch 64 --vae-kl-weight 1e-4 \
  --latent-ddpm-epochs 120 --latent-ddpm-batch 64 --latent-ddpm-base 96 \
  --ddpm-amp auto --ddpm-channels-last --cudnn-benchmark \
  --num-workers 6 --force-lock
```

关键点（都已本地冒烟验证过，见 §23）：

| 参数 | 值 | 为什么 |
|---|---|---|
| `--vae-downsample 0` | 不下采样 | 得到 **latent 64×64**（频点预算 1488 对，是 32×32 的 4.4 倍） |
| `--vae-ch-mults 1` | 单级 | NativeVAE 断言 `downsample == len(ch_mults)-1`，0 必须配 `1` |
| `--vae-resize 64` | 64 | 像素尺寸；DDPM/评测必须一致（否则 latent 尺寸与评测 PSNR 全错） |
| `--latent-ddpm-base 96` | 更大的 UNet | 22GB 卡有余量；本地 4GB 用 base 32 也能跑（峰值 0.2 GiB） |
| `--n-bits 8 --ecc-reps 8 --n-check-bits 16` | 默认已是 | 槽位 80 → **bpb 自动顶到 18 → N=144** |

判定门槛（不达标就别往下走）：

1. **VAE 往返**（`train_vae` 末行 `[summary]` 或 `diag_latent_roundtrip.py`）**≥ 51 dB**；
2. **DDPM 无嵌入往返**（`eval_ldm_pipeline.py` 的 `baseline`）**≥ 42 dB @ S=150**；
   这一条最关键：35 dB 目标需要往返地板留出足够嵌入预算（§12.2）。
3. 解码器训练日志里 `[capacity] res=64 ... bpb=18 (slots=80, pairs=1440)`。

### 1.1 训练完立刻看这三张表

```bash
python scripts/eval_strength_sweep.py --decoder-ckpt checkpoints/decoder_latent64_best.pt \
  --n 24 --batch 8 --strengths 0.15,0.2,0.25,0.3,0.35,0.4 --pixel-res 64 \
  --out results/strength_sweep_latent64.md          # 前沿 + mf vs 解码器

python scripts/eval_key_security.py --decoder-ckpt checkpoints/decoder_latent64_best.pt \
  --n 32 --n-wrong 200 --strength 0.3 --pixel-res 64 \
  --out results/key_security_latent64.md            # P1：BER/FAR/AUC/密钥空间

python scripts/eval_robustness.py --decoder-ckpt checkpoints/decoder_latent64_best.pt \
  --n 64 --batch 8 --strength 0.3 --pixel-res 64 \
  --out results/robustness_latent64.md              # 鲁棒性全表
```

### 1.2 关键判定：N=144 能不能把前沿左移

```bash
python scripts/diag_ecc_frontier.py --n 32 --target-psnr 30 \
  --configs "8/16,16/16,32/16" --out results/ecc_frontier64.json
# 64×64 latent 下 fit_capacity 会给出 bpb 18/9/4（对应 N=144/72/32）
```

- 若同一 PSNR 下 **jpeg50 从 0.879 提到 ≥ 0.93**，或 **jpeg50=0.95 所需 PSNR
  从 26.3 dB 提到 ≥ 31 dB** → 结构性路线成立，写进论文主结果；
- 若提升 < 0.03 → 说明瓶颈是"观测间误差相关"，不要再加 latent，
  转去补几何同步（需要重做 §22 的负结果）或降载荷。

## 2. 备选线路：本地 32×32 latent 复跑（与已有结果对齐的口径）

```bash
# 只重训解码器（VAE/DDPM 复用现有 checkpoints/ddpm_latent32.pt）
python scripts/train_decoder.py --ddpm-ckpt checkpoints/ddpm_latent32.pt \
  --out checkpoints/decoder_latent32_v4.pt --steps 1000 --batch-size 16 \
  --eval-strength 0.3 --mf-residual --num-workers 6
```

默认值已经是当前最优工作点（8bit / ecc8 / check16 / inject_at=0.35 / n_inject=8 /
S=150 / strength∈[0.15,0.4]），**不需要再传一堆参数**。
参照基准（v3，本地跑出）：clean mf 1.000 @ 29.93 dB、jpeg50 mf 0.859、
鲁棒性全表见 §20.4。

## 3. 评测务必要遵守的三条口径（§14/§16）

1. **PSNR 两种口径都报**：批内全局 MSE（`PSNR`，历史口径、随 n 漂移）与
   逐图平均（`PSNR_img`，文献口径，通常高 0.5–1.3 dB）；**n ≥ 32**；
2. **主接收机写 matched filter**（`mf` / `mf@jpeg50` 列）；解码头只作消融（§18.2）；
3. **接收端反演 20–50 步即可**（§20.1：20 步 0.990/0.859 vs 150 步 0.995/0.859），
   省 3–7.5× 算力，且无需步数同步。

## 4. 时间与显存预算（22GB 2080，估）

| 阶段 | 配置 | 时间 |
|---|---|---|
| VAE 64×64（latent 64×64） | 30 epoch, batch 64, base 64 | 1.5–2 h |
| latent DDPM（64×64 latent） | 120 epoch, batch 64, base 96, AMP | 6–8 h |
| 解码器 | 1000 步, batch 16, N=144 | 2–3 h |
| 三张评测表 | — | 1 h |

本地 4GB 实测：latent 64×64 的 DDPM 在 `base 32 / batch 8` 下峰值显存 **0.2 GiB**，
所以 22GB 卡可以把 `base 96 / batch 64` 轻松跑满；若 OOM，先把 batch 减半、
再用 `--ddpm-grad-accum` 补等效批大小。

## 5. 一键脚本的自检（远程也建议先跑一遍）

```bash
python scripts/run_pipeline.py --ldm --stages vae latent_ddpm latent_decoder --tiny \
  --vae-resize 64 --vae-downsample 0 --vae-ch-mults 1 --force-lock   # 数分钟冒烟
python tests/strength_semantics_test.py     # 注入口径回归（不需要 GPU）
python tests/eval_workpoint_guard_test.py   # 评测口径护栏（不需要 GPU）
python tests/psnr_convention_test.py        # PSNR 口径回归
python tests/hide_legacy_equiv_test.py      # 老写法等价性（CPU，数分钟）
python scripts/check_ninject.py             # 训练/评测注入口径一致性
```

`--tiny` 现在**只缩数据量与轮数，不改架构**（§23），所以冒烟通过 = 目标几何能跑通。

# KRD-Steg：密钥条件化扩散轨迹隐写（原型实现）

用扩散模型（U-Net DDPM）做图像隐写：

- **隐藏端步数自由**：cover → DDIM 反演（步数任选）→ 向 x_T 频谱注入密钥门控的比特图案 → DDIM 采样回载密图；
- **复原端公式驱动、步数固定**：stego → 闭式 DDIM 更新式迭代固定 S_rec 步 → 按密钥取频点特征 → 轻量 MLP 解码出比特；
- **密钥门控**：密钥（任意字符串）经 SHA256 确定性决定频点/相位/幅度，错密钥 → 比特准确率 ≈ 50%；
- **鲁棒性**：解码器在可微 JPEG / 噪声 / 模糊 / 缩放 / **几何失真**下训练。

完整方案（文献综述、可行性、赛道性价比、实验计划）见 [PROPOSAL.md](PROPOSAL.md)；
当前问题的诊断与下一步路线见 [NEXT_STEPS.md](NEXT_STEPS.md)。

## v2 评测口径修复（务必先读）

上一轮的所有结果都是在**有缺陷的评测口径**下得到的，以下问题已修复，**旧结果表不可再引用**：

| 问题 | 修复 |
|---|---|
| `crop` 攻击用 `torch.roll` 实现，是**循环平移**而非裁剪（频谱幅度完全不变） | `crop_fill` 真裁剪（裁边 + 边缘回填）；新增 `cropresize`/`rotate`/`translate`/`zoom` |
| 再生攻击的"攻击代价"是 `PSNR(regen, stego)`，参照物是被污染过的载密图 | 改为 `PSNR(regen, cover)` + `LPIPS(regen, cover)`，同时保留 vs stego 仅供对照 |
| nonce = `H(cover)`，必须靠 PNG 元数据传给复原端 → 方案非盲 | nonce = `H(key \|\| counter)`，自包含、**不依赖 cover**，真盲提取 |
| 把"槽位定位 AUC=0.79"当成"攻击失败" | 明确该指标是**存在性定位**而非密钥恢复；新增 §E 盲检测 AUC、§D 残差一致性 |
| 报"密钥空间 1354 bit"，与 `token_key()` 的 64 bit 真实熵冲突 | 分列报告"派生参数空间下界"与"密钥本身熵" |
| 缺 LPIPS/FID 等不可见性指标 | 新增 `krd/perceptual.py`（LPIPS 可选依赖，缺失时优雅降级） |

诊断证据与复现命令见 [NEXT_STEPS.md](NEXT_STEPS.md) §2 与 `scripts/diag_psnr_probe.py`。

## 环境

conda（推荐，配置见 [environment.yml](environment.yml)）：

```bash
conda env create -f environment.yml   # 默认 CUDA 12.6 wheel, 可按文件头注释切换 GPU/CPU 变体
conda activate krd-steg
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

或 pip 直装（见 [requirements.txt](requirements.txt)）。装好后先跑冒烟测试：

```bash
python tests/smoke_test.py        # 冒烟测试（CPU 数秒, 20 项检查含安全性质）
```

## 快速开始（CIFAR-10, 32×32, 容量 16 比特）

```bash
# 0) 一键流水线（推荐）: 预训练 → 解码器 → P1/P2/P3 全部评测, 断点续跑
python scripts/run_pipeline.py --stages all          # 全流程（训练+评测）
python scripts/run_pipeline.py --stages all --tiny   # 冒烟验证流水线（数分钟）
python scripts/run_pipeline.py --stages p1 p2 p3     # 只重跑评测（需已有 checkpoint）
python scripts/run_pipeline.py --stages all --decoder-ablation  # 额外训练 P3 消融对照

# 1) 预训练基础 DDPM（消费级 GPU 约 2-5 小时；--tiny 可冒烟）
python scripts/train_ddpm.py --epochs 60 --batch-size 128

# 2) 训练密钥条件复原解码器（冻结 DDPM；失真感知训练；默认 1000 步已足够收敛，
#    探针实验 150 步即达 clean=1.000；--sched-noise-prob 0.3 加再生攻击代理，
#    --geom-prob 0.25 让几何失真进入训练混合，与评测口径对齐）
python scripts/train_decoder.py --ddpm-ckpt checkpoints/ddpm_cifar.pt

# 3) 隐藏 / 复原（nonce 协议自包含: nonce = H(key || counter)）
python scripts/run_stego.py hide --input cover.png --output stego.png \
    --key my-secret --message "ZC" --counter 0 --ddpm-ckpt checkpoints/ddpm_cifar.pt
python scripts/run_stego.py recover --input stego.png --key my-secret --counter 0 \
    --ddpm-ckpt checkpoints/ddpm_cifar.pt --decoder-ckpt checkpoints/decoder_best.pt \
    --message "ZC"

# 4) 三大护栏评测
python scripts/eval_robustness.py --ddpm-ckpt checkpoints/ddpm_cifar.pt \
    --decoder-ckpt checkpoints/decoder_best.pt --n 64 --out results/robustness.md
# P1 密钥安全: 错密钥BER分布 / matched-filter 校验FAR / 槽位定位AUC-N / 盲检测AUC / 密钥空间
python scripts/eval_key_security.py --ddpm-ckpt checkpoints/ddpm_cifar.pt \
    --decoder-ckpt checkpoints/decoder_best.pt --n 32 --out results/key_security.md
# P2 步数不对称: S_hide×S_rec 网格热图 + 复原步数失配曲线
python scripts/eval_steps_grid.py --ddpm-ckpt checkpoints/ddpm_cifar.pt \
    --decoder-ckpt checkpoints/decoder_best.pt --n 24 --out results/steps_grid.md
python scripts/eval_step_mismatch.py --ddpm-ckpt checkpoints/ddpm_cifar.pt \
    --decoder-ckpt checkpoints/decoder_best.pt --out results/step_mismatch.md
# P3 再生鲁棒性: 预算化攻击网格（代价以 cover 为参照）+ 隐秘性-鲁棒性前沿
python scripts/eval_regen.py --ddpm-ckpt checkpoints/ddpm_cifar.pt \
    --decoder-ckpt checkpoints/decoder_best.pt --n 32 --out results/regen.md
python scripts/plot_frontier.py --ddpm-ckpt checkpoints/ddpm_cifar.pt \
    --decoder-ckpt checkpoints/decoder_best.pt --n 32 --out results/frontier.md

# 5) 诊断探针：分离"DDIM 往返误差"与"图案注入误差"，对比三种注入方案
python scripts/diag_psnr_probe.py --n 6 --device cpu --out results/diag_psnr_probe.json
```

提示：训练解码器时可加 `--sched-noise-prob 0.3` 把"调度坐标加噪"（扩散再生攻击的
训练代理）纳入失真混合，显著提升再生攻击下的鲁棒性。

## 目录结构

```
krd/                 核心库
  schedule.py        DDPM 调度 + DDIM 采样/反演（"复原公式"）
  unet.py            小型 U-Net（32²）
  pattern.py         密钥(+nonce)→频点/相位/幅度；频谱注入；环带特征；ECC；校验比特
  decoders.py        RingDecoder（MLP 解码头）
  stego.py           TrajStego：hide / recover / 潜变量缓存 / 再生攻击
  security.py        密钥安全组件：错密钥分布 / matched-filter 校验 / 槽位定位 AUC
                     / 盲水印检测 AUC / 残差一致性 / 密钥空间与真实熵
  distortions.py     可微 JPEG + 经典失真 + **真实几何攻击** + 调度坐标加噪
  perceptual.py      LPIPS（可选依赖，缺失时优雅降级）
  metrics.py         PSNR / SSIM / 比特准确率
scripts/             train_ddpm / train_decoder / run_stego / run_pipeline
                     + eval_common（统一取数与 nonce 协议）
                     + eval_robustness / eval_key_security / eval_steps_grid
                     / eval_step_mismatch / eval_regen / plot_frontier
                     + diag_psnr_probe（测量口径诊断探针）
tests/smoke_test.py  冒烟测试（含几何攻击与 nonce 协议检查）
```

## 说明与限制

- 原型运行在像素空间 32×32，容量 16 比特（≈2 字节）；正式论文应迁移到
  Stable Diffusion / LDM 隐空间（`krd/pattern.py`、`krd/stego.py` 与分辨率无关，可平移）。
- 隐藏质量受 DDIM 往返重建误差限制（与基础模型质量成正比）；`--strength` 控制隐秘性-鲁棒性折中。
  **实测无嵌入往返上限仅 ~18 dB**（60 epoch DDPM），注入后降到 ~9.5 dB，详见 NEXT_STEPS.md §2。
- ECC 目前是重复码 + 组内均值（软合并）；正式版可换 BCH/LDPC。
- nonce 计数器是**公开参数**：攻击者知道 counter 时可复现图案，因此逐图 nonce 降低的是
  "跨图累积优势"，不能当作密码学保证。
- LPIPS 需额外安装：`pip install lpips`（未安装时报告中标注 `n/a`，不会中断流水线）。

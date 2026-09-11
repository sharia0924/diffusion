# LDM 迁移（隐空间扩散隐写）

本文件说明从**像素空间 DDPM** 迁移到**隐空间扩散模型（LDM）**的实现、用法与验收标准。

## 0. 为什么要迁移（数据依据）

本机 300-epoch 像素空间 DDPM 的实测（见 `RESULTS_ANALYSIS_local300.md`）：

| 量 | 数值 |
|---|---|
| 无嵌入 DDIM 往返 PSNR（**整条链路的上限**） | 18.81 dB |
| strength=1.0 载密图 PSNR | 10.16 dB |
| 论文要求 | ≥ 35 dB |

**即使不嵌入任何信息，像素空间路线的上限也只有 18.8 dB** —— 差距来自基础模型的
往返保真度，与嵌入方案无关。隐空间路线的价值在于两点：

1. **VAE 解码器把往返保真度拉到 30 dB+**（先解决"上限"问题）；
2. **频点预算 ∝ 分辨率²**：像素 32² 单通道半平面约 176 个可用频点，
   而隐空间（多通道）可用频点数提升一个量级 → **每比特观测数**提升，
   解码所需的信噪比下降。

## 1. 架构

```
训练侧：
  CIFAR 图像 --train_vae.py--> VAE（f=2^down 下采样, z_ch 通道）
  CIFAR 图像 --VAE encode(确定式 mu)--> 潜变量 --缓存到磁盘--> train_ddpm.py 训隐空间扩散
复原/隐藏侧（与像素空间同一套代码，只是换空间）：
  cover --VAE encode--> z --DDIM 反演--> z_T --注入密钥图案--> z_T' --DDIM 采样--> 载密 z
  载密图 = VAE decode(载密 z)              <-- 图像级指标都在这里算
  信道：载密 z --decode--> 像素 --失真(JPEG/噪声/几何)--> --encode--> 载密 z'
```

关键实现点：

- **失真必须定义在像素空间**（JPEG/噪声/几何攻击都是像素域概念），
  所以隐空间模型的信道是 `decode -> 失真 -> encode`。这个"解码-再编码"
  本身就是 VAE 带来的额外损耗，**必须在训练解码器时就模拟**，否则训练/测试不一致。
- 潜变量用**确定性编码（mu，不采样）**，保证同一张图每次编码一致 ——
  隐写/水印需要可复现性。

## 2. 新增文件

| 文件 | 作用 |
|---|---|
| `krd/vae.py` | VAE 后端：`native`（自带小型 VAE，无额外依赖）/ `sd`（Stable Diffusion VAE，需 `pip install diffusers`），统一 `encode/decode` 接口 |
| `krd/latent.py` | 潜变量**离线缓存**、`LatentTensorDataset`、容量/每比特观测数报告 |
| `scripts/train_vae.py` | 训练原生 VAE（支持断点续训），结束时报告 VAE 往返 PSNR |
| `scripts/eval_ldm_pipeline.py` | **端到端验收**：潜空间隐藏→复原→PSNR/SSIM/LPIPS/准确率，并可 `--pixel-baseline` 对照 |
| `scripts/dbg_vae.py` | VAE 逐层形状排查 |

## 3. 用法

```bash
# 1) 训练原生 VAE（本机 GTX1650：约 5 min/epoch）
#    默认用 2 级下采样 -> latent 16×16×4，容量预算是像素空间的 1.85 倍
python scripts/train_vae.py --epochs 20 --batch-size 128 --base 64 \
    --z-ch 4 --downsample 2 --ch-mults 1,2,4 --kl-weight 1e-4 \
    --out checkpoints/vae_cifar_d2.pt
#    对照实验：4 级下采样 -> latent 8×8（容量反而低于像素空间，用于验证"分辨率决定容量"）
python scripts/train_vae.py --epochs 15 --downsample 3 --ch-mults 1,2,4,8 \
    --out checkpoints/vae_cifar.pt

# 2) 在隐空间训练扩散模型（自动编码并缓存潜变量）
python scripts/train_ddpm.py --epochs 200 --batch-size 128 \
    --vae-backend native --vae-ckpt checkpoints/vae_cifar_d2.pt \
    --out checkpoints/ddpm_latent.pt

# 3) 训练隐空间解码器（失真在像素空间施加，再编码回隐空间）
python scripts/train_decoder.py --ddpm-ckpt checkpoints/ddpm_latent.pt \
    --out checkpoints/decoder_latent.pt --steps 1000

# 4) 端到端验收 + 容量核算
python scripts/eval_ldm_pipeline.py --ddpm-ckpt checkpoints/ddpm_latent.pt \
    --decoder-ckpt checkpoints/decoder_latent_best.pt --out results/ldm_pipeline.md
python scripts/report_capacity.py --slots 80
python tests/ldm_smoke_test.py
#    与像素空间基线对照（同一样本、同一strength）
python scripts/eval_ldm_pipeline.py --pixel-baseline \
    --ddpm-ckpt checkpoints/ddpm_cifar.pt --out results/ldm_pipeline_pixel.md
```

### 用预训练 SD VAE（远程，推荐出正式数字）

```bash
pip install diffusers
python scripts/train_ddpm.py --epochs 200 --vae-backend sd \
    --out checkpoints/ddpm_latent_sd.pt
```

SD VAE 是 f8 下采样、4 通道 latent、scaling_factor≈0.18215，重建保真度远高于
自带小 VAE（512² 图约 30 dB+）。

## 4. 容量核算：为什么隐空间能降低嵌入功率

`scripts/report_capacity.py` 输出（槽位 80，即当前 16bit 消息 + 32bit 校验的配置）：

| 空间 | 环带对(单通道) | 可用对(全通道) | obs/bit 上限 |
|---|---|---|---|
| 像素 32×32, 1ch | 342 | 342 | 4.28 |
| 隐空间 8×8, 4ch | 14 | 56 | **0.70** |
| 隐空间 16×16, 4ch | 70 | 280 | 3.50 |
| 隐空间 32×32, 4ch | 342 | 1368 | 17.10 |
| 隐空间 64×64, 4ch | 1488 | 5952 | 74.40 |
| 隐空间 128×128, 4ch | 6214 | 24856 | 310.70 |

**关键发现（一个反直觉的坑）**：`obs/bit` 随**分辨率²**增长，而不是随"隐空间"这个
身份增长。当前 4× 压缩的 VAE 产出的 latent 只有 **8×8**，可用频点对仅 56 个 ——
**比像素 32×32 的 342 个还少 6 倍**。也就是说：

> 如果直接沿用"SD 的 f8 下采样 + 32×32 输入"这个组合，隐空间的容量反而**退化**了。

因此本仓库默认把原生 VAE 配成 **2 级下采样（4× 压缩）→ latent 16×16×4**：
可用 280 对，是像素空间的 **1.85×**，同时保留 `-d3`（8×8）配置用于对照实验。
真正宽裕的配置是 32×32 以上的 latent（≥512×512 图像 + f8 下采样），
届时可用预算达像素空间的 8.8×–74×。

## 5. 验收标准

`eval_ldm_pipeline.py` 会输出四个关键量：

| 量 | 含义 | 目标 |
|---|---|---|
| `vae_roundtrip_psnr` | 无嵌入时的往返上限 | **≥ 28 dB**（像素空间是 18.8 dB） |
| `psnr`（strength=1.0） | 载密图质量 | 向 `vae_roundtrip_psnr` 靠近，差距 ≤ 3 dB |
| `dec_clean` / `dec_jpeg50` | 比特准确率 | ≥ 0.95 |
| `capacity.avail_pairs_all_channels` | 频点预算 | **必须 ≥ 像素空间的 342**，否则迁移无意义 |

**判断标准**：如果 `vae_roundtrip_psnr` 仍 < 22 dB，说明瓶颈在 VAE 本身
（自带 native VAE 规模有限），此时应换 SD VAE 或加大 native VAE，
而不是继续调扩散模型。

## 6. 当前状态（2026-09-11）

已完成：

- ✅ `krd/vae.py`（native + SD 双后端）、`krd/latent.py`（缓存/容量核算）
- ✅ `scripts/train_vae.py`（含断点续训，结束时报告 VAE 往返 PSNR）
- ✅ `train_ddpm.py` 支持 `--vae-backend`，自动编码并缓存潜变量
- ✅ `train_decoder.py` 隐空间训练：失真在像素空间施加、再编码回隐空间
- ✅ `eval_robustness.py` 走 `StegoIO` 适配层；`scripts/eval_ldm_pipeline.py` 端到端验收
- ✅ `tests/ldm_smoke_test.py` 全链路冒烟通过（含像素空间直通回归）
- ✅ `pattern.py` 环带 `r_min` 改为随分辨率自适应（原先固定 3，导致小 latent 无频点可用）

进行中：

- ⏳ native VAE 训练：`-d3`（8×8 latent）已到 **epoch1 重建 20.01 dB**，
  已超过像素空间 18.81 dB 的上限；`-d2`（16×16 latent）刚起步
- ⏳ 隐空间 DDPM / 隐空间解码器待 VAE 收敛后训练（本机 GPU 4GB，需串行安排）

尚未完成：

- ⏳ 隐空间工作点下的 P1/P2/P3 重测（应等 `vae_roundtrip_psnr` 稳定后再做）
- ⚠️ 像素空间的所有历史结论（P1/P2/P3 结果表）**仍基于 9.5–10 dB 工作点**，
  在隐空间工作点稳定前不要引用为最终数字

## 7. 预期时间（本机 GTX 1650 4GB）

| 步骤 | 用时 |
|---|---|
| native VAE（16×16 latent, 20 epoch） | ≈ 1.5–2 h |
| 隐空间 DDPM（200 epoch, 4×16×16） | ≈ 1–1.5 h（latent 比像素小 4 倍，比 32²×3 快） |
| 隐空间解码器（1000 step） | ≈ 40 min |
| 端到端验收 | ≈ 5 min |

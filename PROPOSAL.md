# KRD-Steg：密钥条件化扩散轨迹隐写 —— 论文方案（对标 CCF-A）

> Key-conditioned Reverse-Diffusion Steganography
> 面向 "扩散模型 + 信息隐藏" 赛道的 CCF-A 论文孵化项目。本文档包含：文献综述与核验、
> 设想可行性分析、赛道性价比评估、方法方案、实验设计与代码现状。

---

## 1. 赛道综述（CCF-A 扩散隐写 / 扩散水印）

以下论文均在本轮调研中通过 arXiv / OpenReview / 官方会议列表核验（标注"未核验"的除外）。

### 1.1 扩散模型图像隐写（藏图 / 藏比特）

| 工作 | 会议 | 机制摘要 | 密钥 | 容量 | 备注 |
|---|---|---|---|---|---|
| **CRoSS** (NeurIPS 2023) | [arXiv:2305.16936](https://arxiv.org/abs/2305.16936), [代码](https://github.com/vvictoryuki/cross) | 利用预训练扩散模型：先对秘密图加噪、再以"伪装条件"反向生成载密图；复原时走反过程。开创性工作，被引 200+ | 无显式密钥（条件即秘密） | 一整张秘密图 | 对部分扰动鲁棒；但载密图直接由秘密条件驱动，安全性有争议 |
| **MDDM** (ICML 2025) | [ICML 2025 官方列表](https://icml.cc/virtual/2025/papers.html) | Message-Driven 生成式图像隐写（基于扩散模型） | — | 比特/消息 | 最新竞品，投稿前必须精读并对标 |
| **LDStega** (TIFS, [OpenReview](https://openreview.net/forum?id=kEqGgMgIlu)) | IEEE TIFS（CCF-A 期刊） | 把秘密数据编入 Latent Diffusion 的隐空间，可控、鲁棒 | — | 数据级 | 期刊侧竞品 |
| **Training-Free Robust Generative Steganography** ([ACM DL](https://dl.acm.org/doi/10.1145/3802927.3802947)) | 疑似 ACM MM 2025 | 用**确定性逆扩散重建初始隐噪声**来提取——与本项目复原机制同源，需重点对标 | — | — | 未核验细节 |
| **WaDiff** (ECCV 2024) | ECCV | 水印条件化扩散模型，用于 IP 保护 | — | — | 扩散 + 隐藏混合路线 |
| Diffusion-Based Hierarchical Image Steganography | [arXiv:2405.11523](https://arxiv.org/html/2405.11523v1) | 分层扩散隐写 | — | — | arXiv |

### 1.2 扩散模型水印（鲁棒隐形水印，比特载荷）

| 工作 | 会议 | 机制摘要 | 密钥 | 容量 |
|---|---|---|---|---|
| **Tree-Ring Watermarks** (NeurIPS 2023) | [arXiv:2305.20030](https://arxiv.org/abs/2305.20030), [代码](https://github.com/YuxinWenRick/tree-ring-watermark) | 在**初始噪声的傅里叶空间**嵌入环形图案，影响整个生成过程；检测时**反演扩散过程**恢复噪声向量再做匹配 | 图案即指纹（非用户密钥参数化） | 0-bit / 指纹 |
| **ZoDiac** = Attack-Resilient Image Watermarking Using Stable Diffusion (NeurIPS 2024) | [arXiv:2401.04247](https://arxiv.org/html/2401.04247v1), [NeurIPS 页面](https://neurips.cc/virtual/2024/poster/94294) | 在可训练隐空间注入水印，检测端解噪统计检验 | 多密钥多比特 | 多比特 |
| **EditGuard** (CVPR 2024) | [arXiv:2312.08883](https://arxiv.org/abs/2312.08883), [代码](https://github.com/xuanyuzhang21/EditGuard) | 统一 VAE 框架：同时做篡改定位 + 版权水印 | — | 双水印 |
| **Watermark Anything (WAM)** (ICLR 2025) | [arXiv:2411.07231](https://arxiv.org/abs/2411.07231), [代码](https://github.com/facebookresearch/watermark-anything) | 后处理嵌入/提取网络，可**局部化**水印与分割 | 未强调 | 32 bit |
| Stable Signature (ICCV 2023) / RoSteALS (NeurIPS 2023) | LDM 隐空间水印两条经典路线（本次未逐一重核验，ID 以官方为准） | 生成端隐空间调制 / 秘密密钥隐空间嵌入 | 有 | 低比特 |

### 1.3 经典 / 密钥化深度隐写（对照系）

| 工作 | 会议 | 要点 |
|---|---|---|
| **HiNet** (ICCV 2021) | [Semantic Scholar](https://www.semanticscholar.org/paper/6bb33b35a029122ff65b4e9d5f5732cffb9b67d5) | 可逆网络藏图，前向隐藏/逆向恢复 |
| **DeepMIH** (ICCV 2023, TPAMI 扩展) | [TPAMI](https://www.computer.org/csdl/journal/tp/2023/01/09676416/1A3djKkt0BO), [代码](https://github.com/TomTomTommi/DeepMIH) | 多图隐藏 + **密钥引导隐藏/恢复**（只有正确密钥能完美恢复）——与"密钥输入触发复原"最接近的经典工作 |
| **RIIS** (CVPR 2022) | [CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Xu_Robust_Invertible_Image_Steganography_CVPR_2022_paper.html) | 归一化流 + 失真仿真训练 → 鲁棒可逆隐写 |
| PRIS (AAAI 2022)、HiDRNet (ACM MM 2024) | — | 可逆网络路线的持续演进 |

### 1.4 关键结论（Gap 分析）

现有工作对三件事的组合**尚无人同时占据**：

- **(a) 隐藏端步数自由**：CRoSS / Tree-Ring / ZoDiac 的生成或反演步数基本固定；
- **(b) 密钥门控 + 闭式公式驱动的固定多步复原**：Tree-Ring 用反演但图案不由用户密钥参数化、
  检测器是手工匹配；DeepMIH 有密钥但无扩散轨迹；ZoDiac 有密钥但检测是统计检验而非"公式多步 + 学习解码"；
- **(c) 失真感知训练**作用在**扩散轨迹复原**任务上（可微 JPEG + 噪声/模糊/缩放 + 扩散再生攻击）。

**风险提示**：MDDM (ICML 2025) 与 Training-Free Robust Generative Steganography (MM 2025?)
与本项目机制最近，动笔前必须精读、复现并明确差异化，否则novelty 会被压。

---

## 2. 用户设想的可行性分析

| 设想 | 可行性 | 落地方式（本项目） |
|---|---|---|
| 用"简单扩散模型（U-Net DDPM）" | ✅ 直接可行 | 像素空间 DDPM（U-Net 主干），CIFAR-10 32×32 原型即可全流程跑通；正式版换 LDM 隐空间 |
| 隐写过程不指定步数 | ✅ 天然满足 | 隐藏 = DDIM 反演（S_hide 任选）+ 注入 + 采样；S_hide 与质量/时间自由权衡 |
| "通过一个公式实现指定多步复原" | ✅ 数学上严格成立 | 复原 = **闭式 DDIM 更新式**迭代 S_rec 步（固定），每步 `x_{t'} = √ᾱ_{t'}·x̂₀ + √(1-ᾱ_{t'})·ε_θ` |
| 输入密钥 → 网络进行复原 | ✅ 可实现且可量化 | 密钥 SHA256 → 确定性选择频谱频点/相位/幅度 → 特征提取与解码都被密钥门控；错密钥≈50% 比特准确率（已在冒烟测试中验证 corr(true)≈1.0 vs corr(wrong)≈0.02） |
| 创新点：鲁棒性 | ✅ 主推 | 失真感知训练（可微 JPEG 直通估计 + 噪声/模糊/缩放/亮度）+ 扩散再生攻击评测 |
| 创新点：隐秘性 | ✅ 辅推 | 载密图由扩散模型再生成（非贴补丁），天然规避空域隐写分析；可加 FID/隐写分析 AUC 实验 |

**结论：设想整体可行，且恰好落在"扩散水印"与"扩散隐写"的交叉空白上。**
需要注意的两点：(1) 纯像素 DDPM 在 32×32 的往返重建质量有限（PSNR ~25–30dB），
正式论文需换 Stable Diffusion/LDM 隐空间路线把隐秘性提上去；(2) 容量在像素空间较小（本原型 16 bit），
隐空间 64×64×4 时可扩到数百比特。

---

## 3. 赛道性价比评估

**性价比：高（推荐进入），但有明确的前提条件。**

利：
- **发表热度高且仍在持续**：NeurIPS 2023 (CRoSS) → ECCV 2024 (WaDiff) → CVPR 2024 (EditGuard)
  → ICLR 2025 (WAM) → ICML 2025 (MDDM) → NeurIPS 2025 (LD-RoViS)，年年有 CCF-A；TIFS 等期刊也收。
- **算力门槛可控**：主流做法复用冻结的预训练扩散模型，只训轻量组件（本项目解码器仅 ~0.5M 参数）；
  原型在单卡消费级 GPU 上即可完整复现。
- **评测协议标准化**：PSNR/SSIM/LPIPS + 比特准确率 + JPEG/噪声/模糊/再生攻击 + 隐写分析 AUC，
  数据集公开（CIFAR/ImageNet/COCO/DIV2K），无私有数据壁垒。

弊（必须正视）：
- **拥挤**：2023–2025 已有 10+ 篇强基线，审稿人见多识广，"把 X 换成扩散"式增量创新会被拒；
- **基线复现负担**：需要跑通 CRoSS / Tree-Ring / ZoDiac / WAM 的对比表；
- **新颖性窗口收窄**：与本项目机制最近的两篇 2025 竞品必须精读并差异化。

差异化护栏（写论文时的三板斧）：密钥门控安全性（错密钥/密钥空间分析）、步数不对称设计
（自由隐藏 / 固定复原）+ 步数鲁棒性消融、失真感知轨迹复原（含扩散再生攻击）。

---

## 4. 方案：KRD-Steg

### 4.1 方法概述

```
隐藏端（步数自由）:
  cover x₀ --DDIM反演(S_hide 步, 任选)--> x_T
  x_T --密钥k选频点, 注入ECC比特图案--> x_T'    [F[k_i] = m_i·e^{jφ_i}·s_i·scale]
  x_T' --DDIM采样(S_hide 步)--> stego x̃₀

复原端（步数固定, 公式驱动, 密钥门控）:
  stego --失真信道(攻击/传输)--> y
  y --DDIM反演(S_rec 步, 固定)--> x_T          [每步: x_{t'} = √ᾱ_{t'}·x̂₀ + √(1-ᾱ_{t'})·ε_θ]
  x_T --密钥k选频点--> 特征 v = [Re, Im]/norm
  v --RingDecoder(MLP)--> ECC logits --多数合并--> 比特 m̂
```

- **密钥门控**：key --SHA256--> 确定性随机源，决定 (环带, 频点置换, 相位 φ, 幅度 m)。
  错密钥选到无关频点 → 特征与模板去相关 → 解码≈随机。
- **密钥去旋转特征（关键设计，实测教训）**：比特以 F[k]=m·e^{jφ}·s 写入，φ 密钥随机；
  若解码器直接吃原始 [Re,Im]，"特征维→符号"的映射随密钥旋转，密钥盲的解码器不可学习
  （3000 步损失钉死 ln2）。提取特征时按密钥相位去旋 e^{-jφ}·F[k]，同相分量符号即比特、
  与密钥无关——修复后 150 步即达 clean=1.000 / jpeg50=0.998。
- **训练策略**：冻结 DDPM，只训 RingDecoder；训练时随机密钥/比特/强度 + 随机失真注入，
  使解码器学会在"失真 + 反演误差"下读出比特（类似 RIIS 的失真仿真思想迁移到轨迹域）。
- **步数灵活性**：隐藏端 S_hide 任意（质量/时间权衡）；复原端 S_rec 固定并可与协议一起公开；
  可选 `--rec-jitter` 训练对 S_rec 抖动的鲁棒性。

### 4.2 论文级创新点（拟主张）

1. **非对称步数设计**：隐藏步数自由 + 复原步数固定且由闭式 DDIM 公式驱动（首个明确形式化并消融该性质的工作）。
2. **密钥门控频谱图案 + 学习式密钥条件解码**：把 Tree-Ring 的"固定指纹"升级为"用户密钥参数化的多比特信道"，
   并给出错密钥≈50%、密钥空间 2^256 的安全性分析。
3. **失真感知轨迹复原**：可微 JPEG/噪声/模糊/缩放 + 扩散再生攻击下的鲁棒性训练与统一评测协议。

### 4.3 实验设计

- **数据集**：原型 CIFAR-10（32²）；正式版 ImageNet/COCO 子集 + Stable Diffusion 隐空间（512²）。
- **指标**：隐秘性 PSNR/SSIM/LPIPS + stego 分布 FID + 隐写分析 AUC（SRNet/XuNet）；
  鲁棒性比特准确率（JPEG q∈{30..90}、σ∈{0.05,0.1}、blur、resize、brightness、crop、
  **扩散再生攻击**）；密钥安全性（错密钥 acc、密钥个数与容量权衡）。
- **基线**：CRoSS、Tree-Ring、ZoDiac、WAM、DeepMIH、RIIS（多数有官方代码）。
- **消融**：S_hide∈{20,50,100}×S_rec∈{10,25,50,100} 网格；strength 扫描（容量-质量曲线）；
  ECC 重复次数；失真组合开关；错密钥热图。
- **目标 venue 与节奏**：先以 workshop/短文验证管线 → 主会目标 AAAI / ACM MM / CVPR（一年周期），
  或 TIFS/TIP 期刊线。

### 4.4 风险与备选路线

| 风险 | 应对 |
|---|---|
| MDDM/MM25 已覆盖核心点 | 精读后转向：步数不对称理论分析 + 再生攻击鲁棒性 + 密钥安全证明 |
| 像素空间隐秘性不足 | 迁移到 SD/LDM 隐空间（代码已模块化，只需换 schedule/主干） |
| 解码器在强失真下失效 | 引入软判决 + BCH/LDPC、或把解码器扩成 U-Net 直接从中间轨迹读比特 |
| 审稿质疑"步数自由"意义 | 给出 S_hide–质量–时间 Pareto 曲线 + 与固定步数基线的公平对比 |

---

## 5. 代码现状（本仓库）

> **当前主战场是 LDM 隐空间路线**（自训 VAE + 隐空间 DDPM）；像素空间原型保留作对照。
> 迭代史、当前结果与判定点见 **ITERATION_LOG.md**（主日志）；v2 评测口径修正见 README。

```
krd/
  schedule.py      DDPM 调度 + DDIM 采样/反演（"复原公式"）
  unet.py          U-Net（像素/隐空间共用，in_ch 按空间自适应）
  vae.py           VAE（native f=2 / SDVAE / Wrapper）—— LDM 迁移核心
  latent.py        隐空间容量报告（fit_capacity 相关工具）
  pattern.py       密钥(+nonce)→频点/相位/幅度；加性/覆盖式注入；去旋转环带特征；ECC；校验比特
  decoders.py      RingDecoder（MLP 解码头）
  stego.py         TrajStego: hide(inject_at 轨迹任意时刻注入) / recover / 再生攻击
  security.py      错密钥BER分布 / matched-filter 校验 / 槽位定位AUC / 盲检测AUC / 残差一致性
  distortions.py   可微 JPEG + 真实几何攻击（crop_fill/rotate/zoom/translate）+ 调度坐标加噪
  perceptual.py    LPIPS（可选依赖）
  metrics.py       PSNR / SSIM / 比特准确率
scripts/
  train_vae.py / train_ddpm.py（像素与隐空间）/ train_decoder.py（含 fit_capacity 容量自适应）
  run_pipeline.py  一键流水线（--ldm 隐空间模式，断点续跑）
  eval_setup.py / eval_common.py   统一评测入口（像素↔隐空间桥接、nonce=H(key‖counter) 盲协议）
  eval_ldm_pipeline.py / eval_strength_sweep.py   端到端工作点 / strength 扫描
  eval_robustness / eval_key_security / eval_steps_grid / eval_step_mismatch / eval_regen / plot_frontier
  verify_results.py 独立复核脚本（evaluation-only，全关键数字重验）
  diag_*.py        机制诊断探针（往返上限/注入 A/B/t*/载荷）
tests/             smoke_test + inject_at_test 回归
```

### 当前状态（2026-09-19）

- **像素空间判定到头**：300 epoch 重训证实往返上限 18.8 dB（loss 已饱和）→ 弃用；
- **LDM 迁移完成**：VAE 往返 51.5 dB，隐空间 DDPM 无嵌入往返 38 dB@S=150；
  当前工作点 26 dB / dec clean 0.916（strength=0.3，bpb=4 + inject_at=0.35 解码器）；
- **独立复核通过**：全部关键声明数字复现（端到端逐位一致）；发现 FAR@τ=12 不稳健、
  n=8 抽样方差 ±0.05 两个稳健性问题（正式评测须 n≥64、引用 FAR@τ=8）；
- **卡点**：质量-可解码互斥（35dB→~0.7，0.9+→26dB）。杠杆次序：载荷 16→8bit（进行中）
  → 解码器收敛（mf 全面优于 MLP，loss 未到底）→ 软判决 ECC；
- **创新点自评与选刊**：见 ITERATION_LOG.md §5（TIFS 主线 + IH&MMSec/WIFS 短文）。

### 下一步（按优先级）

1. **8bit 载荷重训 + 容量曲线家族**（进行中）：4/8/16/32 bit 的容量–鲁棒性曲线，
   替代单点容量主张（回应"容量太小"）；
2. **解码器收敛**：加长训练/简化输出头，目标至少追平 matched-filter（s=0.3: 0.984 vs 0.916）；
3. **大样本正式评测**（n≥64）：robustness/key_security/steps_grid/regen/frontier 全套主表；
4. **基线同环境对比**：Tree-Ring / ZoDiac / MDDM 复现（回应 novelty 质疑，投稿前必须）；
5. **规模升级**：256²/512²（换 ImageNet/COCO 子集）+ FID/LPIPS（先 `pip install lpips`）；
6. **隐写分析安全**：SRNet/XuNet AUC 补 §E 盲检测。

# 训练配置与加速指南（本机 GTX 1650 4GB / 远程 RTX 2080 22GB）

本文件记录**实测**的吞吐数据、可用的加速项、以及已验证不可用的项。
所有数字都可用 `scripts/bench_local.py` / `scripts/bench_options.py` 复现。

---

## 1. 本机（GTX 1650 4GB, torch 2.5.0+cu118）实测

`python scripts/bench_options.py --base 64 --batches 128,256,512`

| batch | channels_last | cudnn.benchmark | ms/step | img/s | 峰值显存 | 300 epoch 外推 |
|---|---|---|---|---|---|---|
| **128** | False | False | **219.5** | **583** | 1.60 GiB | **7.1 h** |
| 256 | False | False | 477.1 | 537 | 3.12 GiB | 7.8 h |
| 512 | False | False | 5942.3 | 86 | 6.16 GiB | 48.4 h |
| 128 | False | True | 223.0 | 574 | 1.60 GiB | 7.3 h |
| 128 | True | False | 359.2 | 356 | 1.90 GiB | 11.7 h |
| 256 | True | False | 1822.0 | 141 | 3.72 GiB | 29.7 h |

**三条结论：**

1. **batch 不是越大越好**：128 → 256 吞吐略降，512 直接掉到 1/7
   （cuDNN 在该尺寸选到了极差的卷积算法）。**上大卡必须先扫一遍 batch**，
   不要凭经验设成 1024。
2. **`--channels-last` 在本模型上明显更慢**（11.7 h vs 7.1 h），不要开。
3. **`--cudnn-benchmark` 基本无差别**（±2%），可开可不开（确定性会略变）。

交叉验证：历史那次 60 epoch 用了 111.8 min，即 390 step × 286 ms/step；
本次实测 batch=128 为 219 ms/step（含 AdamW+EMA），同一量级 → 说明本机速度没有退化。

**推理时间**：50 步 DDIM 反演 ≈ 53.8 ms/图/趟（batch 16），评测阶段 64 张图约 10 分钟。

---

## 2. 混合精度（AMP）：**当前实现不可用，默认关闭**

- 在 `--amp on` 下，本 U-Net 的 **up-path ResBlock 会溢出成 inf/nan**，
  训练 loss 立刻变 nan（已用 `scripts/_dbg_amp.py` 逐层定位到 `up0.ResBlock`）；
- 把 ResBlock 整体强制 fp32（`ResBlock.forward_fp32` + `model.fp32_resblocks=True`）
  **仍然无法消除**（autocast 只转激活不转权重，已显式 cast 权重后依旧 nan）；
- 因此 `--amp` 默认 `off`，`--amp on` 仅保留作实验，并在启动时打印警告。

**这也是本次新增功能的唯一"负结果"**：在 2080 上想再提速，请优先用
"扫 batch + 增大 num_workers"，而不是指望 fp16。

---

## 3. 远程 RTX 2080 22GB 推荐配置

```bash
# 0) 拉取本次提交（含断点续训 / 大 batch / 环境自适应）
git pull

# 1) 先扫 batch，别直接开跑（约 5 分钟）
python scripts/bench_options.py --base 64 --batches 256,512,1024 --steps 10
#    看 img/s 最高的那一档；若 512/1024 出现异常慢，就是本机 4GB 上同样的
#    "cuDNN 算法退化"现象，往下退一档即可。

# 2) 加长训练到 300 epoch（自动从已有 60 epoch 断点续训）
python scripts/run_pipeline.py --stages ddpm \
    --ddpm-epochs 300 \
    --ddpm-batch <上一步最优> \
    --ddpm-save-every 10 \
    --cudnn-benchmark

# 3) 训练完再跑解码器 + 全部评测
python scripts/run_pipeline.py --stages decoder robust p1 p2 p3 --force
```

要点：

- **22GB 显存足够把 batch 开到 512–1024**（本机 4GB 在 batch=512 只吃 6.16 GiB，
  且那是"算法退化"而非显存问题），但仍**必须实测**，见 §1 第 1 条；
- 数据加载在远程 Linux 上应显式给足 worker（本机沙箱里多进程被禁，自动回退 0）：
  `train_ddpm.py --num-workers 8`；
- `--ddpm-save-every 10` 会写 `checkpoints/ddpm_cifar.last.pt`，
  中断后**直接重跑同一命令**即可续训（会打印 `[resume] 从 ... 恢复`）；
- 本次修复把 nonce 协议、几何攻击、再生攻击代价都改了口径，
  远程跑出来的结果请用 `scripts/results/README.md` 里的说明与旧结果区分。

---

## 4. 关于"保持本次修改"的同步方式

本次改动已提交到 `master`，远程机器 **`git pull` 即可**。
需要注意的兼容点：

| 改动 | 对远程的影响 |
|---|---|
| nonce 协议 `H(key‖counter)` | 评测结果表多一列说明；旧 checkpoint 仍可复用（协议在评测侧，与权重无关） |
| `crop` 改为真裁剪 + 新增几何攻击 | 鲁棒性表会多出 `crop4px/rotate/translate/zoom/cropresize` 行 |
| 再生攻击代价改为 vs cover | `regen.md` 列结构变化，旧表不可直接对比 |
| `train_ddpm.py` 支持续训 | 新的 `.last.pt` 断点文件；旧 `.pt` 仍可直接被 `train_decoder.py` 读取 |
| `UNet.forward(..., fp32_resblocks)` | 默认 False，**推理路径与旧行为完全一致**（冒烟测试已验证） |

---

## 5. 复现命令速查

```bash
# 本机/远程吞吐基准（单点）
python scripts/bench_local.py --batch 128 --decoder-batch 16

# batch × amp 扫描
python scripts/bench_local.py --probe-batches 128,256,512 --amp both

# batch × channels_last × cudnn 扫描（推荐用这个决定远程配置）
python scripts/bench_options.py --base 64 --batches 256,512,1024

# AMP 溢出定位（若将来换 U-Net 想重开混合精度）
python scripts/_dbg_amp.py 64 fp32rb
```

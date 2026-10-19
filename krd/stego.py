"""KRD-Steg 主流水线：隐藏（步数自由）与复原（固定步数、公式驱动、密钥门控）。

hide:    cover --DDIM反演(S_hide 步, 自由)--> x_T --注入密钥图案--> x_T' --DDIM采样--> stego
recover: stego --DDIM反演(S_rec 步, 固定)--> x_T --密钥选频点--> 环带特征 --MLP--> 比特 logits

安全设计：
  - 密钥 + nonce 共同决定频点/相位/幅度（nonce 逐图派生自 H(key||i)，自包含、不依赖 cover）；
  - 嵌入向量 = ECC(消息) + 密钥校验比特（matched-filter 可验证密钥，无需解码器）。
"""

import torch

from . import pattern
from .pattern import ecc_collapse, ecc_encode, inject_pattern, ring_features


class TrajStego:
    def __init__(self, model, schedule, n_bits: int = 16, ecc_reps: int = 3,
                 bins_per_bit: int = 2, res: int = 32, n_check_bits: int = 32,
                 inject_mode: str = "replace", r_max: int | None = None,
                 mag_profile: bool = False, sync: bool = False,
                 depth_sign_align: bool = False, invert_band: int = 15,
                 device="cpu"):
        self.model = model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.sched = schedule
        self.n_bits = n_bits
        self.ecc = ecc_reps
        self.bpb = bins_per_bit
        self.res = res
        self.r_max = r_max  # 环带外半径上限（None = 扩展到 Nyquist）；JPEG 鲁棒频段实验用
        self.n_check_bits = int(n_check_bits)
        # 注入模式：replace = 历史行为（覆盖系数，破坏性）；add = 加性（保留系数结构）
        self.inject_mode = inject_mode
        # mag_profile：按密钥幅度轮廓分配逐频点幅度（该轮廓同时是接收端的同步模板）
        self.mag_profile = bool(mag_profile)
        # sync：接收端做几何对齐搜索（需要 mag_profile=True 才有可靠模板）
        self.sync = bool(sync)
        # depth_sign_align：**默认关闭**。原设想是"浅注入会反号，翻转后能相长"
        # （轨迹 Rake），但 A/B 实测把这个设想否掉了，见下表（res=32, 4bit/N=64,
        # S=150, n_inject=8, inject_at=0.35, n=16，同批次同能量）：
        #     align=OFF: clean d=+1.82 / jpeg50 0.922 (d=1.139) / jpeg75 1.000
        #     align=ON : clean d=+1.05 / jpeg50 0.719 (d=0.634) / jpeg75 0.812
        # 即翻转最浅的 2 个注入点会让总偏转掉 ≈43%（-4.8 dB），说明
        # **8 个注入点本来就是同号相干的**，翻号等于把 2 份 Contribution 变成相消
        # （8→4 份 ⇒ 理论上 -6 dB，实测 -4.3~-5.1 dB，吻合）。
        # 之所以"单深度扫描"在 inject_at 很小时测到负偏转，是因为那里 k 很小、
        # **采样步数不足**导致成像反相，并不是注入下标的传递函数——
        # `timestep_seq(k)` 的刻度随 k 变化，下标 i 不能跨 k 比较。
        # 结论：保留开关与测试（记录机制），但默认关闭。
        self.depth_sign_align = bool(depth_sign_align)
        self.invert_band = int(invert_band)
        self.msg_embed_bits = n_bits * ecc_reps            # ECC 展开后的消息槽位
        self.total_embed_bits = self.msg_embed_bits + self.n_check_bits
        self.n_pairs = self.total_embed_bits * bins_per_bit
        self.device = device
        self._params_cache: dict[tuple[str, str], dict] = {}

    # ---------- 密钥参数（缓存, 按 (key, nonce)） ----------

    def params_for(self, key: str, nonce: str = "") -> dict:
        ck = (key, nonce)
        if ck not in self._params_cache:
            p = pattern.key_params(key, self.n_pairs, self.res, nonce=nonce,
                                   r_max=self.r_max)
            self._params_cache[ck] = {
                "bins": p["bins"].to(self.device),
                "phases": p["phases"].to(self.device),
                "base_mag": p["base_mag"].to(self.device),
                "res": p["res"], "r": p["r"], "key": key, "nonce": nonce,
            }
        return self._params_cache[ck]

    # ---------- 比特组装 ----------

    def full_bits(self, bits: torch.Tensor, key: str, nonce: str = "") -> torch.Tensor:
        """(n_bits,) 消息 -> (total_embed_bits,) 嵌入向量 = ECC(消息) + 密钥校验比特。"""
        b = ecc_encode(bits, self.ecc)
        if self.n_check_bits > 0:
            cb = pattern.check_bits(key, nonce, self.n_check_bits).to(b.device)
            b = torch.cat([b, cb])
        return b

    @staticmethod
    def resolve_nonces(nonces, batch: int, keys=None) -> list[str]:
        """nonces=None 时的默认协议：逐图 nonce = H(key || index)（自包含）。

        早期版本用 H(cover) 派生 nonce，那需要把 nonce 通过侧信道（PNG 元数据）
        传给复原端，方案并非盲提取。现在 nonce 只由密钥与图序号决定：
        复原端只需 key + stego，counter 作为公开参数随图传输即可。
        """
        if nonces is not None:
            assert len(nonces) == batch, f"nonces {len(nonces)} != batch {batch}"
            return list(nonces)
        if keys is None:
            return [""] * batch
        from .utils import derive_nonce_from_key
        return [derive_nonce_from_key(str(keys[i]), i) for i in range(batch)]

    # ---------- 隐藏端：步数可自由选择 ----------

    @torch.no_grad()
    def hide(self, cover: torch.Tensor, bits: torch.Tensor, keys,
             hide_steps: int = 50, strength=1.0, nonces=None,
             inject_at: float | None = None, n_inject: int = 1,
             edit_mask: torch.Tensor | None = None) -> torch.Tensor:
        """cover (B,C,H,W), bits (B,n_bits)∈{0,1}, keys: list[str] len B。

        edit_mask：**局部化扩散编辑**（(H,W) bool，或 None = 全图）。
          只保留被选中的潜变量块经过"反演→注入→重采样"的结果，其余位置回填原
          潜变量（恒等）。采样误差能量 ∝ 编辑比例 f，因此地板可以从整幅重采样的
          30.4 dB 抬到 ~f 分之一（+10log10(1/f)）；但接收到的相干信号也 ∝ f，
          所以注入幅度要按 1/f 放大才能维持同样的判决性——两者的乘积决定最优 f：
              E_tot(f) = f·E_rt + a²/f  ⇒  min = 2√(E_rt·a²)
          即"局部化"能比整幅重采样高 10log10(E_rt / (2√(E_rt a²))) 的 dB。
          块集应由密钥/nonce 派生（见 pattern.edit_mask_for），否则攻击者可
          直接定位被编辑区域。

        nonces: list[str] len B；None 时逐图派生 nonce=H(key||i)（自包含协议）。
        strength: float 或 (B,) tensor，控制图案强度（隐秘性/容量的折中）。
        inject_at: 注入时刻占反演轨迹的比例 ∈ (0,1]。
          - None / 1.0 ：反演到最末端 x_T 再注入（历史行为）
          - 0.25–0.5  ：反演到该时刻注入、再从该时刻采样回 x_0。
            理由（LDM_WORKPOINT_DIAGNOSIS.md §6/§7）：扩散采样每步都经学到的 ε_θ
            投影回数据流形，会把"离流形"的小扰动抹掉。
        n_inject: **多步注入次数**（默认 1 = 历史行为）。
          在轨迹的**下坡段**选 n 个时刻，每处注入一次相同的图案。
          语义（重要）：调用方给出的 `strength` 被理解为**每次注入**的强度；
          若希望"总注入能量固定"，请传 `strength/n`（脚本层 `--n-inject` 即这么处理）。
          实测（latent 4×32×32, S=150, inject_at=0.35, 总能量守恒）：
            总能量 0.3：n=1 -> PSNR 29.35 / mf 0.906；n=8 -> PSNR 30.74 / mf 0.922
          即**准确率由总能量决定，分步只把 PSNR 抬上去**（峰值扰动更小），
          因此这是"零额外计算换 1–1.4 dB"的净收益，但**不会把前沿整体左移**。
          实现上不增加反演/采样次数（仍是 k 步采样）。
        """
        B = cover.shape[0]
        if not torch.is_tensor(strength):
            strength = torch.full((B,), float(strength), device=self.device)
        strength = strength.to(self.device).view(B)
        nonces = self.resolve_nonces(nonces, B, keys=keys)

        # 轨迹下坡段起点：k 步反演 -> 注入 -> k 步采样
        frac = 1.0 if inject_at is None else float(inject_at)
        frac = min(max(frac, 1e-3), 1.0)
        k = min(max(1, int(round(hide_steps * frac))), hide_steps)

        seq = self.sched.timestep_seq(k)
        n = max(1, int(n_inject))
        # 注入时刻（timestep_seq 的下标）：把 [0, len(seq)-2] 等分 n 份，取每份右端
        if n == 1:
            idxs = [len(seq) - 2]
        else:
            idxs = sorted({int(round((j + 1) * (len(seq) - 2) / n)) for j in range(n)})

        # 预取一次图案参数（避免每个注入点重复查缓存）
        params = [self.params_for(keys[i], nonces[i]) for i in range(B)]
        bits_full = [self.full_bits(bits[i], keys[i], nonces[i]) for i in range(B)]

        x = cover
        inj_set = set(idxs)
        # 沿升序把 cover 推到轨迹上：每到一个注入时刻就写一次图案
        #
        # **深度符号对齐（轨迹 Rake）**：注入点越浅，回流时会**反号**——实测
        # （scripts 里 diag，S=150, E=0.3，4bit/N=64）单深度注入在 k≤11 时
        # 读出偏转为负（k=4: −1.33，k=11: −1.01），k≈15 过零，k≥18 转正，
        # 峰值在 k≈52（+1.62）。也就是说 `n_inject>1` 时最浅的那 1~2 次注入
        # 原本是在**相消**而不是相加。翻转这些注入的符号，注入总能量/PSNR/
        # 计算量完全不变，只是把"相消"变成"相加"（实测偏转 +20% ≈ +2 dB）。
        # 接收端无感：读出仍是"槽位值的符号 = 比特"，只是信号更强。
        # 阈值 `invert_band` 以"距 cover 的 DDIM 反演步数"计（= 注入下标 i+1），
        # 是模型性质，换模型时用 diag 重标一次即可。
        for i in range(len(seq) - 1):
            t_cur = torch.full((B,), int(seq[i]), device=self.device, dtype=torch.long)
            t_next = torch.full((B,), int(seq[i + 1]), device=self.device, dtype=torch.long)
            x = self.sched.ddim_update(self.model, x, t_cur, t_next, eta=0.0)
            if i in inj_set:
                sgn = -1.0 if (self.depth_sign_align and (i + 1) <= self.invert_band) else 1.0
                x = torch.stack([
                    inject_pattern(x[j], bits_full[j], params[j],
                                   float(strength[j]) * sgn, mode=self.inject_mode,
                                   use_mag_profile=self.mag_profile)
                    for j in range(B)
                ])
        # 从最深的注入时刻采样回 x_0（同一段轨迹，步数不变）
        out = self.sched.ddim_sample(self.model, x, k)
        if edit_mask is not None:
            # 局部化：只有 mask 内的位置采用"扩散重采样"的结果，其余回填原潜变量
            # 支持 (H,W) 共享掩码、(B,H,W) 逐样本掩码，以及 (B,1,H,W) 已广播形态
            m = edit_mask.to(out.device)
            if m.dtype != torch.bool:
                m = m > 0.5
            if m.dim() == 2:
                m = m[None, None]
            elif m.dim() == 3:
                m = m[:, None]
            out = torch.where(m, out, cover)
        return out

    @torch.no_grad()
    def hide_localized(self, cover: torch.Tensor, bits: torch.Tensor, keys,
                       hide_steps: int = 50, strength=1.0, nonces=None,
                       inject_at: float | None = None, n_inject: int = 1,
                       edit_mask: torch.Tensor | None = None) -> torch.Tensor:
        """**局部化扩散编辑**：RePaint 式掩码条件采样（正确的局部编辑方式）。

        为什么不能用"最后再复合"（`hide(..., edit_mask=...)`）：解码器是空间耦合的，
        把 5%~30% 的潜变量单元换成"另一次采样的结果"会在掩码边界造成不连续，
        实测 f=0.05 时载密图只有 5.34 dB（见 results/localized.json）——那是
        块间不连续而非能量问题。

        正确做法（本函数）：在采样的**每一步**，把未编辑区域替换回 cover 的真实
        前向轨迹值 x_t^known（RePaint 的 replacement 技巧）。于是
          - 未编辑区在 t=0 与原 cover 逐位一致 → 不引入任何块间不连续；
          - 只有掩码内是"模型合成"的内容，且与边界条件相容；
          - 往返地板 ≈ VAE 往返（51 dB）与 f·E_rt 的合成，远高于整幅重采样。

        信号代价：接收端在**全局环带**上相干累加，只有掩码内的部分携带图案，
        因此信号 ∝ f；要保持判决性需把注入幅度按 1/f 放大（成本 a²/f），
        最优 f = √(a²/E_rt)，最优误差 2√(E_rt·a²)（见 docs 与 diag_localized.py）。
        """
        B = cover.shape[0]
        if not torch.is_tensor(strength):
            strength = torch.full((B,), float(strength), device=self.device)
        strength = strength.to(self.device).view(B)
        nonces = self.resolve_nonces(nonces, B, keys=keys)

        frac = 1.0 if inject_at is None else float(inject_at)
        frac = min(max(frac, 1e-3), 1.0)
        k = min(max(1, int(round(hide_steps * frac))), hide_steps)
        seq = self.sched.timestep_seq(k)
        n = max(1, int(n_inject))
        if n == 1:
            idxs = [len(seq) - 2]
        else:
            idxs = sorted({int(round((j + 1) * (len(seq) - 2) / n)) for j in range(n)})

        params = [self.params_for(keys[i], nonces[i]) for i in range(B)]
        bits_full = [self.full_bits(bits[i], keys[i], nonces[i]) for i in range(B)]

        # 1) 前向（反演）把 cover 推到轨迹上，并记录每一步的已知状态
        traj = [cover]
        x = cover
        total_steps = len(seq) - 1
        for i in range(total_steps):
            t_cur = torch.full((B,), int(seq[i]), device=self.device, dtype=torch.long)
            t_next = torch.full((B,), int(seq[i + 1]), device=self.device, dtype=torch.long)
            x = self.sched.ddim_update(self.model, x, t_cur, t_next, eta=0.0)
            if i in set(idxs):
                x = torch.stack([
                    inject_pattern(x[j], bits_full[j], params[j],
                                   float(strength[j]), mode=self.inject_mode,
                                   use_mag_profile=self.mag_profile)
                    for j in range(B)
                ])
            traj.append(x)                      # traj[i+1] 对应 t = seq[i+1]

        # 2) 反向采样，每一步把未编辑区域替换回真实轨迹（RePaint replacement）
        if edit_mask is not None:
            m = edit_mask.to(self.device)
            if m.dtype != torch.bool:
                m = m > 0.5
            if m.dim() == 2:
                m = m[None, None]
            elif m.dim() == 3:
                m = m[:, None]
        else:
            m = None
        x = traj[-1]
        for i in reversed(range(total_steps)):
            t_cur = torch.full((B,), int(seq[i + 1]), device=self.device, dtype=torch.long)
            t_next = torch.full((B,), int(seq[i]), device=self.device, dtype=torch.long)
            x = self.sched.ddim_update(self.model, x, t_cur, t_next, eta=0.0)
            if m is not None:
                x = torch.where(m, x, traj[i])
        return x


    @torch.no_grad()
    def invert_latents(self, images: torch.Tensor, steps: int) -> torch.Tensor:
        """images -> x_T（固定 steps 步 DDIM 反演）。

        反演与密钥无关：密钥安全性评测中可只反演一次，
        再用 features_from_latents 对大量错误密钥做廉价特征提取。
        """
        return self.sched.ddim_invert(self.model, images, steps)

    @torch.no_grad()
    def features_from_latents(self, x_T: torch.Tensor, keys, nonces) -> torch.Tensor:
        """x_T -> 密钥门控环带特征 (B, 2*n_pairs)。"""
        return torch.stack([
            ring_features(x_T[i], self.params_for(keys[i], nonces[i]))
            for i in range(x_T.shape[0])
        ])

    @torch.no_grad()
    def recover_features(self, stego: torch.Tensor, keys, rec_steps: int = 50,
                         nonces=None, sync: bool | None = None) -> torch.Tensor:
        """stego -> 密钥门控环带特征 (B, 2*n_pairs)。

        sync=True 时先做几何对齐搜索（krd/sync.py）：用密钥幅度轮廓定位
        (θ, σ)，再用相位斜坡定平移，然后在该对齐上取点。这解决的是
        "旋转/平移/缩放的载波失配"（§20.4 的短板），因为频谱域可解析补偿，
        不需要重过 VAE/UNet。
        """
        nonces = self.resolve_nonces(nonces, stego.shape[0], keys=keys)
        use_sync = self.sync if sync is None else bool(sync)
        x_T = self.invert_latents(stego, rec_steps)
        if not use_sync:
            return self.features_from_latents(x_T, keys, nonces)
        from . import sync as _sync
        feats = []
        for i in range(x_T.shape[0]):
            params = self.params_for(keys[i], nonces[i])
            F_ = torch.fft.fftshift(torch.fft.fft2(x_T[i], norm="ortho"), dim=(-2, -1))
            est = _sync.estimate_alignment(F_, params, self.res)
            re, im, _, _ = _sync.gather_aligned(F_, params, self.res, est)
            norm = torch.cat([re, im]).abs().median().clamp(min=1e-8)
            feats.append(torch.cat([re, im]) / norm)
        return torch.stack(feats).float()

    def collapse(self, logits_all: torch.Tensor) -> torch.Tensor:
        """(B, total_embed_bits) -> 消息部分 ECC 组内均值 -> (B, n_bits)。"""
        msg_logits = logits_all[..., :self.msg_embed_bits]
        return ecc_collapse(msg_logits, self.n_bits, self.ecc)

    @torch.no_grad()
    def recover(self, stego: torch.Tensor, keys, rec_steps: int = 50,
                decoder=None, nonces=None) -> torch.Tensor:
        """返回 n_bits 维消息 logits（已做 ECC 组内均值）。decoder 为训练好的 RingDecoder。"""
        if decoder is None:
            raise ValueError("recover 需要 decode 头（先 train_decoder.py 训练或加载）")
        feats = self.recover_features(stego, keys, rec_steps, nonces=nonces)
        return self.collapse(decoder(feats))

    # ---------- 攻击: 扩散再生（评测鲁棒性用） ----------

    @torch.no_grad()
    def regeneration_attack(self, stego: torch.Tensor, t_reg: int = 400,
                            steps: int = 50) -> torch.Tensor:
        """把 stego 重新加噪到 t_reg 再采样 —— 模拟“扩散重生成”攻击。"""
        noise = torch.randn_like(stego)
        t = torch.full((stego.shape[0],), int(t_reg), device=self.device, dtype=torch.long)
        x_t = self.sched.add_noise(stego, t, noise)
        return self.sched.ddim_sample(self.model, x_t, steps)

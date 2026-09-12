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
                 inject_mode: str = "replace", device="cpu"):
        self.model = model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.sched = schedule
        self.n_bits = n_bits
        self.ecc = ecc_reps
        self.bpb = bins_per_bit
        self.res = res
        self.n_check_bits = int(n_check_bits)
        # 注入模式：replace = 历史行为（覆盖系数，破坏性）；add = 加性（保留系数结构）
        self.inject_mode = inject_mode
        self.msg_embed_bits = n_bits * ecc_reps            # ECC 展开后的消息槽位
        self.total_embed_bits = self.msg_embed_bits + self.n_check_bits
        self.n_pairs = self.total_embed_bits * bins_per_bit
        self.device = device
        self._params_cache: dict[tuple[str, str], dict] = {}

    # ---------- 密钥参数（缓存, 按 (key, nonce)） ----------

    def params_for(self, key: str, nonce: str = "") -> dict:
        ck = (key, nonce)
        if ck not in self._params_cache:
            p = pattern.key_params(key, self.n_pairs, self.res, nonce=nonce)
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
             inject_at: float | None = None) -> torch.Tensor:
        """cover (B,C,H,W), bits (B,n_bits)∈{0,1}, keys: list[str] len B。

        nonces: list[str] len B；None 时逐图派生 nonce=H(key||i)（自包含协议）。
        strength: float 或 (B,) tensor，控制图案强度（隐秘性/容量的折中）。
        inject_at: 注入时刻占反演轨迹的比例 ∈ (0,1]。
          - None / 1.0 ：反演到最末端 x_T 再注入（历史行为）
          - 0.25–0.5  ：**推荐**。只反演 k=round(inject_at×S_hide) 步、在该时刻注入、
                        再从该时刻采样回 x_0（仍是 k 步采样）。
            理由（见 LDM_WORKPOINT_DIAGNOSIS.md §6/§7）：扩散采样每步都经学到的
            ε_θ 投影回数据流形，会把"离流形"的小扰动抹掉。末端注入后要经过全部
            S_hide 步（被抹 h 次），中点注入只剩一半步数，扰动存活率显著提高。
            实测（latent 4×32×32, S=150, strength=0.1）：
              inject_at=1.0 -> PSNR 9.39 dB, mf 0.453
              inject_at=0.5 -> PSNR 32.40 dB, mf 0.625
        """
        B = cover.shape[0]
        if not torch.is_tensor(strength):
            strength = torch.full((B,), float(strength), device=self.device)
        strength = strength.to(self.device).view(B)
        nonces = self.resolve_nonces(nonces, B, keys=keys)

        # 注入时刻：k 步反演 -> 注入 -> k 步采样（k = S_hide 时即历史行为）
        frac = 1.0 if inject_at is None else float(inject_at)
        frac = min(max(frac, 1e-3), 1.0)
        k = max(1, int(round(hide_steps * frac)))
        k = min(k, hide_steps)

        x_k = self.sched.ddim_invert(self.model, cover, k)
        x_k2 = []
        for i in range(B):
            b_full = self.full_bits(bits[i], keys[i], nonces[i])
            x_k2.append(inject_pattern(x_k[i], b_full, self.params_for(keys[i], nonces[i]),
                                       float(strength[i]), mode=self.inject_mode))
        x_k2 = torch.stack(x_k2)
        return self.sched.ddim_sample(self.model, x_k2, k)

    # ---------- 复原端：步数固定，公式驱动 ----------

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
                         nonces=None) -> torch.Tensor:
        nonces = self.resolve_nonces(nonces, stego.shape[0], keys=keys)
        x_T = self.invert_latents(stego, rec_steps)
        return self.features_from_latents(x_T, keys, nonces)

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

"""DDPM 噪声调度 + DDIM 确定性采样/反演。

本项目“复原公式”就是这里的闭式 DDIM 更新式：

  前向加噪 (q sampling):
      x_t = sqrt(abar_t) * x_0 + sqrt(1 - abar_t) * eps
  去噪预测:
      x0_hat = (x_t - sqrt(1 - abar_t) * eps_theta(x_t, t)) / sqrt(abar_t)
  DDIM 更新 (eta = 0 时为确定性 ODE 积分, 即本项目隐藏/复原共用的“公式”):
      x_{t'} = sqrt(abar_{t'}) * x0_hat + sqrt(1 - abar_{t'}) * eps_theta
      (t' < t 为去噪采样; t' > t 为 DDIM inversion, 即反演)

隐藏端步数可自由选择（inversion + 采样），复原端步数固定，两端共用同一公式。
"""

import math

import numpy as np
import torch


class Schedule:
    def __init__(self, timesteps: int = 1000, mode: str = "linear", device="cpu",
                 clip_denoised: bool = True):
        self.T = int(timesteps)
        # 是否把预测的 x0 截断到 [-1,1]。
        # 像素空间应开启；**隐空间必须关闭** —— 潜变量幅值可达 ±6，
        # 截断到 [-1,1] 会把往返 PSNR 从 ~33.6dB 打到 ~13.1dB（本机实测）。
        # 该值来自 checkpoint 的 args（load_stego 会写回），保证训练/评测口径一致。
        self.clip_denoised = bool(clip_denoised)
        if mode == "linear":
            betas = torch.linspace(1e-4, 0.02, self.T, dtype=torch.float64)
        elif mode == "cosine":
            steps = torch.arange(self.T + 1, dtype=torch.float64) / self.T
            abar = torch.cos((steps + 0.008) / 1.008 * math.pi / 2) ** 2
            abar = abar / abar[0]
            betas = (1 - abar[1:] / abar[:-1]).clamp(1e-5, 0.999)
        else:
            raise ValueError(f"unknown schedule mode: {mode}")

        alphas = 1.0 - betas
        abar = torch.cumprod(alphas, dim=0)
        self.betas = betas.to(device).float()
        self.alphas = alphas.to(device).float()
        self.abar = abar.to(device).float()
        self.device = device

    # ---------- 基础量 ----------

    def timestep_seq(self, steps: int) -> torch.Tensor:
        """把 [0, T-1] 均匀切成 steps+1 个整数时刻（严格递增）。

        隐藏/复原共用该序列：inversion 沿升序、采样沿降序,
        因此隐藏与复原的“公式轨迹”完全对齐。
        """
        assert 1 <= steps <= self.T - 1, f"steps must be in [1, {self.T - 1}]"
        seq = np.round(np.linspace(0, self.T - 1, steps + 1)).astype(np.int64)
        seq = np.unique(seq)
        return torch.from_numpy(seq).to(self.device)

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        abar = self.abar[t].view(-1, 1, 1, 1)
        return abar.sqrt() * x0 + (1 - abar).sqrt() * noise

    def predict_x0(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor,
                   clip_denoised: bool = True) -> torch.Tensor:
        abar = self.abar[t].view(-1, 1, 1, 1)
        x0 = (x_t - (1 - abar).sqrt() * eps) / abar.sqrt()
        if clip_denoised:
            x0 = x0.clamp(-1, 1)
        return x0

    # ---------- 核心: DDIM 更新公式 ----------

    def ddim_update(self, model, x: torch.Tensor, t_cur: torch.Tensor, t_next: torch.Tensor,
                    eta: float = 0.0, clip_denoised: bool | None = None) -> torch.Tensor:
        """x(t_cur) -> x(t_next)。t_next < t_cur 为采样，t_next > t_cur 为反演。

        clip_denoised=None 时使用实例设置 self.clip_denoised（隐空间应为 False）。
        """
        if clip_denoised is None:
            clip_denoised = self.clip_denoised
        eps = model(x, t_cur)
        x0 = self.predict_x0(x, t_cur, eps, clip_denoised)
        if bool((t_next == 0).all()):
            return x0  # 采样终点直接取截断后的 x0 预测（标准 DDIM 做法）
        abar_cur = self.abar[t_cur].view(-1, 1, 1, 1)
        abar_next = self.abar[t_next].view(-1, 1, 1, 1)
        x_next = abar_next.sqrt() * x0 + (1 - abar_next).sqrt() * eps
        if eta > 0 and bool((t_next < t_cur).all()):
            sigma = eta * (1 - abar_next).sqrt() / (1 - abar_cur).sqrt() \
                * (1 - abar_cur / abar_next).sqrt()
            x_next = x_next + sigma * torch.randn_like(x)
        return x_next

    # ---------- 高层接口 ----------

    @torch.no_grad()
    def ddim_invert(self, model, x0: torch.Tensor, steps: int,
                    clip_denoised: bool | None = None) -> torch.Tensor:
        """x_0 -> x_T（确定性反演, 步数 = steps, 由调用方自由选择 —— 隐藏端“不指定步数”）。"""
        seq = self.timestep_seq(steps)
        x = x0
        for i in range(len(seq) - 1):
            t_cur = torch.full((x.shape[0],), int(seq[i]), device=self.device, dtype=torch.long)
            t_next = torch.full((x.shape[0],), int(seq[i + 1]), device=self.device, dtype=torch.long)
            x = self.ddim_update(model, x, t_cur, t_next, eta=0.0, clip_denoised=clip_denoised)
        return x

    @torch.no_grad()
    def ddim_sample(self, model, x_T: torch.Tensor, steps: int,
                    eta: float = 0.0, clip_denoised: bool | None = None) -> torch.Tensor:
        """x_T -> x_0（确定性采样, 默认 eta=0）。"""
        seq = self.timestep_seq(steps)
        x = x_T
        for i in reversed(range(len(seq) - 1)):
            t_cur = torch.full((x.shape[0],), int(seq[i + 1]), device=self.device, dtype=torch.long)
            t_next = torch.full((x.shape[0],), int(seq[i]), device=self.device, dtype=torch.long)
            x = self.ddim_update(model, x, t_cur, t_next, eta=eta, clip_denoised=clip_denoised)
        return x

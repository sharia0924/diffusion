"""密钥条件下的比特解码头（MLP）。输入为环带特征，输出 ECC 扩展比特的 logits。

`mf_residual=True` 时额外加一条**匹配滤波器（matched filter）直线路径**：

    logits = feats @ Mᵀ + MLP(feats)

其中 M 是把每个槽位对应的 g 个"同相"坐标取平均的固定矩阵（即 mf 判决本身）。
配合"最后一层零初始化"，训练**从 mf 的性能出发**再往上走，而不是从随机初始化
慢慢爬。动机（实测）：mf 免训练接收在同等 PSNR 下明显优于训练解码器
（ITERATION_LOG §13.2），说明当前 MLP 欠拟合；既然如此，就没有必要让网络
从零学习"取平均"这件已知最优的线性操作，把它白送给网络，让容量专注在
非线性残差上。若训练不收敛，最坏情况是退化成 mf，而不是退化成一个更差的模型。
"""

import torch
import torch.nn as nn


class RingDecoder(nn.Module):
    def __init__(self, in_dim: int, n_bits_out: int, hidden: int = 512,
                 dropout: float = 0.1, mf_residual: bool = False):
        super().__init__()
        self.n_bits_out = int(n_bits_out)
        self.mf_residual = bool(mf_residual)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.SiLU(),
            nn.Linear(hidden // 2, n_bits_out),
        )
        if self.mf_residual:
            # feats = [同相 (n_pairs); 正交 (n_pairs)]，每个槽位占 g = n_pairs/total 个同相坐标
            n_pairs = int(in_dim) // 2
            assert n_pairs % self.n_bits_out == 0, \
                f"n_pairs({n_pairs}) 必须能被槽位数({self.n_bits_out})整除"
            g = n_pairs // self.n_bits_out
            m = torch.zeros(self.n_bits_out, in_dim)
            for j in range(self.n_bits_out):
                m[j, j * g:(j + 1) * g] = 1.0 / g
            # persistent=False：老 checkpoint 的 state_dict 仍能直接 load
            self.register_buffer("mf_matrix", m, persistent=False)
            # 零初始化最后一层：初始模型 == mf 判决
            last = self.net[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        out = self.net(feats)
        if self.mf_residual:
            out = out + feats @ self.mf_matrix.t()
        return out

    # ---------------- 统一的构造入口 ----------------
    @classmethod
    def from_config(cls, cfg: dict, hidden: int = 512, dropout: float = 0.1) -> "RingDecoder":
        """从解码器 config 构造（评测脚本统一走这里，避免各处漏字段）。"""
        n_pairs = int(cfg["n_pairs"])
        total = int(cfg["n_bits"]) * int(cfg["ecc"]) + int(cfg["n_check_bits"])
        return cls(2 * n_pairs, total, hidden=hidden, dropout=dropout,
                   mf_residual=bool(cfg.get("mf_residual", False)))

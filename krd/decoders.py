"""密钥条件下的比特解码头（MLP）。输入为环带特征，输出 ECC 扩展比特的 logits。"""

import torch
import torch.nn as nn


class RingDecoder(nn.Module):
    def __init__(self, in_dim: int, n_bits_out: int, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.SiLU(),
            nn.Linear(hidden // 2, n_bits_out),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.net(feats)

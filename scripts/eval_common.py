"""评测脚本共用的小工具：标准 CIFAR 载入、评测输入构造、nonce 协议。

抽出来的目的有三个：
  1. **统一 nonce 协议**：nonce = H(key || counter)，自包含、不依赖 cover
     （旧写法 `derive_nonce(cover)` 需要侧信道传输 nonce，方案并非盲提取）；
  2. **修掉重复取批的旧写法**：旧代码里 `while done < n: next(iter(loader))`
     每次都重建迭代器，实际反复取的是同一个第一批；
  3. 让各 eval 脚本的取数与评测口径一致，避免报告之间"口径漂移"。
"""

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from krd.utils import derive_nonces_from_keys, token_key

CIFAR_TF = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


def cifar_loader(data_root: str, train: bool = False, batch_size: int = 16,
                 shuffle: bool = False) -> DataLoader:
    ds = datasets.CIFAR10(data_root, train=train, download=True, transform=CIFAR_TF)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def gather_covers(loader: DataLoader, n: int) -> torch.Tensor:
    """从 loader 顺序取 n 张图（真正推进迭代器，不再重复第一批）。"""
    xs, done = [], 0
    for x, _ in loader:
        xs.append(x)
        done += x.shape[0]
        if done >= n:
            break
    return torch.cat(xs)[:n]


def make_eval_inputs(loader: DataLoader, n: int, n_bits: int, device: str,
                     nonce_start: int = 0, seed_bits: bool = True):
    """构造评测用 (covers, bits, keys, nonces)。

    nonce 协议：nonce_i = H(key_i || nonce_start + i)，逐图不同、**不含 cover 信息**。
    复原端只需要 (key, counter)；counter 是公开参数，随图传输即可。
    """
    covers = gather_covers(loader, n).to(device)
    keys = [token_key() for _ in range(n)]
    if seed_bits:
        bits = torch.randint(0, 2, (n, n_bits), device=device).float()
    else:  # 确定性比特（复现实验用）
        g = torch.Generator(device="cpu").manual_seed(0)
        bits = torch.randint(0, 2, (n, n_bits), generator=g).float().to(device)
    nonces = derive_nonces_from_keys(keys, start=nonce_start)
    return covers, bits, keys, nonces


def repeat_nonces(key: str, n: int, shared: bool = False) -> list[str]:
    """攻击/对照实验用：shared=True 时所有图共用同一 nonce（复现脆弱基线）。"""
    if shared:
        return [derive_nonces_from_keys([key], start=0)[0]] * n
    return derive_nonces_from_keys([key] * n, start=0)

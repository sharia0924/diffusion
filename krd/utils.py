"""小工具：比特/字符串互转、图像 IO、随机种子。"""

import random

import numpy as np
import torch
from PIL import Image


def seed_everything(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def str_to_bits(message: str, n_bits: int) -> torch.Tensor:
    """UTF-8 字节 -> 定长比特。n_bits 比特可容纳 floor(n_bits/8) 字节。"""
    payload = message.encode("utf-8")
    cap = n_bits // 8
    if len(payload) > cap:
        raise ValueError(f"消息 {len(payload)} 字节超出容量 {cap} 字节（n_bits={n_bits}）")
    payload = payload + b"\x00" * (cap - len(payload))
    arr = np.frombuffer(payload, dtype=np.uint8)
    bits = np.unpackbits(arr)[:n_bits]
    return torch.from_numpy(bits.copy()).float()


def bits_to_str(bits: torch.Tensor) -> str:
    bits = (bits.detach().cpu().view(-1) > 0.5).to(torch.uint8).numpy()
    cap = len(bits) // 8
    payload = np.packbits(bits[: cap * 8]).tobytes()
    return payload.decode("utf-8", errors="replace").rstrip("\x00")


def image_to_tensor(img: Image.Image, size: int | None = None) -> torch.Tensor:
    """PIL -> (1,3,H,W), [-1,1]。"""
    if size is not None:
        img = img.convert("RGB").resize((size, size), Image.LANCZOS)
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1)[None]


def tensor_to_image(t: torch.Tensor) -> Image.Image:
    """(1,3,H,W) 或 (3,H,W), [-1,1] -> PIL。"""
    if t.dim() == 4:
        t = t[0]
    arr = ((t.detach().cpu().float().clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)
    return Image.fromarray(arr.permute(1, 2, 0).numpy())


def token_key(rng: random.Random | None = None) -> str:
    rng = rng or random
    return "%016x" % rng.getrandbits(64)


def derive_nonce(cover: torch.Tensor) -> str:
    """【已弃用，保留兼容】由 cover 内容派生 nonce（SHA256）。

    这个协议有缺陷：nonce 由 **cover** 决定，复原端必须靠侧信道（如 PNG 元数据）
    拿到它，方案因此不是盲提取；而且它把 cover 的指纹带进了协议参数。
    新代码请用 `derive_nonce_from_key`。
    """
    import hashlib
    h = hashlib.sha256()
    h.update(cover.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def derive_nonce_from_key(key: str, counter: int = 0) -> str:
    """自包含 nonce：nonce = H(key || counter) 的前 64 bit。

    逐图把 counter 递增即可得到互不相同的 nonce，用于打散图案位置/相位，
    使"多张同密钥载密图差分平均"退化为随机游走。

    与 derive_nonce 的关键区别：**不依赖 cover**。因此
      - 复原端只需 key + stego 即可（真正的盲提取，无需任何侧信息）；
      - 攻击者即使知道 nonce，也无法从 cover 推出额外信息；
      - cover 更换不影响协议参数，counter 可随图传输（公开参数）。
    """
    import hashlib
    digest = hashlib.sha256(f"krd-nonce:{key}:{int(counter)}".encode("utf-8")).digest()
    return digest[:8].hex()


def derive_nonces_from_keys(keys: list[str], start: int = 0) -> list[str]:
    """为一批密钥派生逐图 nonce（counter 从 start 递增），与部署协议一致。"""
    return [derive_nonce_from_key(k, start + i) for i, k in enumerate(keys)]

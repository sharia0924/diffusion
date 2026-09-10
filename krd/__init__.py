"""KRD-Steg: Key-conditioned Reverse-Diffusion image Steganography.

核心思想：
  - 隐藏端步数自由（DDIM inversion + 注入 + 采样，步数可任选）；
  - 复原端由闭式公式（DDIM 更新式）驱动、步数固定；
  - 密钥决定秘密比特在 x_T 频谱上的位置/相位（密钥门控），
    无密钥/错密钥无法定位图案，解码退化为随机猜测；
  - 复原解码器在可微失真（JPEG/噪声/模糊/缩放）下训练，获得鲁棒性。
"""

from .schedule import Schedule
from .unet import UNet
from .pattern import key_params, inject_pattern, ring_features, ecc_encode, ecc_collapse
from .decoders import RingDecoder
from .stego import TrajStego
from . import perceptual, security

__all__ = [
    "Schedule",
    "UNet",
    "key_params",
    "inject_pattern",
    "ring_features",
    "ecc_encode",
    "ecc_collapse",
    "RingDecoder",
    "TrajStego",
    "perceptual",
    "security",
]

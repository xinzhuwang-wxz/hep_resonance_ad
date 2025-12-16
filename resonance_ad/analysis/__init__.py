"""分析模块：bump hunt 和显著性计算"""

from .bump_hunt import BumpHunter
from .significance import compute_significance

__all__ = ["BumpHunter", "compute_significance"]


"""分析模块：bump hunt 和显著性计算"""

from .bump_hunt import (
    BumpHunter,
    bkg_fit_cubic,
    bkg_fit_quintic,
    bkg_fit_septic,
    parametric_fit,
)
from .significance import compute_significance, calculate_test_statistic
from .bdt import BDTTrainer, run_bdt_bump_hunt

__all__ = [
    "BumpHunter",
    "compute_significance",
    "calculate_test_statistic",
    "bkg_fit_cubic",
    "bkg_fit_quintic",
    "bkg_fit_septic",
    "parametric_fit",
    "BDTTrainer",
    "run_bdt_bump_hunt",
]


"""画图模块"""

from .plot_utils import (
    newplot, hist_with_outline, hist_with_errors, function_with_band, stamp, add_whitespace
)
from .figure import (
    plot_histograms_with_fits,
    plot_features,
    plot_sig,
    plot_upsilon_resonances,
    plot_training_losses,
    plot_roc_curve,
    plot_variations,
)

__all__ = [
    "newplot",
    "hist_with_outline",
    "hist_with_errors",
    "function_with_band",
    "stamp",
    "add_whitespace",
    "plot_histograms_with_fits",
    "plot_features",
    "plot_sig",
    "plot_upsilon_resonances",
    "plot_training_losses",
    "plot_roc_curve",
    "plot_variations",
]


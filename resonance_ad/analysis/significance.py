"""
显著性计算模块

计算统计显著性，包括 local 和 global p-value。
"""

import numpy as np
from scipy import stats
from typing import Optional

from resonance_ad.core.logging import get_logger

logger = get_logger(__name__)


def compute_significance(
    observed: int, expected: float, use_poisson: bool = True
) -> float:
    """
    计算显著性（sigma）
    
    物理假设：
    - 使用 Poisson 统计（如果 use_poisson=True）
    - 否则使用 Gaussian 近似
    
    Args:
        observed: 观测值
        expected: 期望值
        use_poisson: 是否使用 Poisson 统计
        
    Returns:
        显著性（sigma）
    """
    if expected <= 0:
        return 0.0
    
    if use_poisson:
        # Poisson 显著性（简化版本）
        # 实际应该使用更精确的方法
        if observed >= expected:
            # 上尾
            p_value = 1 - stats.poisson.cdf(observed - 1, expected)
        else:
            # 下尾
            p_value = stats.poisson.cdf(observed, expected)
        
        # 转换为 sigma（使用正态分布近似）
        significance = stats.norm.ppf(1 - p_value)
    else:
        # Gaussian 近似
        significance = (observed - expected) / np.sqrt(expected)
    
    return significance


def compute_global_pvalue(
    local_pvalues: np.ndarray,
    n_trials: Optional[int] = None,
) -> float:
    """
    计算 global p-value（考虑 look-elsewhere effect）
    
    物理假设：
    - 使用 Bonferroni 校正（简化版本）
    - 更精确的方法应该考虑 bin 之间的相关性
    
    Args:
        local_pvalues: 每个 bin 的 local p-value
        n_trials: 试验次数（如果为 None，则使用 bin 数量）
        
    Returns:
        Global p-value
    """
    if n_trials is None:
        n_trials = len(local_pvalues)
    
    # Bonferroni 校正
    min_local_pvalue = np.min(local_pvalues)
    global_pvalue = min(min_local_pvalue * n_trials, 1.0)
    
    return global_pvalue


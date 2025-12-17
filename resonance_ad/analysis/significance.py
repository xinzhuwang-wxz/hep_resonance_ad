"""
显著性计算模块

计算统计显著性，包括 local 和 global p-value。
"""

import numpy as np
from scipy import stats
from scipy.special import loggamma
from scipy.optimize import minimize
from typing import Optional, Tuple

from resonance_ad.core.logging import get_logger
from resonance_ad.physics.binning import get_bins
from resonance_ad.analysis.bump_hunt import parametric_fit

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


def integral_polynomial(lower: float, upper: float, bin_width: float, *theta) -> float:
    """
    计算多项式在 [lower, upper] 区间内的积分
    
    Args:
        lower: 积分下界
        upper: 积分上界
        bin_width: bin 宽度
        *theta: 多项式系数（从低次到高次）
        
    Returns:
        积分值（归一化到 bin_width）
    """
    degree = len(theta) - 1
    integral_val = 0
    for i in range(degree + 1):
        integral_val += theta[i] / (i + 1) * ((upper)**(i + 1) - (lower)**(i + 1))
    
    return integral_val / bin_width


def build_histogram_with_weights(
    data: np.ndarray, weights: np.ndarray, bins: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list, np.ndarray]:
    """
    构建带权重的直方图
    
    Args:
        data: 数据数组
        weights: 权重数组
        bins: bin 边界
        
    Returns:
        (y_vals, y_counts, bins, bin_weights, bin_err)
    """
    y_vals, _bins = np.histogram(data, bins=bins, density=False, weights=weights)
    y_counts, _ = np.histogram(data, bins=bins, density=False)
    
    digits = np.digitize(data, _bins)
    bin_weights = [weights[digits == i] for i in range(1, len(_bins))]
    # 确保所有 bin 都有权重数组（即使为空）
    bin_err = np.asarray([
        np.linalg.norm(weights[digits == i]) if len(weights[digits == i]) > 0 else 0
        for i in range(1, len(_bins))
    ])
    
    return y_vals, y_counts, bins, bin_weights, bin_err


def binned_likelihood(
    yvals: np.ndarray, ycounts: np.ndarray, weights: list, fit_vals: np.ndarray
) -> float:
    """
    计算 binned Poisson likelihood（考虑权重）
    
    Args:
        yvals: 加权后的 bin 值
        ycounts: bin 计数
        weights: 每个 bin 的权重列表
        fit_vals: 拟合值
        
    Returns:
        Log likelihood
    """
    log_likelihood = 0
    
    for i in range(len(yvals)):
        if len(weights[i]) == 0:
            # 如果没有权重，使用标准 Poisson
            if fit_vals[i] > 0:
                log_likelihood += stats.poisson.logpmf(yvals[i], fit_vals[i])
            continue
            
        expval_weights = np.mean(weights[i])
        expval_weights2 = np.mean(weights[i]**2)
        len_weights = len(weights[i])
        
        if len_weights == 0 or (np.abs(expval_weights - 1) < 1e-3 and np.abs(expval_weights2 - 1) < 1e-3):
            log_likelihood += stats.poisson.logpmf(yvals[i], fit_vals[i])
        else:
            scale_factor = expval_weights2 / expval_weights
            lambda_prime = fit_vals[i] / scale_factor
            n_prime = yvals[i] / scale_factor
            
            # 原代码使用 if True:，意味着总是执行这个分支
            # 添加数值稳定性检查以避免 RuntimeWarning
            if lambda_prime > 0 and n_prime >= 0:
                log_likelihood += n_prime * np.log(lambda_prime) - lambda_prime - loggamma(n_prime + 1)
            # 如果 lambda_prime <= 0 或 n_prime < 0，跳过这个 bin（与原代码行为一致，原代码也会产生警告）
    
    return log_likelihood


def likelihood(
    data: np.ndarray,
    s: float,
    SR_left: float,
    SR_right: float,
    SB_left: float,
    SB_right: float,
    num_bins: int,
    weights: Optional[np.ndarray],
    *theta
) -> float:
    """
    计算 likelihood（考虑信号 s）
    
    Args:
        data: 质量数据
        s: 信号强度
        SR_left: Signal region 左边界
        SR_right: Signal region 右边界
        SB_left: Sideband 左边界
        SB_right: Sideband 右边界
        num_bins: SR 中的 bin 数量
        weights: 权重数组
        *theta: 背景多项式系数
        
    Returns:
        -2 * log likelihood
    """
    plot_bins_all, plot_bins_SR, plot_bins_left, plot_bins_right, plot_centers_all, plot_centers_SR, plot_centers_SB = get_bins(
        SR_left, SR_right, SB_left, SB_right, num_bins, binning="linear"
    )
    plot_centers_left = 0.5 * (plot_bins_left[1:] + plot_bins_left[:-1])
    plot_centers_right = 0.5 * (plot_bins_right[1:] + plot_bins_right[:-1])
    
    if weights is None:
        weights = np.ones_like(data)
    
    # 获取左侧 SB 数据
    loc_bkg_left = data[data < SR_left]
    weights_left = weights[data < SR_left]
    y_vals_left, y_counts_left, _bins, left_weights, left_err = build_histogram_with_weights(
        loc_bkg_left, weights_left, plot_bins_left
    )
    
    # 获取右侧 SB 数据
    loc_bkg_right = data[data > SR_right]
    weights_right = weights[data > SR_right]
    y_vals_right, y_counts_right, _bins, right_weights, right_err = build_histogram_with_weights(
        loc_bkg_right, weights_right, plot_bins_right
    )
    
    # SB bins 的 log likelihood
    log_likelihood = 0
    fit_vals_left = parametric_fit(plot_centers_left, *theta)
    fit_vals_right = parametric_fit(plot_centers_right, *theta)
    
    log_likelihood += binned_likelihood(y_vals_left, y_counts_left, left_weights, fit_vals_left)
    log_likelihood += binned_likelihood(y_vals_right, y_counts_right, right_weights, fit_vals_right)
    
    # SR 的 log likelihood
    bin_width = plot_bins_SR[1] - plot_bins_SR[0]
    loc_data = data[np.logical_and(data > SR_left, data < SR_right)]
    loc_weights = weights[np.logical_and(data > SR_left, data < SR_right)]
    num_SR = np.sum(loc_weights)
    num_bkg = integral_polynomial(SR_left, SR_right, bin_width, *theta)
    s_prime = s * (s > 0)  # 确保信号为正
    
    expval_weights = np.mean(loc_weights) if len(loc_weights) > 0 else 1.0
    expval_weights2 = np.mean(loc_weights**2) if len(loc_weights) > 0 else 1.0
    
    if len(loc_weights) == 0 or (np.abs(expval_weights - 1) < 1e-3 and np.abs(expval_weights2 - 1) < 1e-3):
        log_likelihood += stats.poisson.logpmf(num_SR, num_bkg + s_prime)
    else:
        scale_factor = expval_weights2 / expval_weights
        lambda_prime = (num_bkg + s_prime) / scale_factor
        n_prime = num_SR / scale_factor
        if lambda_prime > 0 and n_prime >= 0:
            log_likelihood += n_prime * np.log(lambda_prime) - lambda_prime - loggamma(n_prime + 1)
    
    return -2 * log_likelihood


def cheat_likelihood(
    data: np.ndarray,
    SR_left: float,
    SR_right: float,
    SB_left: float,
    SB_right: float,
    num_bins: int,
    weights: Optional[np.ndarray],
    *theta
) -> float:
    """
    计算 cheat likelihood（只拟合背景，不包含信号）
    
    Args:
        data: 质量数据
        SR_left: Signal region 左边界
        SR_right: Signal region 右边界
        SB_left: Sideband 左边界
        SB_right: Sideband 右边界
        num_bins: SR 中的 bin 数量
        weights: 权重数组
        *theta: 背景多项式系数
        
    Returns:
        -2 * log likelihood
    """
    plot_bins_all, plot_bins_SR, plot_bins_left, plot_bins_right, plot_centers_all, plot_centers_SR, plot_centers_SB = get_bins(
        SR_left, SR_right, SB_left, SB_right, num_bins, binning="linear"
    )
    plot_centers_left = 0.5 * (plot_bins_left[1:] + plot_bins_left[:-1])
    plot_centers_right = 0.5 * (plot_bins_right[1:] + plot_bins_right[:-1])
    
    if weights is None:
        weights = np.ones_like(data)
    
    # 获取左侧 SB 数据
    loc_bkg_left = data[data < SR_left]
    weights_left = weights[data < SR_left]
    y_vals_left, y_counts_left, _bins, left_weights, left_err = build_histogram_with_weights(
        loc_bkg_left, weights_left, plot_bins_left
    )
    
    # 获取右侧 SB 数据
    loc_bkg_right = data[data > SR_right]
    weights_right = weights[data > SR_right]
    y_vals_right, y_counts_right, _bins, right_weights, right_err = build_histogram_with_weights(
        loc_bkg_right, weights_right, plot_bins_right
    )
    
    # SB bins 的 log likelihood
    log_likelihood = 0
    fit_vals_left = parametric_fit(plot_centers_left, *theta)
    fit_vals_right = parametric_fit(plot_centers_right, *theta)
    
    log_likelihood += binned_likelihood(y_vals_left, y_counts_left, left_weights, fit_vals_left)
    log_likelihood += binned_likelihood(y_vals_right, y_counts_right, right_weights, fit_vals_right)
    
    return -2 * log_likelihood


def null_hypothesis(
    data: np.ndarray,
    SR_left: float,
    SR_right: float,
    SB_left: float,
    SB_right: float,
    num_bins: int,
    weights: Optional[np.ndarray],
    *theta
) -> float:
    """
    计算 null hypothesis likelihood（s=0）
    
    Args:
        data: 质量数据
        SR_left: Signal region 左边界
        SR_right: Signal region 右边界
        SB_left: Sideband 左边界
        SB_right: Sideband 右边界
        num_bins: SR 中的 bin 数量
        weights: 权重数组
        *theta: 背景多项式系数
        
    Returns:
        -2 * log likelihood (s=0)
    """
    return likelihood(data, 0, SR_left, SR_right, SB_left, SB_right, num_bins, weights, *theta)


def calculate_test_statistic(
    data: np.ndarray,
    SR_left: float,
    SR_right: float,
    SB_left: float,
    SB_right: float,
    num_bins: int,
    weights: Optional[np.ndarray] = None,
    degree: int = 5,
    starting_guess: Optional[np.ndarray] = None,
    verbose_plot: bool = False,
    return_popt: bool = False,
    max_iter_nelder: int = 15090,  # 原代码使用 15090
) -> Tuple[float, float, float]:
    """
    计算 test statistic q0（profiled log likelihood ratio）
    
    完全对齐原代码 dimuonAD/helpers/stats_functions.py:calculate_test_statistic
    
    物理假设：
    - 使用 profiled log likelihood ratio test
    - q0 = -2 * [L(s=0, theta_hat_hat) - L(s_hat, theta_hat)]
    
    Args:
        data: 质量数据
        SR_left: Signal region 左边界
        SR_right: Signal region 右边界
        SB_left: Sideband 左边界
        SB_right: Sideband 右边界
        num_bins: SR 中的 bin 数量
        weights: 权重数组（可选）
        degree: 多项式阶数（原代码中未使用，但保留以兼容）
        starting_guess: 初始猜测参数
        verbose_plot: 是否显示优化信息
        return_popt: 是否返回拟合参数
        max_iter_nelder: Nelder-Mead 最大迭代次数（原代码使用 15090）
        
    Returns:
        (integrated_signal, integrated_background, test_statistic) 或
        (integrated_signal, integrated_background, test_statistic, popt) 如果 return_popt=True
    """
    plot_bins_all, plot_bins_SR, plot_bins_left, plot_bins_right, plot_centers_all, plot_centers_SR, plot_centers_SB = get_bins(
        SR_left, SR_right, SB_left, SB_right, num_bins, binning="linear"
    )
    
    if weights is None:
        weights = np.ones_like(data)
    
    bin_width = plot_bins_SR[1] - plot_bins_SR[0]
    if starting_guess is None:
        average_bin_count = len(data) / len(plot_centers_all)
        # 原代码使用固定的 10 个参数：[average_bin_count, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # 但实际多项式参数数量是 degree+1，这里使用 degree+1 以确保兼容性
        starting_guess = [average_bin_count] + [0 for i in range(max(degree, 9))]
    
    # 拟合 s=0 假设（null hypothesis）
    # 原代码直接使用 Nelder-Mead，不检查 fit.success
    lambda_null = lambda theta: null_hypothesis(
        data, SR_left, SR_right, SB_left, SB_right, num_bins, weights, *theta
    )
    fit = minimize(
        lambda_null,
        x0=starting_guess,
        method='Nelder-Mead',
        options={'maxiter': max_iter_nelder, "disp": verbose_plot}
    )
    theta_hat_hat = fit.x
    null_fit_likelihood = null_hypothesis(
        data, SR_left, SR_right, SB_left, SB_right, num_bins, weights, *theta_hat_hat
    )
    
    # 拟合 s=float 假设（alternative hypothesis）
    # 原代码直接使用 Nelder-Mead，不检查 fit.success
    lambda_cheat = lambda theta: cheat_likelihood(
        data, SR_left, SR_right, SB_left, SB_right, num_bins, weights, *theta
    )
    fit = minimize(
        lambda_cheat,
        x0=theta_hat_hat,
        method='Nelder-Mead',
        options={'maxiter': max_iter_nelder, "disp": verbose_plot}
    )
    theta_hat = fit.x
    integrated_background = integral_polynomial(SR_left, SR_right, bin_width, *theta_hat)
    loc_weights = weights[np.logical_and(data > SR_left, data < SR_right)]
    num_SR = np.sum(loc_weights)
    integrated_signal = num_SR - integrated_background
    best_fit_likelihood = likelihood(
        data, integrated_signal, SR_left, SR_right, SB_left, SB_right, num_bins, weights, *theta_hat
    )
    
    # 计算 test statistic
    # 原代码：test_statistic = (null_fit_likelihood - best_fit_likelihood)
    test_statistic = null_fit_likelihood - best_fit_likelihood
    if integrated_signal < 0:
        test_statistic = 0
    if test_statistic < 0:
        test_statistic = 0
    
    if return_popt:
        return integrated_signal, integrated_background, test_statistic, theta_hat
    
    return integrated_signal, integrated_background, test_statistic


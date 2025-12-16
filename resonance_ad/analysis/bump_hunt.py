"""
Bump Hunt 模块

实现背景拟合和 anomaly score 计算。
"""

import numpy as np
import torch
from scipy.optimize import curve_fit
from typing import Dict, Tuple, Optional

from resonance_ad.core.logging import get_logger

logger = get_logger(__name__)


def parametric_fit(x: np.ndarray, *theta) -> np.ndarray:
    """
    多项式拟合函数
    
    Args:
        x: 自变量
        *theta: 多项式系数（从低次到高次）
        
    Returns:
        拟合值
    """
    degree = len(theta) - 1
    y = np.zeros_like(x)
    for i in range(degree + 1):
        y += theta[i] * (x) ** i
    return y


def build_histogram(
    data: np.ndarray, weights: Optional[np.ndarray], bins: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    构建直方图
    
    Args:
        data: 数据数组
        weights: 权重数组（可选）
        bins: bin 边界
        
    Returns:
        (y_vals, y_counts, bins, bin_weights, bin_err)
    """
    if weights is None:
        weights = np.ones_like(data)
    
    y_counts, _ = np.histogram(data, bins=bins)
    y_vals, _ = np.histogram(data, bins=bins, weights=weights)
    bin_weights = y_vals
    bin_err = np.sqrt(y_counts)  # 简化：使用 Poisson 误差
    
    return y_vals, y_counts, bins, bin_weights, bin_err


def fit_background(
    masses: np.ndarray,
    fit_degree: int,
    SR_left: float,
    SR_right: float,
    plot_bins_left: np.ndarray,
    plot_bins_right: np.ndarray,
    plot_centers_SB: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    拟合背景（sideband 数据）
    
    物理假设：
    - 背景可以用多项式描述
    - Sideband 区域没有信号
    
    Args:
        masses: 质量数组
        fit_degree: 多项式阶数
        SR_left: Signal region 左边界
        SR_right: Signal region 右边界
        plot_bins_left: 左侧 bin 边界
        plot_bins_right: 右侧 bin 边界
        plot_centers_SB: Sideband bin 中心
        weights: 权重数组（可选）
        
    Returns:
        (popt, pcov, chi2)
        - popt: 拟合参数
        - pcov: 协方差矩阵
        - chi2: chi-square 值
    """
    if weights is None:
        weights = np.ones_like(masses)
    
    # 获取 sideband 数据
    loc_bkg_left = masses[masses < SR_left]
    weights_left = weights[masses < SR_left]
    plot_centers_left = 0.5 * (plot_bins_left[1:] + plot_bins_left[:-1])
    y_vals_left, _, _, _, bin_err_left = build_histogram(
        loc_bkg_left, weights_left, plot_bins_left
    )
    
    loc_bkg_right = masses[masses > SR_right]
    weights_right = weights[masses > SR_right]
    plot_centers_right = 0.5 * (plot_bins_right[1:] + plot_bins_right[:-1])
    y_vals_right, _, _, _, bin_err_right = build_histogram(
        loc_bkg_right, weights_right, plot_bins_right
    )
    
    # 合并 sideband 数据
    y_vals = np.concatenate((y_vals_left, y_vals_right))
    errs = np.concatenate((bin_err_left, bin_err_right))
    y_err = np.sqrt(errs ** 2 + 1)
    
    # 初始化参数
    average_bin_count = np.mean(y_vals)
    p0 = [average_bin_count] + [0 for i in range(fit_degree)]
    n_params_fit = fit_degree + 1
    
    # 拟合
    lower_bounds = [-np.inf for x in range(n_params_fit)]
    upper_bounds = [np.inf for x in range(n_params_fit)]
    
    popt, pcov = curve_fit(
        parametric_fit,
        plot_centers_SB,
        y_vals,
        p0,
        sigma=y_err,
        maxfev=20000,
        bounds=(lower_bounds, upper_bounds),
    )
    
    # 计算 chi-square
    y_fit = parametric_fit(plot_centers_SB, *popt)
    chi2 = np.sum((y_fit - y_vals) ** 2 / y_err ** 2)
    
    return popt, pcov, chi2


class BumpHunter:
    """
    Bump Hunter
    
    实现基于 anomaly score 的 bump hunt。
    """
    
    def __init__(self, config):
        """
        初始化 Bump Hunter
        
        Args:
            config: Config 对象
        """
        self.config = config
        self.logger = get_logger(__name__)
    
    def compute_anomaly_scores(
        self,
        model,
        data: np.ndarray,
        device: torch.device,
    ) -> np.ndarray:
        """
        计算 anomaly scores
        
        Anomaly score = -log p(x|m)，其中 p 是 flow 模型给出的概率密度。
        
        Args:
            model: CATHODE 模型
            data: 数据数组 (n_events, n_features)，最后一列是 mass
            device: 设备
            
        Returns:
            Anomaly scores 数组
        """
        import torch
        
        model.eval()
        with torch.no_grad():
            data_tensor = torch.FloatTensor(data).to(device)
            features = data_tensor[:, :-1]
            masses = torch.reshape(data_tensor[:, -1], (-1, 1))
            
            log_probs = model.log_probs(features, masses)
            scores = -log_probs.cpu().numpy().flatten()
        
        return scores
    
    def hunt_bump(
        self,
        SR_data: np.ndarray,
        SR_scores: np.ndarray,
        bins_SR: np.ndarray,
        background_expectation: np.ndarray,
    ) -> Dict:
        """
        执行 bump hunt
        
        Args:
            SR_data: Signal region 数据
            SR_scores: Signal region anomaly scores
            bins_SR: Signal region bin 边界
            background_expectation: 每个 bin 的背景期望值
            
        Returns:
            包含结果的字典
        """
        # 计算每个 bin 的观测值和期望值
        observed, _ = np.histogram(SR_data[:, -1], bins=bins_SR)
        
        # 计算显著性（简化版本）
        significances = []
        for i in range(len(observed)):
            obs = observed[i]
            exp = background_expectation[i]
            if exp > 0:
                # 使用 Poisson 显著性
                significance = (obs - exp) / np.sqrt(exp)
            else:
                significance = 0
            significances.append(significance)
        
        # 找到最大显著性
        max_significance_idx = np.argmax(significances)
        max_significance = significances[max_significance_idx]
        
        results = {
            "observed": observed,
            "expected": background_expectation,
            "significances": np.array(significances),
            "max_significance": max_significance,
            "max_significance_bin": max_significance_idx,
            "bins": bins_SR,
        }
        
        return results


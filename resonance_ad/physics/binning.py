"""
分 bin 函数

用于定义 sideband 和 signal region 的 binning。
"""

import numpy as np
from typing import Tuple


def get_bins(
    SR_left: float,
    SR_right: float,
    SB_left: float,
    SB_right: float,
    num_bins_SR: int,
    binning: str = "linear",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    生成 sideband 和 signal region 的 binning
    
    物理假设：
    - SR 的 binning 是均匀的（linear）或对数的（log）
    - SB 的 binning 与 SR 的 bin width 一致
    
    Args:
        SR_left: Signal region 左边界
        SR_right: Signal region 右边界
        SB_left: Sideband 左边界
        SB_right: Sideband 右边界
        num_bins_SR: Signal region 的 bin 数量
        binning: "linear" 或 "log"
        
    Returns:
        (plot_bins_all, plot_bins_SR, plot_bins_left, plot_bins_right,
         plot_centers_all, plot_centers_SR, plot_centers_SB)
    """
    if binning == "linear":
        # SR binning（与原始代码一致：num_bins_SR 个点，即 num_bins_SR-1 个 bins）
        plot_bins_SR = np.linspace(SR_left, SR_right, num_bins_SR)
        plot_centers_SR = 0.5 * (plot_bins_SR[1:] + plot_bins_SR[:-1])
        width = plot_bins_SR[1] - plot_bins_SR[0]
        
        # SBL binning（向左扩展，与原始代码一致）
        plot_bins_left = np.arange(SR_left, SB_left - width, -width)[::-1]
        if plot_bins_left[0] < SB_left:
            plot_bins_left = plot_bins_left[1:]
        plot_centers_left = 0.5 * (plot_bins_left[1:] + plot_bins_left[:-1])
        
        # SBH binning（向右扩展，与原始代码一致）
        plot_bins_right = np.arange(SR_right, SB_right + width, width)
        if plot_bins_right[-1] > SB_right:
            plot_bins_right = plot_bins_right[:-1]
        plot_centers_right = 0.5 * (plot_bins_right[1:] + plot_bins_right[:-1])
        
    elif binning == "log":
        # SR binning（对数，与原始代码一致：num_bins_SR 个点）
        plot_bins_SR = np.logspace(np.log10(SR_left), np.log10(SR_right), num_bins_SR)
        plot_centers_SR = np.array([np.sqrt(plot_bins_SR[i]*plot_bins_SR[i+1]) for i in range(len(plot_bins_SR)-1)])
        ratio = plot_bins_SR[1]/plot_bins_SR[0]
        
        # SBL binning（与原始代码一致）
        current_endpoint = plot_bins_SR[0]
        plot_bins_left = [current_endpoint]
        while current_endpoint > SB_left:
            next_endpoint = current_endpoint/ratio
            plot_bins_left.insert(0, next_endpoint)
            current_endpoint = next_endpoint
        if plot_bins_left[0] < SB_left:
            plot_bins_left = plot_bins_left[1:]
        plot_centers_left = np.array([np.sqrt(plot_bins_left[i]*plot_bins_left[i+1]) for i in range(len(plot_bins_left)-1)])
        
        # SBH binning（与原始代码一致）
        current_endpoint = plot_bins_SR[-1]
        plot_bins_right = [current_endpoint]
        while current_endpoint < SB_right:
            next_endpoint = current_endpoint*ratio
            plot_bins_right.append(next_endpoint)
            current_endpoint = next_endpoint
        if plot_bins_right[-1] > SB_right:
            plot_bins_right = plot_bins_right[:-1]
        plot_centers_right = np.array([np.sqrt(plot_bins_right[i]*plot_bins_right[i+1]) for i in range(len(plot_bins_right)-1)])
    else:
        raise ValueError(f"Unknown binning type: {binning}")
    
    # 合并所有 bins（与原始代码完全一致）
    plot_centers_all = np.concatenate((plot_centers_left, plot_centers_SR, plot_centers_right))
    plot_centers_SB = np.concatenate((plot_centers_left, plot_centers_right))
    plot_bins_all = np.concatenate([plot_bins_left[:-1], plot_bins_SR, plot_bins_right[1:]])
    
    return (
        plot_bins_all,
        plot_bins_SR,
        plot_bins_left,
        plot_bins_right,
        plot_centers_all,
        plot_centers_SR,
        plot_centers_SB,
    )


def get_bins_for_scan(
    path_to_bin_defs_folder: str,
    window_index: int,
    scale_bins: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    从保存的 bin 定义文件中加载 binning
    
    用于批量扫描或预处理的场景。
    
    Args:
        path_to_bin_defs_folder: bin 定义文件夹路径
        window_index: 窗口索引
        scale_bins: 是否应用 scaler 变换
        
    Returns:
        同 get_bins 的返回值
    """
    import pickle
    from pathlib import Path
    
    path = Path(path_to_bin_defs_folder)
    
    # 加载 bin 定义
    with open(path / "bin_definitions", "rb") as infile:
        bin_definitions = pickle.load(infile)
    
    window_bin_definitions = bin_definitions[window_index]
    
    plot_bins_SR = window_bin_definitions["SR"]
    plot_bins_left = window_bin_definitions["SBL"]
    plot_bins_right = window_bin_definitions["SBH"]
    
    plot_centers_SR = np.array(
        [
            np.sqrt(plot_bins_SR[i] * plot_bins_SR[i + 1])
            for i in range(len(plot_bins_SR) - 1)
        ]
    )
    plot_centers_left = np.array(
        [
            np.sqrt(plot_bins_left[i] * plot_bins_left[i + 1])
            for i in range(len(plot_bins_left) - 1)
        ]
    )
    plot_centers_right = np.array(
        [
            np.sqrt(plot_bins_right[i] * plot_bins_right[i + 1])
            for i in range(len(plot_bins_right) - 1)
        ]
    )
    
    plot_centers_all = np.concatenate(
        (plot_centers_left, plot_centers_SR, plot_centers_right)
    )
    plot_centers_SB = np.concatenate((plot_centers_left, plot_centers_right))
    plot_bins_all = np.concatenate(
        [plot_bins_left[:-1], plot_bins_SR, plot_bins_right[1:]]
    )
    
    if scale_bins:
        # 加载 scaler 并应用变换
        with open(path / f"mass_scaler_window{window_index}", "rb") as infile:
            mass_scaler = pickle.load(infile)
        
        plot_bins_all = mass_scaler.transform(
            np.array(plot_bins_all).reshape(-1, 1)
        ).reshape(-1)
        plot_bins_SR = mass_scaler.transform(
            np.array(plot_bins_SR).reshape(-1, 1)
        ).reshape(-1)
        plot_bins_left = mass_scaler.transform(
            np.array(plot_bins_left).reshape(-1, 1)
        ).reshape(-1)
        plot_bins_right = mass_scaler.transform(
            np.array(plot_bins_right).reshape(-1, 1)
        ).reshape(-1)
        plot_centers_all = mass_scaler.transform(
            np.array(plot_centers_all).reshape(-1, 1)
        ).reshape(-1)
        plot_centers_SR = mass_scaler.transform(
            np.array(plot_centers_SR).reshape(-1, 1)
        ).reshape(-1)
        plot_centers_SB = mass_scaler.transform(
            np.array(plot_centers_SB).reshape(-1, 1)
        ).reshape(-1)
    
    return (
        plot_bins_all,
        plot_bins_SR,
        plot_bins_left,
        plot_bins_right,
        plot_centers_all,
        plot_centers_SR,
        plot_centers_SB,
    )


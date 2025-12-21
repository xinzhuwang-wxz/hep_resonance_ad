"""
绘图函数模块

完全复刻原始代码 08_render.ipynb 中的绘图方式。
包含三种主要图：
1. plot_histograms_with_fits - Cut Histograms
2. plot_features - Feature Histograms  
3. plot_sig - Significances
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import norm
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

from resonance_ad.plotting.plot_utils import (
    newplot,
    hist_with_outline,
    function_with_band,
)
from resonance_ad.physics.binning import get_bins


def plot_upsilon_resonances(ax):
    """绘制 Upsilon 共振线（与原始代码完全一致）"""
    ax.axvline(9.460, color="black", linestyle="--", alpha=0.15, lw=1.5)
    ax.text(
        9.460 * 0.995, 1e4, r"$\Upsilon(1S)$",
        rotation=90, verticalalignment="center",
        horizontalalignment="right", fontsize=5
    )
    ax.axvline(10.023, color="black", linestyle="--", alpha=0.15, lw=1.5)
    ax.text(
        10.023 * 0.995, 1e4, r"$\Upsilon(2S)$",
        rotation=90, verticalalignment="center",
        horizontalalignment="right", fontsize=5
    )
    ax.axvline(10.355, color="black", linestyle="--", alpha=0.15, lw=1.5)
    ax.text(
        10.355 * 0.995, 1e4, r"$\Upsilon(3S)$",
        rotation=90, verticalalignment="center",
        horizontalalignment="right", fontsize=5
    )


def plot_histograms_with_fits(
    save_data: Dict,
    SB_left: float,
    SR_left: float,
    SR_right: float,
    SB_right: float,
    fit_function,
    num_bins_SR: int,
    title: Optional[str] = None,
    upsilon_lines: bool = True,
    colors: Optional[List] = None,
    alphas: Optional[np.ndarray] = None,
    line_0: Optional[str] = None,
):
    """
    绘制多个 FPR 阈值下的质量谱和拟合（Cut Histograms）
    
    完全复刻原始代码的 plot_histograms_with_fits 函数。
    
    Args:
        save_data: 包含多个 FPR 阈值结果的字典
            - fpr_thresholds: FPR 阈值数组
            - popts: 拟合参数列表（每个 FPR 阈值一个）
            - pcovs: 协方差矩阵列表（每个 FPR 阈值一个）
            - significances: 显著性列表（每个 FPR 阈值一个）
            - filtered_masses: 过滤后的质量数组列表（每个 FPR 阈值一个）
            - y_vals: y 值列表（每个 FPR 阈值一个，可选）
        SB_left, SR_left, SR_right, SB_right: 窗口边界
        fit_function: 拟合函数
        num_bins_SR: SR 的 bin 数量
        title: 图标题
        upsilon_lines: 是否绘制 Upsilon 共振线
        colors: 颜色列表（每个 FPR 阈值一个）
        alphas: 透明度数组（每个 FPR 阈值一个）
        line_0: 第一行文本标签
    """
    fpr_thresholds = save_data["fpr_thresholds"]
    popts = save_data["popts"]
    pcovs = save_data["pcovs"]
    significances = save_data["significances"]
    filtered_masses = save_data["filtered_masses"]
    y_vals = save_data.get("y_vals", [None] * len(fpr_thresholds))
    
    # 定义 bins（与原始代码一致）
    plot_bins_all, plot_bins_SR, plot_bins_left, plot_bins_right, plot_centers_all, plot_centers_SR, plot_centers_SB = get_bins(
        SR_left, SR_right, SB_left, SB_right, num_bins_SR=num_bins_SR
    )
    
    fig, ax = newplot("column", width=4, height=4)
    
    # 默认颜色和透明度（与原始代码一致）
    if colors is None:
        colors = ['#0C5DA5', '#00B945', '#FF9500', '#FF2C00', '#845B97', '#474747', '#9e9e9e']
    if alphas is None:
        alphas = np.linspace(1, 0.5, len(fpr_thresholds))[::-1]
    
    # 为每个 FPR 阈值绘制（与原始代码一致：只绘制偶数索引）
    for t, threshold in enumerate(fpr_thresholds):
        if t % 2 == 1:
            continue
        
        filtered_masses_t = filtered_masses[t]
        popt = popts[t]
        pcov = pcovs[t]
        significance = significances[t]
        y_vals_t = y_vals[t] if t < len(y_vals) else None
        
        # 确保是 numpy array
        if hasattr(filtered_masses_t, 'to_numpy'):
            filtered_masses_t = filtered_masses_t.to_numpy()
        elif not isinstance(filtered_masses_t, np.ndarray):
            filtered_masses_t = np.array(filtered_masses_t)
        
        # 绘制拟合函数（与原始代码一致）
        plt.plot(
            plot_centers_all,
            fit_function(plot_centers_all, *popt),
            lw=2,
            linestyle="dashed",
            color=colors[t]
        )
        function_with_band(
            ax,
            f=fit_function,
            range=(SB_left, SB_right),
            params=popt,
            pcov=pcov,
            color=colors[t],
            alpha_band=0.2,
            alpha_line=0.85,
            linestyle="dashed",
            lw=2,
        )
        
        # 绘制数据（与原始代码完全一致）
        # 原始代码：label_string = str(round(100*threshold, 2))+r"%, "+str(round(significance,2) + r"$\sigma$")
        # 注意：原始代码中 round(significance,2) + r"$\sigma$" 在 str() 内部，但 Python 中需要先转换
        # 原始代码的写法在 Python 中会报错，我们修正为正确的语法
        label_string = str(round(100*threshold, 2)) + r"%, " + str(round(significance, 2)) + r"$\sigma$"
        hist_with_outline(
            ax,
            points=filtered_masses_t,
            bins=plot_bins_all,
            range=(SB_left, SB_right),
            color=colors[t],
            label=label_string,
            alpha_1=0.005,
            lw=1.5,
            alpha_2=alphas[t],
        )
    
    # 添加文本标签（与原始代码一致）
    line1 = r"8.7 fb$^{-1}$"
    line2 = r"$\sqrt{s} = 13$ TeV"
    line3 = r"Anti-Isolated"
    line4 = r""
    
    starting_x = 0.04500
    starting_y = 0.95
    delta_y = 0.055
    text_alpha = 0.75
    
    if line_0 is not None:
        # 原始代码：ax.text(starting_x, starting_y - (0) * delta_y, line_0, ...)
        ax.text(
            starting_x, starting_y - (0) * delta_y,
            line_0,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            alpha=text_alpha
        )
    
    ax.text(
        starting_x, starting_y - 1 * delta_y,
        line1,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        alpha=text_alpha
    )
    ax.text(
        starting_x, starting_y - 2 * delta_y,
        line2,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        alpha=text_alpha
    )
    ax.text(
        starting_x, starting_y - 3 * delta_y,
        line3,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        alpha=text_alpha
    )
    ax.text(
        starting_x, starting_y - 4 * delta_y,
        line4,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        alpha=text_alpha
    )
    
    # 图例和标题（与原始代码完全一致）
    plt.legend(loc="upper right", fontsize=9, title="False Positive Rate", title_fontsize=10)
    if title:
        plt.title(title, fontsize=10, loc="right")
    
    # 标记窗口边界（与原始代码完全一致）
    plt.axvline(SR_left, color="k", lw=1.5, zorder=10)
    plt.axvline(SR_right, color="k", lw=1.5, zorder=10)
    
    # 设置标签和刻度（与原始代码完全一致）
    plt.xlabel("Dimuon Mass $m_{\mu\mu}$ [GeV]")
    plt.ylabel("Events")
    plt.yscale("log")
    plt.xlim(SB_left, SB_right)
    plt.ylim(0.01, 1e5)
    plt.xticks()
    plt.yticks()
    plt.minorticks_on()
    plt.tick_params(axis='x', which='minor', bottom=True)
    plt.tick_params(axis='y', which='minor', left=True)
    
    # 绘制 Upsilon 共振线（如果启用）
    if upsilon_lines:
        plot_upsilon_resonances(ax)
    
    return fig, ax


def plot_features(
    save_data: Dict,
    isolation_data: Dict[str, np.ndarray],
    feature_set: List[str],
    colors: List,
    alphas: List,
    num_bins_SR: int,
    fit_type: str,
    num_points: int = 7,
):
    """
    绘制三个辅助特征的分布图（Feature Histograms）
    
    完全复刻原始代码的 plot_features 函数。
    
    Args:
        save_data: 包含特征数据的字典
            - fpr_thresholds: FPR 阈值数组
            - features: 过滤后的特征列表（每个 FPR 阈值一个）
        isolation_data: 反隔离切割前的数据 {feature: array}
        feature_set: 要绘制的特征列表（应该是 ["dimu_pt", "mu0_ip3d", "mu1_ip3d"]）
        colors: 颜色列表（每个 FPR 阈值一个）
        alphas: 透明度列表（每个 FPR 阈值一个）
        num_bins_SR: SR 的 bin 数量（用于文本标签）
        fit_type: 拟合类型（用于文本标签）
        num_points: 要绘制的 FPR 点数（默认 7）
    """
    fpr_thresholds = save_data["fpr_thresholds"]
    filtered_features_all = save_data["features"]
    
    n_features = len(feature_set)
    nbins = 35
    
    # bins 定义（与原始代码完全一致，扩展以支持所有可能用到的特征）
    bins = {
        "dimu_pt": np.linspace(0, 150, nbins),
        "dimu_eta": np.linspace(-5, 5, nbins),
        "mu0_pt": np.linspace(0, 120, nbins),
        "mu1_pt": np.linspace(0, 120, nbins),
        "mu0_eta": np.linspace(-3, 3, nbins),
        "mu1_eta": np.linspace(-3, 3, nbins),
        "mu0_ip3d": np.logspace(-4, 0, nbins),  # 原始代码使用 -6, 0，我们使用 -4, 0（与08_render.ipynb一致）
        "mu1_ip3d": np.logspace(-4, 0, nbins),
        "mu0_iso04": np.linspace(0, 1, nbins),
        "mu1_iso04": np.linspace(0, 1, nbins),
        "mumu_deltapT": np.linspace(0, 100, nbins),
        "mumu_deltaR": np.linspace(0, 1, nbins),
    }
    labels = {
        "dimu_pt": "Dimuon $p_T$ [GeV]",
        "dimu_eta": "Dimuon $\eta$",
        "mu0_pt": "Muon 1 $p_T$ [GeV]",
        "mu1_pt": "Muon 2 $p_T$ [GeV]",
        "mu0_eta": "Muon 1 $\eta$",
        "mu1_eta": "Muon 2 $\eta$",
        "mu0_ip3d": "Harder Mu IP3D [cm]",
        "mu1_ip3d": "Softer Mu IP3D [cm]",
        "mu0_iso04": "Muon 1 iso04",
        "mu1_iso04": "Muon 2 iso04",
        "mumu_deltapT": "Muon $\Delta p_T$ [GeV]",
        "mumu_deltaR": "Muon $\Delta R$",
    }
    
    # 默认 bins 逻辑（兼容性：如果特征不在字典中，使用自动 bins）
    def get_bins_for_feature(feat_name):
        """获取特征的 bins，如果不在字典中则使用默认值"""
        if feat_name in bins:
            return bins[feat_name]
        else:
            # 默认使用自动 bins（基于数据范围）
            logger.warning(f"Feature {feat_name} not in bins dictionary, using auto bins")
            # 尝试从 isolation_data 获取数据范围
            if feat_name in isolation_data:
                data_min = np.min(isolation_data[feat_name])
                data_max = np.max(isolation_data[feat_name])
                # 根据数据范围选择合适的 bins
                if data_min >= 0 and data_max > 10:
                    # 正数，较大范围，使用线性 bins
                    return np.linspace(data_min, data_max * 1.1, nbins)
                elif data_min < 0:
                    # 包含负数，使用线性 bins
                    return np.linspace(data_min * 1.1, data_max * 1.1, nbins)
                else:
                    # 小范围正数，使用 logspace
                    return np.logspace(np.log10(max(data_min, 1e-6)), np.log10(max(data_max, 1e-6)), nbins)
            else:
                # 如果连 isolation_data 都没有，使用默认线性 bins
                return np.linspace(0, 100, nbins)
    
    def get_label_for_feature(feat_name):
        """获取特征的标签，如果不在字典中则使用特征名"""
        if feat_name in labels:
            return labels[feat_name]
        else:
            # 默认使用特征名（首字母大写，下划线替换为空格）
            return feat_name.replace("_", " ").title()
    
    # 使用 newplot 创建符合论文标准的图形（与原始代码完全一致）
    fig, ax = newplot("column", width=9, height=3, subplot_array=(1, n_features))
    
    # 绘制反隔离切割前的数据（与原始代码完全一致）
    for i_feat in range(n_features):
        feat_name = feature_set[i_feat]
        if feat_name not in isolation_data:
            logger.warning(f"Feature {feat_name} not found in isolation_data, skipping")
            continue
        
        feature = isolation_data[feat_name]
        label_string = "Pre Anti-Isolation Cut"
        
        # 使用 get_bins_for_feature 获取 bins（支持兼容性）
        feat_bins = get_bins_for_feature(feat_name)
        
        ax[i_feat].hist(
            feature,
            bins=feat_bins,
            lw=1.0,
            histtype="step",
            color="black",
            label=label_string,
            alpha=0.75,
            ls="--"
        )
    
    # 绘制不同 FPR 阈值下的数据（与原始代码完全一致）
    for t, threshold in enumerate(fpr_thresholds[:num_points]):
        if t % 2 == 1:
            continue
        
        filtered_features = filtered_features_all[t]  # 这是一个字典，键是特征名
        
        for i_feat in range(n_features):
            feat_name = feature_set[i_feat]
            if feat_name not in filtered_features:
                logger.warning(f"Feature {feat_name} not found in filtered_features for FPR {threshold:.3f}, skipping")
                continue  # 跳过缺失的特征
            
            label_string = str(round(100*threshold, 2)) + r"$\%$ FPR"
            
            # 使用 get_bins_for_feature 获取 bins（支持兼容性）
            feat_bins = get_bins_for_feature(feat_name)
            
            # 计算白色填充颜色（与原始代码一致）
            white_color = (np.array([4, 4, 4]) + colors[t]) / 5
            
            # 绘制轮廓（与原始代码一致）
            ax[i_feat].hist(
                filtered_features[feat_name],
                bins=feat_bins,
                lw=1.5,
                histtype="step",
                color=colors[t],
                label=label_string,
                alpha=alphas[t]
            )
            # 绘制填充（与原始代码一致）
            ax[i_feat].hist(
                filtered_features[feat_name],
                bins=feat_bins,
                lw=1.5,
                histtype="stepfilled",
                color=white_color,
                alpha=0.99
            )
            ax[i_feat].set_yscale("log")
            
            # 设置 x 轴（与原始代码一致，根据特征类型自适应）
            # 定义哪些特征使用 log scale（与原始代码的 plot_log 字典一致）
            log_scale_features = ["mu0_ip3d", "mu1_ip3d", "mu0_iso04", "mu1_iso04"]
            
            # 对于 dimu_pt，使用固定范围和线性 scale
            if feat_name == "dimu_pt":
                ax[i_feat].set_xlim(10, 150)
                ax[i_feat].set_xticks([0, 50, 100, 150])
                ax[i_feat].set_xscale("linear")
            # 对于 IP3D，使用 log scale
            elif feat_name in ["mu0_ip3d", "mu1_ip3d"]:
                ax[i_feat].set_xscale("log")
                ax[i_feat].set_xticks([1e-3, 1e-2, 1e-1])
            # 对于 iso04，根据原始代码使用 log scale（虽然范围是 [0, 1]）
            elif feat_name in ["mu0_iso04", "mu1_iso04"]:
                ax[i_feat].set_xscale("log")
                # iso04 的范围是 [0, 1]，但原始代码使用 logscale，我们保持一致
                # 如果数据范围太小，可能需要调整
            # 对于其他特征，根据 bins 类型决定
            else:
                # 如果 bins 是 logspace，使用 log scale
                if np.all(np.diff(feat_bins) > 0) and feat_bins[0] > 0:
                    # 检查是否是 logspace（相邻比值的方差较小）
                    ratios = np.diff(feat_bins) / feat_bins[:-1]
                    if np.std(ratios) < 0.1:  # 近似等比例，可能是 logspace
                        ax[i_feat].set_xscale("log")
                else:
                    ax[i_feat].set_xscale("linear")
            
            # 使用 get_label_for_feature 获取标签（支持兼容性）
            ax[i_feat].set_xlabel(get_label_for_feature(feat_name))
            ax[i_feat].set_ylim(5e-1, 5e4)
            ax[i_feat].tick_params()
            
            if i_feat > 0:
                ax[i_feat].set_yticklabels([])
    
    # 添加文本标签（与原始代码完全一致）
    starting_x = 0.075
    starting_y = 0.955
    delta_y = 0.05
    text_alpha = 0.75
    
    # 原始代码：line0 = r"\textbf{Opposite Sign}: $\mu^+\mu^-$"
    # matplotlib mathtext 不支持 \textbf，使用 $\bf{...}$ 格式
    line0 = r"$\bf{Opposite Sign}$: $\mu^+\mu^-$"
    bin_percent = {8: 2.3, 12: 1.5, 16: 1.1}
    line1 = f"Bin width = {bin_percent[num_bins_SR]}\%"
    line2 = f"Fit Type: {fit_type.capitalize()}"
    line3 = r"Muon Iso_04 $\geq$ 0.55"
    line4 = r"8.7 fb$^{-1}$, $\sqrt{s} = 13$ TeV"
    
    # 在第二个子图上添加文本（与原始代码一致）
    if len(ax) > 1:
        # 原始代码使用 \texttt，但 matplotlib mathtext 不支持，使用 \mathtt 或普通文本
        ax[1].text(
            starting_x, starting_y - (0) * delta_y,
            r"$\mathtt{HLT\_TrkMu15\_DoubleTrkMu5NoFiltersNoVt}$",
            transform=ax[1].transAxes,
            fontsize=7,
            verticalalignment='top',
            alpha=text_alpha,
            zorder=10
        )
        ax[1].text(
            starting_x, starting_y - (1) * delta_y,
            line0,
            transform=ax[1].transAxes,
            fontsize=10,
            verticalalignment='top',
            alpha=text_alpha,
            zorder=10
        )
    
    # 在第三个子图上添加文本（与原始代码一致）
    if len(ax) > 2:
        ax[2].text(
            starting_x, starting_y - 0 * delta_y,
            line1,
            transform=ax[2].transAxes,
            fontsize=10,
            verticalalignment='top',
            alpha=text_alpha,
            zorder=10
        )
        ax[2].text(
            starting_x, starting_y - 1 * delta_y,
            line2,
            transform=ax[2].transAxes,
            fontsize=10,
            verticalalignment='top',
            alpha=text_alpha,
            zorder=10
        )
        ax[2].text(
            starting_x, starting_y - 2 * delta_y,
            line3,
            transform=ax[2].transAxes,
            fontsize=10,
            verticalalignment='top',
            alpha=text_alpha,
            zorder=10
        )
        ax[2].text(
            starting_x, starting_y - 3 * delta_y,
            line4,
            transform=ax[2].transAxes,
            fontsize=10,
            verticalalignment='top',
            alpha=text_alpha,
            zorder=10
        )
    
    # 图例和标签（与原始代码完全一致）
    ax[0].legend(loc="upper right", fontsize=8)
    ax[0].set_ylabel("Events")
    
    # 标题（与原始代码完全一致）
    title = r"$\bf{ 2016\,\, CMS\,\, Open\,\, Data\,\, DoubleMuon}$"
    plt.title(title, fontsize=10, loc="right")
    
    plt.subplots_adjust(wspace=0)
    
    # 原始代码返回 save_data，但我们返回 fig, ax 以便保存
    return fig, ax


def inverse_quantile(sigma):
    """将显著性（sigma）转换为 p-value（与原始代码一致）"""
    return 1 - norm.cdf(sigma)


def plot_sig(
    feature_sigs: Dict[str, np.ndarray],
    random_sigs: np.ndarray,
    fpr_thresholds_finegrained: np.ndarray,
    num_bins_SR: int,
    fit_type: str,
    ymax: float = 10,
    ymin: float = 1e-15,
    bonus: Optional[float] = None,
    line_0: Optional[str] = None,
):
    """
    绘制 p-value vs FPR 图（Significances）
    
    完全复刻原始代码的 plot_sig 函数。
    
    Args:
        feature_sigs: 特征显著性字典 {key: array}，每个 array 是 (n_fpr, ...) 形状
        random_sigs: Random Cut 的显著性数组 (n_fpr, ...)
        fpr_thresholds_finegrained: FPR 阈值数组（finegrained）
        num_bins_SR: SR 的 bin 数量（用于文本标签）
        fit_type: 拟合类型（用于文本标签）
        ymax: y 轴最大值（默认 10）
        ymin: y 轴最小值（默认 1e-15）
        bonus: 额外的显著性值（用于绘制 $\ell$-Reweighting 线）
        line_0: 第一行文本标签
    """
    # 使用 newplot 创建符合论文标准的图形（与原始代码完全一致）
    fig, ax = newplot("column", width=4, height=4)
    
    # 颜色和样式定义（与原始代码完全一致）
    primary_colors = ["red", "orange", "green", "blue"]
    colors = ["lightcoral", "gold", "lime", "cornflowerblue"]
    labels = {
        "CATHODE": r"CATHODE",
        "dimu_pt": "Dimuon $p_T$",
        "mu0_ip3d": "Harder Mu IP3D",
        "mu1_ip3d": "Softer Mu IP3D"
    }
    linestyles = {
        "CATHODE": "-",
        "dimu_pt": "--",
        "mu0_ip3d": "--",
        "mu1_ip3d": "--"
    }
    markersize = {
        "CATHODE": 5,
        "dimu_pt": 0,
        "mu0_ip3d": 0,
        "mu1_ip3d": 0
    }
    
    min_x = 2e-4
    
    # 绘制 Random Cut（与原始代码完全一致）
    SIG_random_observed = random_sigs[:, 0] if random_sigs.ndim > 1 else random_sigs
    p_values = inverse_quantile(SIG_random_observed)
    plt.plot(
        fpr_thresholds_finegrained, p_values,
        color="black", lw=3, alpha=0.75, marker="o", ms=0,
        label="Random Cut", ls="--"
    )
    plt.plot(
        fpr_thresholds_finegrained, p_values,
        color="black", lw=0, alpha=0.99, marker="o", ms=0
    )
    
    # 绘制各个特征（与原始代码完全一致）
    feature_order = ["dimu_pt", "mu1_ip3d", "mu0_ip3d", "CATHODE"]
    for i, key in enumerate(feature_order):
        if key not in feature_sigs:
            continue
        
        sigs = feature_sigs[key]
        SIG_observed = sigs[:, 0] if sigs.ndim > 1 else sigs
        p_values = inverse_quantile(SIG_observed)
        
        ax.plot(
            fpr_thresholds_finegrained, p_values,
            color=primary_colors[i],
            lw=3, alpha=0.75, marker="o", ms=markersize[key],
            label=labels[key], ls=linestyles[key]
        )
        ax.plot(
            fpr_thresholds_finegrained, p_values,
            color=primary_colors[i],
            lw=0, alpha=0.99, marker="o", ms=markersize[key]
        )
        
        # 打印最大值（与原始代码一致）
        max_observed = np.nanmax(SIG_observed)
        print(f"{key}: {max_observed}")
        argmax = np.argmax(SIG_observed)
        print(f"{key}: {fpr_thresholds_finegrained[argmax]}")
        print(f"{key}: {p_values[argmax]}")
    
    # 绘制 bonus（$\ell$-Reweighting）（与原始代码完全一致）
    if bonus is not None:
        p_value = inverse_quantile(bonus)
        plt.axhline(
            p_value,
            color="purple",
            lw=3, alpha=0.75, ms=3,
            label=r"$\ell$-Reweighting"
        )
    
    # 设置标签（与原始代码完全一致）
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("Background-Only $p$-Value")
    plt.yscale("log")
    
    # 文本标签（与原始代码完全一致）
    # 原始代码：line_0 = r"\textbf{Opposite Sign} Muons"（没有 $ 包围）
    # matplotlib mathtext 不支持 \textbf，使用 $\bf{...}$ 格式
    if line_0 is None:
        line_0 = r"$\bf{Opposite Sign}$ Muons"
        if ymax < 10:
            line_0 = r"$\bf{Same Sign}$ Muons"
    else:
        # 如果传入的 line_0 包含 \textbf，需要转换
        if r"\textbf" in line_0:
            import re
            line_0 = re.sub(r'\\textbf\{([^}]+)\}', r'$\bf{\1}$', line_0)
    
    bin_percent = {8: 2.3, 12: 1.5, 16: 1.1}
    line1 = f"Bin width = {bin_percent[num_bins_SR]}\%"
    line2 = f"Fit Type: {fit_type.capitalize()}"
    line3 = r"Muon Iso_04 $\geq$ 0.55"
    line4 = r"8.7 fb$^{-1}$, $\sqrt{s} = 13$ TeV"
    
    starting_x = 0.050
    starting_y = 0.25
    delta_y = 0.05
    text_alpha = 0.75
    
    if line_0 is not None:
        ax.text(
            starting_x, starting_y - (-1.5) * delta_y,
            r"$\mathtt{HLT\_TrkMu15\_DoubleTrkMu5NoFiltersNoVt}$",
            transform=ax.transAxes,
            fontsize=5,
            verticalalignment='top',
            alpha=text_alpha,
            zorder=10
        )
        ax.text(
            starting_x, starting_y - (-1) * delta_y,
            line_0,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            alpha=text_alpha,
            zorder=10
        )
    
    ax.text(
        starting_x, starting_y - 0 * delta_y,
        line1,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        alpha=text_alpha,
        zorder=10
    )
    ax.text(
        starting_x, starting_y - 1 * delta_y,
        line2,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        alpha=text_alpha,
        zorder=10
    )
    ax.text(
        starting_x, starting_y - 2 * delta_y,
        line3,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        alpha=text_alpha,
        zorder=10
    )
    ax.text(
        starting_x, starting_y - 3 * delta_y,
        line4,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        alpha=text_alpha,
        zorder=10
    )
    
    # 图例和标题（与原始代码完全一致）
    legend_title = r"$\bf{2016\,\, CMS\,\, Open\,\, Data\,\, DoubleMuon}$"
    plt.legend(loc="lower right", ncol=1, fontsize=9)
    plt.title(legend_title, loc="right", fontsize=10)
    
    # 垂直参考线（与原始代码完全一致）
    plt.axvline(1e-4, color="grey", linestyle="dashed", alpha=0.5, lw=1)
    
    # 绘制 sigma 参考线（与原始代码完全一致）
    i = 0
    while inverse_quantile(i) > ymin:
        p_value = inverse_quantile(i)
        plt.axhline(p_value, color="grey", linestyle="dashed", alpha=0.5, lw=1)
        
        if i > 0 and inverse_quantile(i + 1) > ymin:
            plt.text(
                3e-4, p_value * 1.5,
                f"{i}$\sigma$",
                fontsize=10,
                verticalalignment="center"
            )
        # 填充区域（与原始代码一致）
        plt.fill_between([min_x, 1], p_value, 0.5, color="grey", alpha=0.025)
        i += 1
    
    plt.xscale("log")
    plt.ylim(ymin, 0.5)
    plt.xlim(min_x, 1)
    
    return fig, ax


def plot_training_losses(
    train_losses: np.ndarray,
    val_losses: np.ndarray,
    yrange: Optional[Tuple[float, float]] = None,
    save_path: Optional[Path] = None,
) -> Tuple:
    """
    绘制训练损失曲线
    
    完全复刻原始代码 helpers/ANODE_training_utils.py 中的 plot_ANODE_losses 函数。
    
    Args:
        train_losses: 训练损失数组 (epochs+1,)
        val_losses: 验证损失数组 (epochs+1,)
        yrange: y轴范围 (可选)
        save_path: 保存路径 (可选)
        
    Returns:
        (fig, ax)
    """
    # 计算5-epoch移动平均（与原始代码完全一致）
    avg_train_losses = (
        train_losses[5:] + train_losses[4:-1] + train_losses[3:-2]
        + train_losses[2:-3] + train_losses[1:-4]
    ) / 5
    avg_val_losses = (
        val_losses[5:] + val_losses[4:-1] + val_losses[3:-2]
        + val_losses[2:-3] + val_losses[1:-4]
    ) / 5
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # 绘制每epoch的值（虚线）
    ax.plot(
        range(1, len(train_losses)), train_losses[1:],
        linestyle=":", color="blue", alpha=0.5
    )
    ax.plot(
        range(1, len(val_losses)), val_losses[1:],
        linestyle=":", color="orange", alpha=0.5
    )
    
    # 绘制5-epoch移动平均（实线）
    ax.plot(
        range(3, len(train_losses) - 2), avg_train_losses,
        label="Training", color="blue", linewidth=2
    )
    ax.plot(
        range(3, len(val_losses) - 2), avg_val_losses,
        label="Validation", color="orange", linewidth=2
    )
    
    # 添加图例说明
    ax.plot(np.nan, np.nan, linestyle="None", label=" ")
    ax.plot(np.nan, np.nan, linestyle=":", color="black", label="Per Epoch Value")
    ax.plot(np.nan, np.nan, linestyle="-", color="black", label="5-Epoch Average")
    
    if yrange is not None:
        ax.set_ylim(*yrange)
    
    ax.set_xlabel("Training Epoch")
    ax.set_ylabel("(Mean) Negative Log Likelihood Loss")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(True, alpha=0.3)
    
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    
    return fig, ax


def plot_roc_curve(
    data: np.ndarray,
    samples: np.ndarray,
    n_runs: int = 3,
    save_path: Optional[Path] = None,
) -> Tuple[float, float]:
    """
    绘制ROC曲线（用于评估flow质量）
    
    参考原始代码 05_eval_cathode.py 中的 run_discriminator 函数。
    使用XGBoost分类器区分真实数据和flow生成的样本。
    如果flow训练良好，AUC应该接近0.5（随机分类器）。
    
    Args:
        data: 真实SB数据 (n_samples, n_features)
        samples: Flow生成的SB样本 (n_samples, n_features)
        n_runs: 运行次数（用于计算平均AUC）
        save_path: 保存路径 (可选)
        
    Returns:
        (auc_mean, auc_std) - 平均AUC和标准差
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, roc_curve
    import xgboost as xgb
    
    # 准备数据
    data_train, data_test = train_test_split(data, test_size=0.1, random_state=42)
    samples_train, samples_test = train_test_split(samples, test_size=0.1, random_state=42)
    
    X_train = np.vstack([data_train, samples_train])
    Y_train = np.hstack([
        np.ones(data_train.shape[0]),
        np.zeros(samples_train.shape[0])
    ])
    
    X_test = np.vstack([data_test, samples_test])
    Y_test = np.hstack([
        np.ones(data_test.shape[0]),
        np.zeros(samples_test.shape[0])
    ])
    
    # 运行多次取平均
    auc_list = []
    fprs_list = []
    tprs_list = []
    
    for i in range(n_runs):
        # 使用XGBoost分类器
        bst = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42 + i,
            eval_metric='logloss',
        )
        
        bst.fit(X_train, Y_train)
        y_pred = bst.predict_proba(X_test)[:, 1]
        
        auc = roc_auc_score(Y_test, y_pred)
        auc_list.append(auc)
        
        fpr, tpr, _ = roc_curve(Y_test, y_pred)
        fprs_list.append(fpr)
        tprs_list.append(tpr)
    
    auc_mean = np.mean(auc_list)
    auc_std = np.std(auc_list)
    
    # 绘制ROC曲线
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # 绘制每次运行的曲线
    for i in range(n_runs):
        ax.plot(
            fprs_list[i], tprs_list[i],
            alpha=0.3, color='blue', linewidth=1
        )
    
    # 绘制平均曲线
    # 使用插值来对齐不同长度的FPR/TPR数组
    from scipy.interpolate import interp1d
    fpr_mean = np.linspace(0, 1, 100)
    tpr_mean = np.zeros_like(fpr_mean)
    successful_runs = 0
    
    for i in range(n_runs):
        fpr_i = fprs_list[i]
        tpr_i = tprs_list[i]
        
        # 检查并处理重复的 FPR 值（避免插值警告）
        # 如果 FPR 中有重复值，对对应的 TPR 取平均
        if len(fpr_i) != len(np.unique(fpr_i)):
            # 使用字典来平均相同 FPR 对应的 TPR
            fpr_tpr_dict = {}
            for f, t in zip(fpr_i, tpr_i):
                if f in fpr_tpr_dict:
                    fpr_tpr_dict[f].append(t)
                else:
                    fpr_tpr_dict[f] = [t]
            
            # 重新构建去重后的数组
            fpr_i_unique = np.array(sorted(fpr_tpr_dict.keys()))
            tpr_i_unique = np.array([np.mean(fpr_tpr_dict[f]) for f in fpr_i_unique])
            
            fpr_i = fpr_i_unique
            tpr_i = tpr_i_unique
        
        # 确保 FPR 是单调递增的（虽然 roc_curve 应该已经保证，但为了安全）
        if not np.all(np.diff(fpr_i) >= 0):
            # 如果非单调，排序
            sort_idx = np.argsort(fpr_i)
            fpr_i = fpr_i[sort_idx]
            tpr_i = tpr_i[sort_idx]
        
        # 确保至少有两个点才能插值
        if len(fpr_i) < 2:
            continue
        
        # 检查是否有足够的唯一值
        if len(fpr_i) < 2 or np.all(fpr_i == fpr_i[0]):
            # 如果所有 FPR 值相同，跳过插值
            continue
        
        try:
            interp_func = interp1d(fpr_i, tpr_i, kind='linear', fill_value='extrapolate', bounds_error=False)
            tpr_interp = interp_func(fpr_mean)
            # 检查插值结果是否有效
            if np.any(np.isfinite(tpr_interp)):
                tpr_mean += tpr_interp
                successful_runs += 1
        except Exception as e:
            # 如果插值失败，跳过这次运行
            logger.warning(f"Interpolation failed for run {i}: {e}")
            continue
    
    # 计算平均值（只考虑成功的运行）
    if successful_runs > 0:
        tpr_mean /= successful_runs
    else:
        # 如果所有插值都失败，使用最后一次运行的原始数据
        if len(fprs_list) > 0 and len(tprs_list) > 0:
            fpr_mean = fprs_list[-1]
            tpr_mean = tprs_list[-1]
    
    ax.plot(
        fpr_mean, tpr_mean,
        color='red', linewidth=2,
        label=f'Mean (AUC = {auc_mean:.3f} ± {auc_std:.3f})'
    )
    
    # 绘制对角线（随机分类器）
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.5)')
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve (Flow Quality Check)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    
    return auc_mean, auc_std


def plot_variations(
    variations_data: Dict[str, np.ndarray],
    fpr_thresholds_finegrained: np.ndarray,
    save_path: Optional[Path] = None,
) -> Tuple:
    """
    绘制不同拟合类型和bin宽度的p值变化图（Figure 4）
    
    完全复刻原始代码的 plot_variations 函数。
    
    Args:
        variations_data: 字典，键为 "{degree}_{num_bins}"，值为显著性数组 (n_fpr,)
            例如: {"3_8": sigs_3_8, "5_8": sigs_5_8, ...}
        fpr_thresholds_finegrained: FPR 阈值数组
        save_path: 保存路径（可选）
    
    Returns:
        (fig, ax) 元组
    """
    fig, ax = newplot("column")
    
    fits = ["cubic", "quintic", "septic"]
    degrees = [3, 5, 7]
    colors = ["red", "purple", "blue"]
    linestyles = ["dashed", "solid", "dotted"]
    bins = [8, 12, 16]
    percentages = [1.1, 1.5, 2.3]
    
    # 使用所有FPR阈值（包括FPR=1.0，与原始代码一致）
    fpr_filtered = fpr_thresholds_finegrained
    
    # 绘制所有变体
    for i, fit in enumerate(fits):
        for j, bin_num in enumerate(bins):
            key = f"{degrees[i]}_{bin_num}"
            if key not in variations_data:
                continue  # 跳过缺失的数据
            
            sigs = variations_data[key]
            # 如果 sigs 是二维数组，取第一列（observed）
            if len(sigs.shape) > 1:
                SIG_observed = sigs[:, 0]
            else:
                SIG_observed = sigs
            
            # 确保长度匹配（应该与fpr_thresholds_finegrained一致）
            if len(SIG_observed) != len(fpr_filtered):
                logger.warning(f"Length mismatch for {key}: sigs={len(SIG_observed)}, fpr={len(fpr_filtered)}")
                # 如果长度不匹配，尝试截断或填充
                if len(SIG_observed) < len(fpr_filtered):
                    # 如果sigs较短，可能是旧数据（跳过了FPR>=0.99），使用对应的FPR
                    if len(SIG_observed) == len(fpr_thresholds_finegrained) - 1:
                        # 旧数据跳过了FPR=1.0，需要过滤FPR
                        mask = fpr_thresholds_finegrained < 1.0
                        fpr_filtered = fpr_thresholds_finegrained[mask]
                    else:
                        continue
                else:
                    # 如果sigs较长，截断
                    SIG_observed = SIG_observed[:len(fpr_filtered)]
            
            p_values = inverse_quantile(SIG_observed)
            
            label = f"Deg. {degrees[i]}, {bin_num} bins"
            
            # 中心选择（Quintic, 12 bins）使用更粗的线
            if i == 1 and j == 1:
                linewidth = 3
            else:
                linewidth = 1.5
            
            ax.plot(
                fpr_filtered,
                p_values,
                color=colors[i],
                alpha=0.99,
                ls=linestyles[j],
                label=label,
                lw=linewidth
            )
    
    # 设置坐标轴
    plt.xscale("log")
    plt.xlabel("False Positive Rate")
    plt.ylabel(r"Background-Only $p$-Value")
    
    # 绘制显著性水平线
    ymin = 1e-20
    for i in range(7 + 1):
        p_value = inverse_quantile(i)
        plt.axhline(p_value, color="grey", linestyle="dashed", alpha=0.5, lw=1)
        
        if i > 0 and inverse_quantile(i + 1) > ymin:
            plt.text(3e-4, p_value * 1.5, f"{i}$\\sigma$", fontsize=10, verticalalignment="center")
        
        # 填充上方区域
        plt.fill_between([2e-4, 1], p_value, 0.5, color="grey", alpha=0.025)
    
    plt.yscale("log")
    
    # 创建图例
    legend_items = []
    # 拟合类型（颜色）
    for i, fit in enumerate(fits):
        legend_item = mpatches.Patch(
            label=fit.capitalize(),
            edgecolor=colors[i],
            facecolor=colors[i],
            alpha=0.5
        )
        legend_items.append(legend_item)
    
    # Bin宽度（线型）
    for j, bin_num in enumerate(bins):
        legend_item = plt.Line2D(
            [0], [0],
            color='black',
            lw=1,
            linestyle=linestyles[j],
            label=f"Binwidth: {percentages[j]}\\%"
        )
        legend_items.append(legend_item)
    
    plt.legend(
        handles=legend_items,
        loc="lower center",
        ncol=2,
        fontsize=9,
        columnspacing=2,
        title="CATHODE",
        title_fontsize=10
    )
    plt.title(r"$\bf{2016\,\, CMS\,\, Open\,\, Data\,\, DoubleMuon}$", loc="right", fontsize=10)
    
    plt.ylim(1e-20, 0.5)
    plt.xlim(2e-4, 1)
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    
    return fig, ax


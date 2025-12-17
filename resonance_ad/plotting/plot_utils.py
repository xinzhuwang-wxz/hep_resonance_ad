"""
绘图工具函数

从原始代码的 helpers/plotting.py 迁移，确保与论文图一致。
参考: https://github.com/rikab/rikabplotlib/blob/main/src/rikabplotlib/plot_utils.py
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Callable


# Constants (from original code)
DPI = 72
FULL_WIDTH_PX = 510
COLUMN_WIDTH_PX = 245

FULL_WIDTH_INCHES = FULL_WIDTH_PX / DPI
COLUMN_WIDTH_INCHES = COLUMN_WIDTH_PX / DPI

GOLDEN_RATIO = 1.618


def newplot(
    scale: Optional[str] = None,
    subplot_array: Optional[Tuple[int, int]] = None,
    width: Optional[float] = None,
    height: Optional[float] = None,
    aspect_ratio: float = 1,
    golden_ratio: bool = False,
    stamp: Optional[str] = None,
    stamp_kwargs: Optional[dict] = None,
    use_tex: bool = False,
    **kwargs
):
    """
    创建符合论文标准的图形
    
    参考原始代码的 newplot 函数。
    
    Args:
        scale: "full" 或 "column"
        subplot_array: (nrows, ncols) 子图数组
        width: 自定义宽度（英寸）
        height: 自定义高度（英寸）
        aspect_ratio: 宽高比
        golden_ratio: 是否使用黄金比例
        stamp: 右上角文本
        stamp_kwargs: stamp 文本的参数
        use_tex: 是否使用 LaTeX（需要系统支持）
        **kwargs: 传递给 plt.subplots 的其他参数
    """
    # Determine plot aspect ratio
    if golden_ratio:
        aspect_ratio = GOLDEN_RATIO

    # Determine plot size if not directly set
    if scale is None:
        plot_scale = "full"
    else:
        plot_scale = scale
        
    if plot_scale == "full":
        fig_width = FULL_WIDTH_INCHES / aspect_ratio
        fig_height = FULL_WIDTH_INCHES
    elif plot_scale == "column":
        fig_width = COLUMN_WIDTH_INCHES / aspect_ratio
        fig_height = COLUMN_WIDTH_INCHES
    else:
        raise ValueError("Invalid scale argument. Must be 'full' or 'column'.")

    if width is not None:
        fig_width = width
    if height is not None:
        fig_height = height

    if subplot_array is not None:
        fig, ax = plt.subplots(
            subplot_array[0], subplot_array[1],
            figsize=(fig_width, fig_height), **kwargs
        )
        stamp_kwargs_default = {
            "style": 'italic',
            "horizontalalignment": 'right',
            "verticalalignment": 'bottom',
            "transform": ax[0].transAxes if subplot_array[0] * subplot_array[1] > 1 else ax.transAxes
        }
    else:
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), **kwargs)
        stamp_kwargs_default = {
            "style": 'italic',
            "horizontalalignment": 'right',
            "verticalalignment": 'bottom',
            "transform": ax.transAxes
        }

    # Plot title/stamp
    if stamp_kwargs is not None:
        stamp_kwargs_default.update(stamp_kwargs)

    if stamp is not None:
        plt.text(1, 1, stamp, **stamp_kwargs_default)

    return fig, ax


def hist_with_outline(
    ax,
    points: np.ndarray,
    bins: np.ndarray,
    range: Tuple[float, float],
    weights: Optional[np.ndarray] = None,
    color: str = "purple",
    alpha_1: float = 0.25,
    alpha_2: float = 0.75,
    label: Optional[str] = None,
    **kwargs
):
    """
    绘制带轮廓的直方图（填充+轮廓）
    
    参考原始代码的 hist_with_outline 函数。
    先绘制填充的直方图（alpha_1），再绘制轮廓（alpha_2）。
    与原始代码完全一致。
    
    Args:
        ax: matplotlib axes
        points: 数据点
        bins: bin 边界数组
        range: (min, max) 范围
        weights: 权重数组
        color: 颜色
        alpha_1: 填充透明度
        alpha_2: 轮廓透明度
        label: 标签
        **kwargs: 传递给 hist 的其他参数（如 lw）
    """
    if weights is None:
        weights = np.ones_like(points)

    # 填充的直方图（与原始代码完全一致）
    ax.hist(
        points,
        bins=bins,
        range=range,
        weights=weights,
        color=color,
        alpha=alpha_1,
        histtype='stepfilled',
        **kwargs
    )
    
    # 轮廓（与原始代码完全一致）
    ax.hist(
        points,
        bins=bins,
        range=range,
        weights=weights,
        color=color,
        alpha=alpha_2,
        histtype='step',
        label=label,
        **kwargs
    )


def hist_with_errors(
    ax,
    points: np.ndarray,
    bins: int,
    range: Tuple[float, float],
    weights: Optional[np.ndarray] = None,
    show_zero: bool = False,
    show_errors: bool = True,
    label: Optional[str] = None,
    **kwargs
):
    """
    绘制带误差条的直方图
    
    参考原始代码的 hist_with_errors 函数。
    使用 sqrt(N) 误差，可归一化到单位面积。
    
    Args:
        ax: matplotlib axes
        points: 数据点
        bins: bin 数量或边界数组
        range: (min, max) 范围
        weights: 权重数组
        show_zero: 是否显示零值
        show_errors: 是否显示误差条
        label: 标签
        **kwargs: 传递给 errorbar 的其他参数
    """
    if weights is None:
        weights = np.ones_like(points)

    hist, bin_edges = np.histogram(points, bins=bins, range=range, weights=weights)
    errs2 = np.histogram(points, bins=bins, range=range, weights=weights**2)[0]

    # Check if density is a keyword argument
    density = kwargs.pop("density", False)

    if density:
        bin_widths = (bin_edges[1:] - bin_edges[:-1])
        errs2 = errs2 / (np.sum(hist * bin_widths))
        hist = hist / np.sum(hist * bin_widths)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths = (bin_edges[1:] - bin_edges[:-1])

    if show_errors:
        mask = hist > 0 if not show_zero else np.ones_like(hist, dtype=bool)
        ax.errorbar(
            bin_centers[mask],
            hist[mask],
            np.sqrt(errs2[mask]),
            xerr=bin_widths[mask] / 2,
            fmt="o",
            label=label,
            **kwargs
        )
    else:
        mask = hist > 0 if not show_zero else np.ones_like(hist, dtype=bool)
        ax.scatter(bin_centers[mask], hist[mask], label=label, **kwargs)


def function_with_band(
    ax,
    f: Callable,
    range: Tuple[float, float],
    params: np.ndarray,
    pcov: Optional[np.ndarray] = None,
    color: str = "purple",
    alpha_line: float = 0.75,
    alpha_band: float = 0.25,
    lw: float = 3,
    **kwargs
):
    """
    绘制函数曲线和误差带
    
    参考原始代码的 function_with_band 函数。
    如果提供了协方差矩阵，会绘制误差带。
    与原始代码完全一致。
    
    Args:
        ax: matplotlib axes
        f: 函数 f(x, *params)
        range: (xmin, xmax) x 范围
        params: 函数参数
        pcov: 参数协方差矩阵（可选）
        color: 颜色
        alpha_line: 线条透明度
        alpha_band: 误差带透明度
        lw: 线宽
        **kwargs: 传递给 plot 的其他参数（如 linestyle）
    """
    x = np.linspace(range[0], range[1], 1000)

    if pcov is not None:
        # Vary the parameters within their errors（与原始代码一致）
        n = 1000
        try:
            temp_params = np.random.multivariate_normal(params, pcov, n)
            y = np.array([f(x, *p) for p in temp_params])

            # Plot the band（与原始代码一致）
            y_mean = np.mean(y, axis=0)
            y_std = np.std(y, axis=0)

            ax.fill_between(
                x, y_mean - y_std, y_mean + y_std,
                color=color, alpha=alpha_band, **kwargs
            )
        except Exception:
            # 如果协方差矩阵有问题，只绘制中心线
            pass

    # 绘制中心线（与原始代码一致）
    y = f(x, *params)
    ax.plot(x, y, color=color, alpha=alpha_line, lw=lw, **kwargs)


def stamp(
    left_x: float,
    top_y: float,
    ax=None,
    delta_y: float = 0.06,
    textops_update: Optional[dict] = None,
    boldfirst: bool = True,
    **kwargs
):
    """
    在图上添加文本标签
    
    参考原始代码的 stamp 函数。
    逐行添加文本，第一行可以加粗。
    
    Args:
        left_x: 左侧 x 位置（相对坐标）
        top_y: 顶部 y 位置（相对坐标）
        ax: matplotlib axes（如果为 None，使用当前 axes）
        delta_y: 行间距
        textops_update: 文本选项更新
        boldfirst: 第一行是否加粗
        **kwargs: 文本行，格式为 line_0, line_1, ...
    """
    # handle default axis
    if ax is None:
        ax = plt.gca()

    # text options
    textops = {
        'horizontalalignment': 'left',
        'verticalalignment': 'center',
        'transform': ax.transAxes
    }
    if isinstance(textops_update, dict):
        textops.update(textops_update)

    # add text line by line
    i = 0
    while f'line_{i}' in kwargs:
        y = top_y - i * delta_y
        t = kwargs.get(f'line_{i}')

        if t is not None:
            if boldfirst and i == 0:
                ax.text(left_x, y, r"$\textbf{%s}$" % t, weight='bold', **textops)
            else:
                ax.text(left_x, y, t, **textops)
        i += 1


def add_whitespace(ax=None, upper_fraction: float = 1.333, lower_fraction: float = 1):
    """
    在 y 轴添加空白空间
    
    参考原始代码的 add_whitespace 函数。
    
    Args:
        ax: matplotlib axes（如果为 None，使用当前 axes）
        upper_fraction: 上方空白倍数
        lower_fraction: 下方空白倍数
    """
    # handle default axis
    if ax is None:
        ax = plt.gca()

    # check if log scale
    scale_str = ax.get_yaxis().get_scale()

    bottom, top = ax.get_ylim()

    if scale_str == "log":
        upper_fraction = np.power(10, upper_fraction - 1)
        lower_fraction = np.power(10, lower_fraction - 1)

    ax.set_ylim([bottom / lower_fraction, top * upper_fraction])


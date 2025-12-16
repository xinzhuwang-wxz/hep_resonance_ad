"""
论文图生成模块

所有论文中的图都应该通过这个模块生成。
参考原始仓库的 helpers/plotting.py，但重新组织为清晰的接口。
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import norm

from resonance_ad.core.logging import get_logger

logger = get_logger(__name__)


# 默认的特征 bins 和 labels（参考原始仓库的 helpers/plotting.py）
DEFAULT_FEATURE_BINS = {
    "mu0_ip3d": np.linspace(0, 0.25, 60),
    "mu1_ip3d": np.linspace(0, 0.25, 60),
    "mu0_jetiso": np.linspace(0, 8, 60),
    "mu1_jetiso": np.linspace(0, 8, 60),
    "mu0_pt": np.linspace(0, 50, 60),
    "mu1_pt": np.linspace(0, 50, 60),
    "mu0_eta": np.linspace(-3, 3, 60),
    "mu1_eta": np.linspace(-3, 3, 60),
    "mu0_phi": np.linspace(-3.2, 3.2, 60),
    "mu1_phi": np.linspace(-3.2, 3.2, 60),
    "mu0_iso03": np.linspace(0, 1, 60),
    "mu1_iso03": np.linspace(0, 1, 60),
    "mu0_iso04": np.linspace(0, 1, 60),
    "mu1_iso04": np.linspace(0, 1, 60),
    "dimu_pt": np.linspace(0, 150, 60),
    "dimu_eta": np.linspace(-6, 6, 60),
    "dimu_phi": np.linspace(-3.2, 3.2, 60),
    "dimu_mass": np.linspace(0, 120, 60),
    "n_electrons": np.linspace(0, 10, 11),
    "n_muons": np.linspace(0, 10, 11),
    "n_jets": np.linspace(0, 10, 11),
    "mumu_deltaR": np.linspace(0, 2, 60),
    "mumu_deltapT": np.linspace(0, 100, 60),
    "dimujet_deltaR": np.linspace(0, 2, 60),
}

DEFAULT_FEATURE_LABELS = {
    "mu0_pt": r"$\mu_0$ $p_T$",
    "mu1_pt": r"$\mu_1$ $p_T$",
    "mu0_eta": r"$\mu_0$ $\eta$",
    "mu1_eta": r"$\mu_1$ $\eta$",
    "mu0_phi": r"$\mu_0$ $\phi$",
    "mu1_phi": r"$\mu_1$ $\phi$",
    "mu0_ip3d": r"$\mu_0$ IP3D",
    "mu1_ip3d": r"$\mu_1$ IP3D",
    "mu0_jetiso": r"$\mu_0$ jetISO",
    "mu1_jetiso": r"$\mu_1$ jetISO",
    "mu0_iso03": r"$\mu_0$ isoR03",
    "mu1_iso03": r"$\mu_1$ isoR03",
    "mu0_iso04": r"$\mu_0$ isoR04",
    "mu1_iso04": r"$\mu_1$ isoR04",
    "dimu_pt": r"Dimu $p_T$",
    "dimu_eta": r"Dimu $\eta$",
    "dimu_phi": r"Dimu $\phi$",
    "dimu_mass": r"Dimu $M$",
    "n_electrons": "Num. electrons",
    "n_muons": "Num. muons",
    "n_jets": "Num. Jets",
    "mumu_deltaR": r"$\mu\mu$ $\Delta R$",
    "mumu_deltapT": r"$\mu\mu$ $\Delta p_T$",
    "dimujet_deltaR": r"$\mu\mu$ Jet $\Delta R$",
}


class PaperFigureGenerator:
    """
    论文图生成器
    
    负责生成论文中的所有关键图：
    - Mass spectrum（带 sideband 和 signal region）
    - Anomaly score 分布
    - Significance 图
    - Score vs mass 散点图
    等
    """
    
    def __init__(self, config, output_dir: Optional[Path] = None):
        """
        初始化图生成器
        
        Args:
            config: Config 对象
            output_dir: 输出目录
        """
        self.config = config
        self.output_dir = output_dir or (config.output_dir / "figures")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(__name__)
        
        # 设置 matplotlib 样式
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = (8, 6)
        plt.rcParams['font.size'] = 12
    
    def plot_mass_spectrum(
        self,
        region_data: Dict[str, Dict[str, np.ndarray]],
        bins: np.ndarray,
        window: Dict[str, float],
        save_path: Optional[Path] = None,
        show: bool = False,
    ):
        """
        绘制质量谱
        
        显示 sideband 和 signal region 的质量分布。
        
        Args:
            region_data: 区域数据字典 {band: {feature: array}}
            bins: bin 边界数组
            window: 窗口定义字典
            save_path: 保存路径
            show: 是否显示图
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 绘制各个区域
        colors = {"SBL": "green", "SR": "red", "SBH": "purple"}
        labels = {"SBL": "Sideband Left", "SR": "Signal Region", "SBH": "Sideband High"}
        
        for band in ["SBL", "SR", "SBH"]:
            if band in region_data and "dimu_mass" in region_data[band]:
                masses = region_data[band]["dimu_mass"]
                ax.hist(
                    masses,
                    bins=bins,
                    histtype="step",
                    color=colors.get(band, "blue"),
                    label=labels.get(band, band),
                    linewidth=2,
                    density=False,
                )
        
        # 标记窗口边界
        ax.axvline(window["SR_left"], color="red", linestyle="--", alpha=0.5, label="SR boundaries")
        ax.axvline(window["SR_right"], color="red", linestyle="--", alpha=0.5)
        
        ax.set_xlabel("Dimuon Invariant Mass [GeV]", fontsize=14)
        ax.set_ylabel("Events", fontsize=14)
        ax.set_title("Mass Spectrum", fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            self.logger.info(f"Saved mass spectrum to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_anomaly_score_distribution(
        self,
        scores: Dict[str, np.ndarray],
        bins: int = 50,
        save_path: Optional[Path] = None,
        show: bool = False,
    ):
        """
        绘制 anomaly score 分布
        
        比较不同区域的 score 分布。
        
        Args:
            scores: {region: score_array} 字典
            bins: bin 数量
            save_path: 保存路径
            show: 是否显示图
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = {"SBL": "green", "SR": "red", "SBH": "purple", "SB": "blue"}
        labels = {
            "SBL": "Sideband Left",
            "SR": "Signal Region",
            "SBH": "Sideband High",
            "SB": "Sideband",
        }
        
        for region, score_array in scores.items():
            if len(score_array) > 0:
                ax.hist(
                    score_array,
                    bins=bins,
                    histtype="step",
                    color=colors.get(region, "blue"),
                    label=f"{labels.get(region, region)} (mean={score_array.mean():.2f})",
                    linewidth=2,
                    density=True,
                    alpha=0.7,
                )
        
        ax.set_xlabel("Anomaly Score", fontsize=14)
        ax.set_ylabel("Density", fontsize=14)
        ax.set_title("Anomaly Score Distribution", fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            self.logger.info(f"Saved score distribution to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_significance(
        self,
        mass_centers: np.ndarray,
        significances: np.ndarray,
        bins: np.ndarray,
        save_path: Optional[Path] = None,
        show: bool = False,
    ):
        """
        绘制显著性图
        
        显示每个质量 bin 的显著性。
        
        Args:
            mass_centers: 质量 bin 中心
            significances: 显著性数组
            bins: bin 边界
            save_path: 保存路径
            show: 是否显示图
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 绘制显著性
        ax.step(mass_centers, significances, where="mid", linewidth=2, color="blue")
        ax.fill_between(mass_centers, 0, significances, alpha=0.3, step="mid")
        
        # 标记显著性阈值
        ax.axhline(3, color="orange", linestyle="--", label="3σ threshold", alpha=0.7)
        ax.axhline(5, color="red", linestyle="--", label="5σ threshold", alpha=0.7)
        
        ax.set_xlabel("Dimuon Invariant Mass [GeV]", fontsize=14)
        ax.set_ylabel("Significance [σ]", fontsize=14)
        ax.set_title("Bump Hunt Significance", fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # 标注最大显著性
        max_idx = np.argmax(significances)
        max_sig = significances[max_idx]
        max_mass = mass_centers[max_idx]
        ax.plot(max_mass, max_sig, "ro", markersize=10)
        ax.annotate(
            f"Max: {max_sig:.2f}σ @ {max_mass:.2f} GeV",
            xy=(max_mass, max_sig),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=12,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            self.logger.info(f"Saved significance plot to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_score_vs_mass(
        self,
        mass: np.ndarray,
        score: np.ndarray,
        bins_mass: int = 50,
        bins_score: int = 50,
        region: Optional[str] = None,
        save_path: Optional[Path] = None,
        show: bool = False,
    ):
        """
        绘制 score vs mass 2D 直方图
        
        Args:
            mass: 质量数组
            score: score 数组
            bins_mass: 质量方向的 bin 数量
            bins_score: score 方向的 bin 数量
            region: 区域标签
            save_path: 保存路径
            show: 是否显示图
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 2D 直方图
        h, xedges, yedges = np.histogram2d(mass, score, bins=[bins_mass, bins_score])
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        
        im = ax.imshow(
            h.T,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="viridis",
            interpolation="nearest",
        )
        
        ax.set_xlabel("Dimuon Invariant Mass [GeV]", fontsize=14)
        ax.set_ylabel("Anomaly Score", fontsize=14)
        title = "Anomaly Score vs Mass"
        if region:
            title += f" ({region})"
        ax.set_title(title, fontsize=16)
        
        plt.colorbar(im, ax=ax, label="Events")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            self.logger.info(f"Saved score vs mass plot to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_background_fit(
        self,
        masses: np.ndarray,
        bins: np.ndarray,
        fit_params: np.ndarray,
        fit_function,
        window: Dict[str, float],
        save_path: Optional[Path] = None,
        show: bool = False,
    ):
        """
        绘制背景拟合结果
        
        Args:
            masses: 质量数组
            bins: bin 边界
            fit_params: 拟合参数
            fit_function: 拟合函数
            window: 窗口定义
            save_path: 保存路径
            show: 是否显示图
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 绘制数据
        counts, bin_edges, _ = ax.hist(
            masses, bins=bins, histtype="step", color="blue", label="Sideband data", linewidth=2
        )
        
        # 绘制拟合曲线
        centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        x_fit = np.linspace(bins[0], bins[-1], 200)
        y_fit = fit_function(x_fit, *fit_params)
        
        ax.plot(x_fit, y_fit, "r-", linewidth=2, label="Background fit")
        
        # 标记窗口
        ax.axvline(window["SR_left"], color="red", linestyle="--", alpha=0.5, label="SR")
        ax.axvline(window["SR_right"], color="red", linestyle="--", alpha=0.5)
        
        ax.set_xlabel("Dimuon Invariant Mass [GeV]", fontsize=14)
        ax.set_ylabel("Events", fontsize=14)
        ax.set_title("Background Fit", fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            self.logger.info(f"Saved background fit to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_feature_distributions(
        self,
        data_dicts: List[Dict[str, np.ndarray]],
        data_labels: List[str],
        feature_set: List[str],
        kwargs_dict: Dict[str, Dict],
        feature_bins: Optional[Dict[str, np.ndarray]] = None,
        feature_labels: Optional[Dict[str, str]] = None,
        scaled_features: bool = False,
        plot_bound: float = 3,
        save_path: Optional[Path] = None,
        yscale_log: bool = False,
        nice_labels: bool = True,
        show: bool = False,
    ):
        """
        绘制所有特征的分布图
        
        参考原始仓库的 hist_all_features_dict() 函数。
        为每个特征生成一个直方图，比较不同数据集的分布。
        
        Args:
            data_dicts: 数据字典列表，每个字典包含 {feature: array}
            data_labels: 数据集标签列表
            feature_set: 要绘制的特征列表
            feature_bins: 每个特征的 bin 定义 {feature: bins_array}
            feature_labels: 每个特征的显示标签 {feature: label}
            kwargs_dict: 绘图参数字典 {label: {plot_kwargs}}
            scaled_features: 是否使用缩放后的特征（使用 plot_bound）
            plot_bound: 缩放特征的边界（如果 scaled_features=True）
            save_path: 保存路径（如果提供，会保存为 PDF）
            yscale_log: 是否使用对数 y 轴
            nice_labels: 是否使用美观的标签
            show: 是否显示图
        """
        # 使用默认值如果没有提供
        if feature_bins is None:
            feature_bins = DEFAULT_FEATURE_BINS
        if feature_labels is None:
            feature_labels = DEFAULT_FEATURE_LABELS
        
        n_bins = 60
        scaled_feature_bins = [np.linspace(-plot_bound, plot_bound, n_bins) for _ in range(len(feature_set))]
        
        if save_path:
            p = PdfPages(f"{save_path}.pdf")
        
        for i, feat in enumerate(feature_set):
            fig = plt.figure(figsize=(5, 3))
            
            for j, data_dict in enumerate(data_dicts):
                if feat not in data_dict:
                    continue
                    
                if scaled_features:
                    bins_to_use = scaled_feature_bins[i]
                else:
                    bins_to_use = feature_bins.get(feat, np.linspace(0, 1, n_bins))
                
                plt.hist(
                    data_dict[feat],
                    bins=bins_to_use,
                    **kwargs_dict.get(data_labels[j], {})
                )
            
            if yscale_log:
                plt.yscale("log")
            
            if nice_labels:
                plt.xlabel(feature_labels.get(feat, feat))
            else:
                plt.xlabel(feat)
            
            plt.legend()
            plt.ylabel("Density")
            plt.tight_layout()
            
            if save_path:
                fig.savefig(p, format='pdf')
            
            if show:
                plt.show()
            else:
                plt.close()
        
        if save_path:
            p.close()
            self.logger.info(f"Saved feature distributions to {save_path}.pdf")
    
    def plot_significance_variations(
        self,
        significance_results: Dict[str, Dict],
        fpr_thresholds: np.ndarray,
        fit_degrees: List[int] = [3, 5, 7],
        num_bins_list: List[int] = [8, 12, 16],
        save_path: Optional[Path] = None,
        ymax: float = 10,
        ymin: float = 1e-15,
        show: bool = False,
    ):
        """
        绘制不同配置下的显著性变化图
        
        参考原始仓库的 plot_variations() 函数。
        比较不同 bin 数和拟合阶数组合的显著性结果。
        
        Args:
            significance_results: 显著性结果字典
                {f"{degree}_{bins}": {"significances": array, "fpr_thresholds": array}}
            fpr_thresholds: FPR 阈值数组（finegrained）
            fit_degrees: 拟合阶数列表
            num_bins_list: bin 数量列表
            save_path: 保存路径
            ymax: y 轴最大值
            ymin: y 轴最小值
            show: 是否显示图
        """
        def inverse_quantile(sigma):
            """将显著性（sigma）转换为 p-value"""
            return 1 - norm.cdf(sigma)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        colors = ["red", "purple", "blue"]
        linestyles = ["dashed", "solid", "dotted"]
        
        for i, degree in enumerate(fit_degrees):
            for j, num_bins in enumerate(num_bins_list):
                key = f"{degree}_{num_bins}"
                
                if key not in significance_results:
                    self.logger.warning(f"Missing significance results for {key}")
                    continue
                
                sigs = significance_results[key]["significances"]
                if len(sigs) == 0:
                    continue
                
                # 取观测到的显著性（第一列）
                if sigs.ndim > 1:
                    SIG_observed = sigs[:, 0]
                else:
                    SIG_observed = sigs
                
                p_values = inverse_quantile(SIG_observed)
                
                label = f"Deg. {degree}, {num_bins} bins"
                
                # 主配置（quintic, 12 bins）用粗线
                if i == 1 and j == 1:
                    linewidth = 3
                else:
                    linewidth = 1.5
                
                ax.plot(
                    fpr_thresholds,
                    p_values,
                    color=colors[i],
                    alpha=0.99,
                    ls=linestyles[j],
                    label=label,
                    lw=linewidth
                )
        
        # 添加显著性参考线
        for i in range(8):
            p_value = inverse_quantile(i)
            ax.axhline(
                p_value,
                color="grey",
                linestyle="dashed",
                alpha=0.5,
                lw=1
            )
            if i > 0 and inverse_quantile(i + 1) > ymin:
                ax.text(
                    3e-4,
                    p_value * 1.5,
                    f"{i}$\\sigma$",
                    fontsize=10,
                    verticalalignment="center"
                )
            # 填充区域
            ax.fill_between(
                [2e-4, 1],
                p_value,
                0.5,
                color="grey",
                alpha=0.025
            )
        
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("False Positive Rate", fontsize=14)
        ax.set_ylabel(r"Background-Only $p$-Value", fontsize=14)
        ax.set_title("Significance Variations", fontsize=16)
        ax.set_ylim(ymin, ymax)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            self.logger.info(f"Saved significance variations to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_roc_curve(
        self,
        data: np.ndarray,
        samples: np.ndarray,
        classifier=None,
        n_runs: int = 3,
        save_path: Optional[Path] = None,
        show: bool = False,
    ):
        """
        绘制 ROC 曲线用于评估 flow 质量
        
        参考原始仓库的 05_eval_cathode.py 中的 run_discriminator() 函数。
        使用分类器（如 XGBoost）来区分 SB data 和 SB samples。
        如果 flow 训练良好，ROC AUC 应该接近 0.5（随机分类器）。
        
        Args:
            data: SB 数据数组 (n_samples, n_features)
            samples: SB samples 数组 (n_samples, n_features)
            classifier: 分类器对象（如果为 None，会使用 XGBoost）
            n_runs: 运行次数（用于平均）
            save_path: 保存路径
            show: 是否显示图
        """
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score, roc_curve
        from resonance_ad.data.preprocessor import clean_data
        
        # 如果没有提供分类器，使用 XGBoost
        if classifier is None:
            try:
                import xgboost as xgb
                use_xgb = True
            except ImportError:
                self.logger.warning("XGBoost not available, using sklearn RandomForestClassifier")
                from sklearn.ensemble import RandomForestClassifier
                use_xgb = False
        
        # 分割数据
        data_train, data_test = train_test_split(data, test_size=0.1, random_state=42)
        samples_train, samples_test = train_test_split(samples, test_size=0.1, random_state=42)
        
        # 清理数据
        samples_train = clean_data(samples_train)
        samples_test = clean_data(samples_test)
        
        # 准备训练数据
        X_train = np.vstack([data_train, samples_train])
        Y_train = np.hstack([np.ones(len(data_train)), np.zeros(len(samples_train))])
        
        X_test = np.vstack([data_test, samples_test])
        Y_test = np.hstack([np.ones(len(data_test)), np.zeros(len(samples_test))])
        
        auc_list = []
        fpr_list = []
        tpr_list = []
        
        for i in range(n_runs):
            if use_xgb:
                # 使用 XGBoost（参考原始代码）
                bst = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.1,
                    subsample=0.8,
                    early_stopping_rounds=10,
                    objective='binary:logistic',
                    random_state=i,
                    eval_metric="logloss"
                )
                bst.fit(
                    X_train, Y_train,
                    eval_set=[(X_train, Y_train), (X_test, Y_test)],
                    verbose=False
                )
                scores = bst.predict_proba(X_test, iteration_range=(0, bst.best_iteration))[:, 1]
            else:
                # 使用 RandomForest
                clf = RandomForestClassifier(n_estimators=100, random_state=i)
                clf.fit(X_train, Y_train)
                scores = clf.predict_proba(X_test)[:, 1]
            
            # 计算 ROC
            fpr, tpr, _ = roc_curve(Y_test, scores)
            auc = roc_auc_score(Y_test, scores)
            
            auc_list.append(auc)
            fpr_list.append(fpr)
            tpr_list.append(tpr)
        
        # 绘制 ROC 曲线
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 绘制每次运行的曲线
        for i in range(n_runs):
            ax.plot(
                fpr_list[i],
                tpr_list[i],
                alpha=0.3,
                color="blue",
                linewidth=1,
                label=f"Run {i+1} (AUC={auc_list[i]:.3f})" if i == 0 else None
            )
        
        # 绘制平均曲线（使用最后一次运行作为代表）
        mean_auc = np.mean(auc_list)
        std_auc = np.std(auc_list)
        ax.plot(
            fpr_list[-1],
            tpr_list[-1],
            color="blue",
            linewidth=2,
            label=f"Mean AUC = {mean_auc:.3f} ± {std_auc:.3f}"
        )
        
        # 绘制随机分类器线
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label="Random (AUC=0.5)")
        
        ax.set_xlabel("False Positive Rate", fontsize=14)
        ax.set_ylabel("True Positive Rate", fontsize=14)
        ax.set_title("ROC Curve: SB Data vs SB Samples", fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        # 添加文本说明
        if mean_auc < 0.55:
            quality_text = "Good flow quality (AUC ≈ 0.5)"
            text_color = "green"
        elif mean_auc < 0.7:
            quality_text = "Moderate flow quality"
            text_color = "orange"
        else:
            quality_text = "Poor flow quality (AUC >> 0.5)"
            text_color = "red"
        
        ax.text(
            0.6, 0.2,
            quality_text,
            fontsize=12,
            color=text_color,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            self.logger.info(f"Saved ROC curve to {save_path} (AUC={mean_auc:.3f}±{std_auc:.3f})")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return mean_auc, std_auc
    
    def plot_training_losses(
        self,
        train_losses: np.ndarray,
        val_losses: np.ndarray,
        yrange: Optional[Tuple[float, float]] = None,
        save_path: Optional[Path] = None,
        show: bool = False,
    ):
        """
        绘制训练损失曲线
        
        参考原始仓库的 helpers/ANODE_training_utils.py 中的 plot_ANODE_losses() 函数。
        显示每 epoch 的训练和验证损失，以及 5-epoch 移动平均。
        
        Args:
            train_losses: 训练损失数组
            val_losses: 验证损失数组
            yrange: y 轴范围 (ymin, ymax)
            save_path: 保存路径
            show: 是否显示图
        """
        # 计算 5-epoch 移动平均
        if len(train_losses) > 5:
            avg_train_losses = (
                train_losses[5:] + train_losses[4:-1] + train_losses[3:-2]
                + train_losses[2:-3] + train_losses[1:-4]
            ) / 5
        else:
            avg_train_losses = train_losses[1:]
        
        if len(val_losses) > 5:
            avg_val_losses = (
                val_losses[5:] + val_losses[4:-1] + val_losses[3:-2]
                + val_losses[2:-3] + val_losses[1:-4]
            ) / 5
        else:
            avg_val_losses = val_losses[1:]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 绘制每 epoch 的值（虚线）
        epochs = range(1, len(train_losses))
        ax.plot(epochs, train_losses[1:], linestyle=":", color="blue", alpha=0.5)
        ax.plot(epochs, val_losses[1:], linestyle=":", color="orange", alpha=0.5)
        
        # 绘制移动平均（实线）
        if len(avg_train_losses) > 0:
            avg_epochs_train = range(3, len(train_losses) - 2)
            ax.plot(avg_epochs_train, avg_train_losses, label="Training", color="blue", linewidth=2)
        
        if len(avg_val_losses) > 0:
            avg_epochs_val = range(3, len(val_losses) - 2)
            ax.plot(avg_epochs_val, avg_val_losses, label="Validation", color="orange", linewidth=2)
        
        # 添加图例说明
        ax.plot([], [], linestyle=":", color="black", label="Per Epoch Value")
        ax.plot([], [], linestyle="-", color="black", label="5-Epoch Average")
        
        if yrange is not None:
            ax.set_ylim(*yrange)
        
        ax.set_xlabel("Training Epoch", fontsize=14)
        ax.set_ylabel("(Mean) Negative Log Likelihood Loss", fontsize=14)
        ax.set_title("Training History", fontsize=16)
        ax.legend(loc="upper right", fontsize=12, frameon=False)
        ax.grid(True, alpha=0.3)
        
        # 标注最佳 epoch（验证损失最低）
        if len(val_losses) > 1:
            best_epoch = np.argmin(val_losses[1:]) + 1
            best_val_loss = val_losses[best_epoch]
            ax.plot(best_epoch, best_val_loss, "ro", markersize=10)
            ax.annotate(
                f"Best: Epoch {best_epoch}\nVal Loss: {best_val_loss:.4f}",
                xy=(best_epoch, best_val_loss),
                xytext=(10, 10),
                textcoords="offset points",
                fontsize=11,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            self.logger.info(f"Saved training losses to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()

"""
生成论文图

完全复刻原始代码 08_render.ipynb 的绘图流程。
生成三种主要图：
1. histogram_{fit_type}_{num_bins_SR}.pdf - Cut Histograms
2. features_{fit_type}_{num_bins_SR}.pdf - Feature Histograms
3. significance_{fit_type}_{num_bins_SR}.pdf - Significances
"""

import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import logging

from resonance_ad.core.config import load_config
from resonance_ad.core.logging import get_logger
from resonance_ad.plotting.figure import (
    plot_histograms_with_fits,
    plot_features,
    plot_sig,
    plot_training_losses,
    plot_roc_curve,
    plot_variations,
)
from resonance_ad.analysis.bump_hunt import (
    bkg_fit_cubic,
    bkg_fit_quintic,
    bkg_fit_septic,
)
from resonance_ad.models import DensityEstimator
from resonance_ad.data import assemble_banded_datasets
from resonance_ad.data.preprocessor import DataPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--evaluation-results",
        "--eval-results",  # 添加别名以支持 --eval-results
        type=str,
        dest="evaluation_results",
        help="Path to evaluation results pickle file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for figures (default: outputs/{dataset_id}/figures)",
    )
    parser.add_argument(
        "--fit-type",
        type=str,
        default="quintic",
        choices=["cubic", "quintic", "septic"],
        help="Fit type (default: quintic)",
    )
    parser.add_argument(
        "--num-bins-SR",
        type=int,
        default=12,
        help="Number of bins in SR (default: 12)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for model loading (default: 42)",
    )
    parser.add_argument(
        "--include-training-plots",
        action="store_true",
        help="Include training loss and ROC curve plots",
    )
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 确定输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("outputs") / config.dataset_id / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 确定拟合函数
    if args.fit_type == "cubic":
        fit_function = bkg_fit_cubic
        fit_degree = 3
    elif args.fit_type == "quintic":
        fit_function = bkg_fit_quintic
        fit_degree = 5
    elif args.fit_type == "septic":
        fit_function = bkg_fit_septic
        fit_degree = 7
    else:
        raise ValueError(f"Unknown fit type: {args.fit_type}")
    
    # 确定评估结果文件路径
    if args.evaluation_results:
        eval_path = Path(args.evaluation_results)
    else:
        # 自动查找最新的评估结果
        eval_dir = Path("outputs") / config.dataset_id / "evaluation"
        eval_files = list(eval_dir.glob("bump_hunt_results_seed*.pkl"))
        if not eval_files:
            raise FileNotFoundError(f"No evaluation results found in {eval_dir}")
        eval_path = sorted(eval_files)[-1]
        logger.info(f"Using evaluation results: {eval_path}")
    
    # 加载评估结果
    with open(eval_path, "rb") as f:
        eval_results = pickle.load(f)
    
    # 获取窗口定义
    window = config.get_window()
    SB_left = window["SB_left"]
    SB_right = window["SB_right"]
    SR_left = window["SR_left"]
    SR_right = window["SR_right"]
    
    # 获取特征集合
    training_config = config.raw_config.get("training", {})
    feature_set_name = training_config.get("feature_set", "mix_0")
    feature_set = config.get_feature_set(feature_set_name)
    # 移除 dimu_mass，因为它已经单独画过了
    feature_set_for_plot = [f for f in feature_set if f != "dimu_mass"]
    
    # 颜色方案（与原始代码一致）
    num_points = 7
    bsx_c = [((0.99 * i) * np.power(i, 0), 0, 0.99*(1-i) * np.power(i, 0.0)) for i in np.linspace(0, 1, num_points)]
    bsx_a = np.linspace(0.99, 0.99, num_points)
    
    # 1. 绘制 Cut Histograms
    logger.info("Generating Cut Histograms...")
    multi_fpr_data = eval_results.get("multi_fpr_data", None)
    if multi_fpr_data:
        fig, ax = plot_histograms_with_fits(
            save_data=multi_fpr_data,
            SB_left=SB_left,
            SR_left=SR_left,
            SR_right=SR_right,
            SB_right=SB_right,
            fit_function=fit_function,
            num_bins_SR=args.num_bins_SR,
            title=r"$\bf{ 2016\,\, CMS\,\, Open\,\, Data\,\, DoubleMuon}$",
            upsilon_lines=True,
            colors=bsx_c,
            alphas=bsx_a,
            line_0=r"$\bf{Opposite Sign}$",  # 原始代码使用 r"$\textbf{Opposite Sign}$"，但需要转换为 $\bf{...}$
        )
        save_path = output_dir / f"histogram_{args.fit_type}_{args.num_bins_SR}.pdf"
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved Cut Histograms to {save_path}")
    else:
        logger.warning("Multi-FPR data not found in evaluation results. Skipping Cut Histograms.")
    
    # 2. 绘制 Feature Histograms
    logger.info("Generating Feature Histograms...")
    feature_data = eval_results.get("feature_data", None)
    isolation_data = eval_results.get("isolation_data", None)
    if feature_data and isolation_data and feature_set_for_plot:
        fig, ax = plot_features(
            save_data=feature_data,
            isolation_data=isolation_data,
            feature_set=feature_set_for_plot,
            colors=bsx_c,
            alphas=bsx_a,
            num_bins_SR=args.num_bins_SR,
            fit_type=args.fit_type,
            num_points=num_points,
        )
        save_path = output_dir / f"features_{args.fit_type}_{args.num_bins_SR}.pdf"
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved Feature Histograms to {save_path}")
    else:
        logger.warning("Feature data or isolation data not found. Skipping Feature Histograms.")
    
    # 3. 绘制 Significances
    logger.info("Generating Significances plot...")
    feature_sigs = eval_results.get("feature_sigs", {})
    random_sigs = eval_results.get("random_sigs", np.array([]))
    fpr_thresholds_finegrained = eval_results.get("fpr_thresholds_finegrained", np.array([]))
    bonus = eval_results.get("full_q0", None)
    
    if feature_sigs and random_sigs.size > 0 and fpr_thresholds_finegrained.size > 0:
        fig, ax = plot_sig(
            feature_sigs=feature_sigs,
            random_sigs=random_sigs,
            fpr_thresholds_finegrained=fpr_thresholds_finegrained,
            num_bins_SR=args.num_bins_SR,
            fit_type=args.fit_type,
            ymax=10,
            ymin=1e-15,
            bonus=bonus,
            line_0=r"$\bf{Opposite Sign}$ Muons",
        )
        save_path = output_dir / f"significance_{args.fit_type}_{args.num_bins_SR}.pdf"
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved Significances plot to {save_path}")
    else:
        logger.warning("Significance data not found. Skipping Significances plot.")
    
    # 4. 绘制训练损失曲线（如果启用）
    if args.include_training_plots:
        logger.info("Generating Training Loss plot...")
        model_dir = config.output_dir / "models" / f"seed{args.seed}"
        train_losses_path = model_dir / "train_losses.npy"
        val_losses_path = model_dir / "val_losses.npy"
        
        if train_losses_path.exists() and val_losses_path.exists():
            train_losses = np.load(train_losses_path)
            val_losses = np.load(val_losses_path)
            
            fig, ax = plot_training_losses(
                train_losses=train_losses,
                val_losses=val_losses,
            )
            save_path = output_dir / f"training_losses.pdf"
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved Training Loss plot to {save_path}")
        else:
            logger.warning(f"Training loss files not found. Expected:")
            logger.warning(f"  - {train_losses_path}")
            logger.warning(f"  - {val_losses_path}")
        
        # 5. 绘制ROC曲线（如果启用，需要模型生成样本）
        logger.info("Generating ROC Curve plot...")
        try:
            # 加载区域数据（用于获取SB的mass值）
            data_config = config.raw_config.get("data", {})
            charge_filter = data_config.get("charge_filter", "OS")
            region_data_path = config.output_dir / "processed_data" / f"region_data_{charge_filter}.pkl"
            
            if not region_data_path.exists():
                logger.warning(f"Region data not found: {region_data_path}. Skipping ROC curve.")
            else:
                with open(region_data_path, "rb") as f:
                    region_data = pickle.load(f)
                
                # 获取特征集合
                training_config = config.raw_config.get("training", {})
                feature_set_name = training_config.get("feature_set", "mix_0")
                feature_set = config.get_feature_set(feature_set_name)
                
                # 组装SB数据（包括mass）
                bands = ["SBL", "SBH"]
                banded_data = assemble_banded_datasets(region_data, feature_set, bands)
                
                if "SBL" not in banded_data or "SBH" not in banded_data:
                    logger.warning("SB data not found. Skipping ROC curve.")
                else:
                    # 合并SBL和SBH数据
                    SB_data_raw = np.vstack([banded_data["SBL"], banded_data["SBH"]])
                    
                    # 找到mass在feature_set中的索引
                    mass_idx = feature_set.index("dimu_mass")
                    SB_masses = SB_data_raw[:, mass_idx]
                    
                    # 加载预处理信息
                    processed_data_dir = config.output_dir / "processed_data"
                    preprocessing_info_path = processed_data_dir / f"preprocessing_info_bootstrap{args.seed}"
                    mass_scaler_path = processed_data_dir / f"mass_scaler_bootstrap{args.seed}"
                    
                    if not preprocessing_info_path.exists() or not mass_scaler_path.exists():
                        logger.warning("Preprocessing info not found. Skipping ROC curve.")
                    else:
                        with open(preprocessing_info_path, "rb") as f:
                            preprocessing_info = pickle.load(f)
                        with open(mass_scaler_path, "rb") as f:
                            mass_scaler = pickle.load(f)
                        
                        # 预处理mass值（用于条件输入）
                        SB_masses_scaled = mass_scaler.transform(SB_masses.reshape(-1, 1)).reshape(-1)
                        
                        # 加载模型
                        model_dir = config.output_dir / "models" / f"seed{args.seed}"
                        model_path = model_dir / "best_model.pt"
                        if not model_path.exists():
                            epoch_files = sorted(model_dir.glob("model_epoch_*.pt"))
                            if epoch_files:
                                model_path = epoch_files[-1]
                            else:
                                logger.warning(f"No model found in {model_dir}. Skipping ROC curve.")
                                model_path = None
                        
                        if model_path and model_path.exists():
                            from resonance_ad.core import get_device, get_device_info
                            logger.info(get_device_info())
                            device = get_device()
                            model_config_path = config.working_dir / "configs" / training_config.get("config_file", "CATHODE_8.yml")
                            num_features = len(feature_set) - 1  # 不包括mass
                            
                            logger.info(f"Loading model for ROC curve generation...")
                            cathode = DensityEstimator(
                                config_path=model_config_path,
                                num_inputs=num_features,
                                eval_mode=True,
                                load_path=model_path,
                                device=device,
                                verbose=False,
                            )
                            
                            # 从模型生成样本
                            logger.info(f"Generating {len(SB_masses_scaled)} samples from model...")
                            preprocessor = DataPreprocessor(cushion=0.02)
                            
                            # 批量生成样本以避免内存问题
                            batch_size = 10000
                            SB_samples_list = []
                            for i in range(0, len(SB_masses_scaled), batch_size):
                                batch_masses = SB_masses_scaled[i:i+batch_size]
                                batch_cond = torch.tensor(batch_masses.reshape(-1, 1)).float().to(device)
                                batch_samples = cathode.sample(
                                    num_samples=len(batch_masses),
                                    cond_inputs=batch_cond
                                )
                                SB_samples_list.append(batch_samples.detach().cpu().numpy())
                            SB_samples = np.vstack(SB_samples_list)
                            
                            # 预处理真实SB数据（使用相同的预处理）
                            SB_data_processed = preprocessor.apply_preprocessing(
                                SB_data_raw,
                                feature_set,
                                preprocessing_info,
                                mass_scaler=mass_scaler,
                                mass_key="dimu_mass",
                            )
                            
                            # 提取特征部分（不包括mass，mass在最后）
                            # feature_set的顺序是 [feat1, feat2, ..., dimu_mass]
                            # 所以特征部分是前 num_features 列
                            SB_data_features = SB_data_processed[:, :num_features]
                            SB_samples_features = SB_samples[:, :num_features]
                            
                            # 清理数据（去除NaN和Inf）
                            SB_data_features = preprocessor.clean_data(SB_data_features)
                            SB_samples_features = preprocessor.clean_data(SB_samples_features)
                            
                            # 确保数据长度一致
                            min_len = min(len(SB_data_features), len(SB_samples_features))
                            SB_data_features = SB_data_features[:min_len]
                            SB_samples_features = SB_samples_features[:min_len]
                            
                            logger.info(f"SB data shape: {SB_data_features.shape}, SB samples shape: {SB_samples_features.shape}")
                            
                            # 绘制ROC曲线
                            auc_mean, auc_std = plot_roc_curve(
                                data=SB_data_features,
                                samples=SB_samples_features,
                                n_runs=3,
                                save_path=output_dir / "roc_curve",
                            )
                            logger.info(f"ROC AUC: {auc_mean:.3f} ± {auc_std:.3f}")
                            logger.info(f"Saved ROC Curve plot to {output_dir / 'roc_curve.pdf'}")
        except Exception as e:
            import traceback
            logger.warning(f"Error generating ROC curve: {e}")
            logger.warning(f"Traceback: {traceback.format_exc()}")
            logger.warning("ROC curve generation failed. This is optional and can be skipped.")
    
    # 6. 绘制 Variations 图（不同拟合类型和bin宽度）
    logger.info("Generating Variations plot...")
    variations_data = eval_results.get("variations_data", None)
    if variations_data:
        fig, ax = plot_variations(
            variations_data=variations_data,
            fpr_thresholds_finegrained=fpr_thresholds_finegrained,
            save_path=output_dir / "significance_variations.pdf",
        )
        logger.info(f"Saved Variations plot to {output_dir / 'significance_variations.pdf'}")
    else:
        logger.warning("Variations data not found. Skipping Variations plot.")
        logger.warning("  To generate this plot, run evaluate.py with --all-variations flag")
    
    logger.info(f"All figures saved to: {output_dir}")
    logger.info("Figure generation completed successfully")


if __name__ == "__main__":
    main()


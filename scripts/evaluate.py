#!/usr/bin/env python
"""
评估 CATHODE 模型并进行 Bump Hunt

加载训练好的模型，计算 anomaly scores，并进行 bump hunt。
"""

import argparse
import pickle
import numpy as np
import torch
from pathlib import Path

from resonance_ad.core.config import load_config
from resonance_ad.core.logging import setup_logging, get_logger
from resonance_ad.models import DensityEstimator
from resonance_ad.analysis import BumpHunter
from resonance_ad.analysis.bump_hunt import fit_background, parametric_fit
from resonance_ad.analysis.significance import calculate_test_statistic
from resonance_ad.data import assemble_banded_datasets
from resonance_ad.data.preprocessor import DataPreprocessor
from resonance_ad.physics.binning import get_bins

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate CATHODE and perform bump hunt")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--model-dir", type=str, default=None, help="Model directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--input", type=str, default=None, help="Input region data file")
    parser.add_argument("--max-iter", type=int, default=15090, help="Maximum iterations for Nelder-Mead optimizer (default: 15090, same as original code)")
    parser.add_argument("--fast", action="store_true", help="Use reduced iterations (400) for faster evaluation")
    parser.add_argument("--all-variations", action="store_true", help="Calculate significances for all fit types and bin numbers (cubic/quintic/septic × 8/12/16 bins)")
    args = parser.parse_args()
    
    # 如果使用 --fast，减少迭代次数
    if args.fast:
        args.max_iter = 400
    
    # 加载配置
    config = load_config(args.config)
    setup_logging(log_dir=config.output_dir / "logs")
    
    logger.info(f"Evaluating CATHODE with config: {args.config}")
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 确定输入文件
    if args.input is None:
        data_config = config.raw_config.get("data", {})
        charge_filter = data_config.get("charge_filter", "OS")
        input_path = config.output_dir / "processed_data" / f"region_data_{charge_filter}.pkl"
    else:
        input_path = Path(args.input)
    
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return
    
    # 加载区域数据
    logger.info(f"Loading region data from: {input_path}")
    with open(input_path, "rb") as f:
        region_data_original = pickle.load(f)  # 保存原始数据用于后续获取物理质量值
    
    # 获取特征集合
    training_config = config.raw_config.get("training", {})
    feature_set_name = training_config.get("feature_set", "mix_0")
    feature_set = config.get_feature_set(feature_set_name)
    
    logger.info(f"Using feature set: {feature_set_name}")
    
    # 组装数据
    bands = ["SBL", "SR", "SBH"]
    banded_data = assemble_banded_datasets(region_data_original, feature_set, bands)
    
    # 加载预处理信息
    processed_data_dir = config.output_dir / "processed_data"
    bootstrap_seed = args.seed
    preprocessing_info_path = processed_data_dir / f"preprocessing_info_bootstrap{bootstrap_seed}"
    mass_scaler_path = processed_data_dir / f"mass_scaler_bootstrap{bootstrap_seed}"
    
    if not preprocessing_info_path.exists() or not mass_scaler_path.exists():
        logger.error(
            f"Preprocessing info not found. Please run training first.\n"
            f"Expected files:\n"
            f"  - {preprocessing_info_path}\n"
            f"  - {mass_scaler_path}"
        )
        return
    
    logger.info(f"Loading preprocessing info from: {preprocessing_info_path}")
    with open(preprocessing_info_path, "rb") as f:
        preprocessing_info = pickle.load(f)
    
    logger.info(f"Loading mass scaler from: {mass_scaler_path}")
    with open(mass_scaler_path, "rb") as f:
        mass_scaler = pickle.load(f)
    
    # 应用预处理
    logger.info("Applying preprocessing to data...")
    preprocessor = DataPreprocessor(cushion=0.02)
    banded_data_processed = {}
    for band in bands:
        if band in banded_data:
            banded_data_processed[band] = preprocessor.apply_preprocessing(
                banded_data[band],
                feature_set,
                preprocessing_info,
                mass_scaler=mass_scaler,
                mass_key="dimu_mass",
            )
            logger.info(f"{band} data: original shape {banded_data[band].shape}, processed shape {banded_data_processed[band].shape}")
    
    # 使用预处理后的数据
    banded_data = banded_data_processed
    
    # 设备
    from resonance_ad.core import get_device, get_device_info
    logger.info(get_device_info())
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # 加载模型
    if args.model_dir is None:
        model_dir = config.output_dir / "models" / f"seed{args.seed}"
    else:
        model_dir = Path(args.model_dir)
    
    model_path = model_dir / "best_model.pt"
    if not model_path.exists():
        logger.warning(f"Best model not found, trying latest epoch...")
        # 尝试加载最后一个 epoch
        epoch_files = sorted(model_dir.glob("model_epoch_*.pt"))
        if epoch_files:
            model_path = epoch_files[-1]
        else:
            logger.error(f"No model found in {model_dir}")
            return
    
    logger.info(f"Loading model from: {model_path}")
    
    model_config_path = config.working_dir / "configs" / training_config.get("config_file", "CATHODE_8.yml")
    num_features = len(feature_set) - 1
    
    cathode = DensityEstimator(
        config_path=model_config_path,
        num_inputs=num_features,
        eval_mode=True,
        load_path=model_path,
        device=device,
        verbose=True,
    )
    
    # 计算 anomaly scores（所有区域）
    logger.info("Computing anomaly scores for all regions...")
    bump_hunter = BumpHunter(config)
    
    # 计算所有区域的 scores
    all_scores = {}
    # 获取物理质量值（用于区域判断和显著性计算）
    all_masses_physical = {}
    for region in ["SBL", "SR", "SBH"]:
        if region in banded_data:
            region_data_array = banded_data[region]  # 避免变量名冲突
            region_scores = bump_hunter.compute_anomaly_scores(cathode.model, region_data_array, device)
            all_scores[region] = region_scores
            
            # 获取物理质量值
            region_masses_physical = region_data_original[region]["dimu_mass"]
            if hasattr(region_masses_physical, 'to_numpy'):
                region_masses_physical = region_masses_physical.to_numpy()
            elif not isinstance(region_masses_physical, np.ndarray):
                region_masses_physical = np.array(region_masses_physical)
            all_masses_physical[region] = region_masses_physical
            
            logger.info(f"{region} data shape: {region_data_array.shape}, scores: mean={region_scores.mean():.4f}, std={region_scores.std():.4f}")
    
    SR_data = banded_data["SR"]
    SR_scores = all_scores["SR"]
    SR_masses = all_masses_physical["SR"]  # 使用物理值用于显著性计算
    
    # 获取窗口定义
    window = config.get_window()
    SR_left = window["SR_left"]
    SR_right = window["SR_right"]
    SB_left = window["SB_left"]
    SB_right = window["SB_right"]
    
    # 获取 binning
    region_config = config.raw_config.get("region_selection", {})
    num_bins_SR = region_config.get("num_bins_SR", 12)
    
    _, bins_SR, bins_left, bins_right, _, _, centers_SB = get_bins(
        SR_left, SR_right, SB_left, SB_right, num_bins_SR, binning="linear"
    )
    
    # 定义 FPR 阈值（与原始代码一致）
    fpr_thresholds = np.array([1.0, 0.25, 0.1, 0.05, 0.01, 0.005])
    
    # 计算多个 FPR 阈值的结果
    logger.info(f"Computing results for {len(fpr_thresholds)} FPR thresholds...")
    
    # 准备 SB 数据用于计算 FPR
    SB_scores = np.concatenate([all_scores["SBL"], all_scores["SBH"]])
    SB_masses = np.concatenate([all_masses_physical["SBL"], all_masses_physical["SBH"]])
    
    # 打印初始数据量（用于调试）
    logger.info("=" * 60)
    logger.info("Initial Data Statistics:")
    logger.info(f"  SBL events: {len(all_masses_physical['SBL']):,}")
    logger.info(f"  SR events: {len(all_masses_physical['SR']):,}")
    logger.info(f"  SBH events: {len(all_masses_physical['SBH']):,}")
    logger.info(f"  Total events: {len(all_masses_physical['SBL']) + len(all_masses_physical['SR']) + len(all_masses_physical['SBH']):,}")
    logger.info(f"  SB events (SBL+SBH): {len(SB_scores):,}")
    logger.info("=" * 60)
    
    # 计算 score cutoff points（用于找到对应的 FPR）
    score_cut_points = np.linspace(SB_scores.min(), SB_scores.max(), 10000)
    
    # 计算每个 cutoff 对应的 FPR
    FPR_values = []
    for cut in score_cut_points:
        fpr = np.sum(SB_scores >= cut) / len(SB_scores)
        FPR_values.append(fpr)
    FPR_values = np.array(FPR_values)
    
    # 存储多个 FPR 阈值的结果
    save_data = {
        "fpr_thresholds": fpr_thresholds,
        "popts": [],
        "pcovs": [],
        "significances": [],
        "filtered_masses": [],
        "y_vals": [],
    }
    
    # 对每个 FPR 阈值进行处理
    for t, threshold in enumerate(fpr_thresholds):
        logger.info(f"Processing FPR threshold {t+1}/{len(fpr_thresholds)}: {threshold:.3f}")
        
        # FPR=1.0 时仍然计算，但使用较少的迭代次数
        if threshold >= 0.999:
            logger.info(f"  Processing FPR=1.0 with reduced iterations (large dataset)")
            # 使用所有数据（无过滤）
            filtered_masses_SBL = all_masses_physical["SBL"]
            filtered_masses_SR = all_masses_physical["SR"]
            filtered_masses_SBH = all_masses_physical["SBH"]
            filtered_masses = np.concatenate([filtered_masses_SBL, filtered_masses_SR, filtered_masses_SBH])
            
            logger.info(f"  Filtered events: {len(filtered_masses)} (no cutoff, FPR=1.0)")
            logger.info(f"    SBL events: {len(filtered_masses_SBL):,}, SR events: {len(filtered_masses_SR):,}, SBH events: {len(filtered_masses_SBH):,}")
            
            # 拟合背景
            popt, pcov, chi2 = fit_background(
                filtered_masses,
                fit_degree=5,
                SR_left=SR_left,
                SR_right=SR_right,
                plot_bins_left=bins_left,
                plot_bins_right=bins_right,
                plot_centers_SB=centers_SB,
            )
            
            # 使用 calculate_test_statistic 但减少迭代次数（减少20倍）
            try:
                logger.debug(f"  Computing test statistic with reduced iterations...")
                S, B, q0 = calculate_test_statistic(
                    filtered_masses,
                    SR_left=SR_left,
                    SR_right=SR_right,
                    SB_left=SB_left,
                    SB_right=SB_right,
                    num_bins=num_bins_SR,
                    weights=None,
                    degree=5,
                    starting_guess=popt,
                    verbose_plot=False,
                    max_iter_nelder=args.max_iter,  # 原代码使用 15090，可通过 --max-iter 或 --fast 调整
                )
                max_significance = np.sqrt(q0) if q0 > 0 else 0.0
                s_over_b = S/B if B > 0 else 0
                logger.info(f"  [DEBUG] FPR={threshold:.4f}: Total={len(filtered_masses):,}, SR={len(filtered_masses_SR):,}, S={S:.2f}, B={B:.2f}, S/B={s_over_b:.4f}, sqrt(q0)={max_significance:.2f}σ")
                logger.debug(f"  Test statistic: S={S:.2f}, B={B:.2f}, sqrt(q0)={max_significance:.2f}")
            except Exception as e:
                logger.warning(f"  Error computing test statistic for FPR {threshold:.3f}: {e}")
                logger.warning(f"  Falling back to simplified Poisson significance...")
                # 回退到简化的 Poisson 显著性
                centers_SR = 0.5 * (bins_SR[1:] + bins_SR[:-1])
                background_expectation = parametric_fit(centers_SR, *popt)
                SR_counts, _ = np.histogram(filtered_masses_SR, bins=bins_SR)
                significances = []
                for i in range(len(SR_counts)):
                    obs = SR_counts[i]
                    exp = background_expectation[i]
                    if exp > 0:
                        significance = (obs - exp) / np.sqrt(exp)
                    else:
                        significance = 0
                    significances.append(significance)
                max_significance = np.max(significances) if len(significances) > 0 else 0
        else:
            # 找到对应的 score cutoff
            best_cut_idx = np.argmin(np.abs(FPR_values - threshold))
            score_cutoff = score_cut_points[best_cut_idx]
            
            # 根据 score cutoff 过滤数据（使用物理质量值）
            filtered_masses_SBL = all_masses_physical["SBL"][all_scores["SBL"] >= score_cutoff]
            filtered_masses_SR = all_masses_physical["SR"][all_scores["SR"] >= score_cutoff]
            filtered_masses_SBH = all_masses_physical["SBH"][all_scores["SBH"] >= score_cutoff]
            
            # 合并所有过滤后的质量
            filtered_masses = np.concatenate([filtered_masses_SBL, filtered_masses_SR, filtered_masses_SBH])
            
            logger.info(f"  Filtered events: {len(filtered_masses)} (cutoff={score_cutoff:.4f})")
            logger.info(f"    SBL events: {len(filtered_masses_SBL):,}, SR events: {len(filtered_masses_SR):,}, SBH events: {len(filtered_masses_SBH):,}")
            
            # 拟合背景
            popt, pcov, chi2 = fit_background(
                filtered_masses,
                fit_degree=5,  # 从配置读取
                SR_left=SR_left,
                SR_right=SR_right,
                plot_bins_left=bins_left,
                plot_bins_right=bins_right,
                plot_centers_SB=centers_SB,
            )
            
            # 使用 calculate_test_statistic 计算显著性（减少20倍迭代次数以加快速度）
            try:
                logger.debug(f"  Computing test statistic with reduced iterations...")
                S, B, q0 = calculate_test_statistic(
                    filtered_masses,
                    SR_left=SR_left,
                    SR_right=SR_right,
                    SB_left=SB_left,
                    SB_right=SB_right,
                    num_bins=num_bins_SR,
                    weights=None,
                    degree=5,
                    starting_guess=popt,
                    verbose_plot=False,
                    max_iter_nelder=args.max_iter,  # 原代码使用 15090，可通过 --max-iter 或 --fast 调整
                )
                max_significance = np.sqrt(q0) if q0 > 0 else 0.0
                s_over_b = S/B if B > 0 else 0
                logger.info(f"  [DEBUG] FPR={threshold:.4f}: Total={len(filtered_masses):,}, SR={len(filtered_masses_SR):,}, S={S:.2f}, B={B:.2f}, S/B={s_over_b:.4f}, sqrt(q0)={max_significance:.2f}σ")
                logger.debug(f"  Test statistic: S={S:.2f}, B={B:.2f}, sqrt(q0)={max_significance:.2f}")
            except Exception as e:
                logger.warning(f"  Error computing test statistic for FPR {threshold:.3f}: {e}")
                logger.warning(f"  Falling back to simplified Poisson significance...")
                # 回退到简化的 Poisson 显著性
                centers_SR = 0.5 * (bins_SR[1:] + bins_SR[:-1])
                background_expectation = parametric_fit(centers_SR, *popt)
                SR_filtered_masses = filtered_masses_SR
                SR_counts, _ = np.histogram(SR_filtered_masses, bins=bins_SR)
                significances = []
                for i in range(len(SR_counts)):
                    obs = SR_counts[i]
                    exp = background_expectation[i]
                    if exp > 0:
                        significance = (obs - exp) / np.sqrt(exp)
                    else:
                        significance = 0
                    significances.append(significance)
                max_significance = np.max(significances) if len(significances) > 0 else 0
        
        logger.info(f"  Max significance: {max_significance:.2f} sigma")
        
        # 存储结果
        save_data["popts"].append(popt)
        save_data["pcovs"].append(pcov)
        save_data["significances"].append(max_significance)
        save_data["filtered_masses"].append(filtered_masses)
        save_data["y_vals"].append(None)  # y_vals 可以后续计算
    
    # 计算单个 FPR 阈值的结果（用于兼容性）
    logger.info("Computing single FPR threshold result for compatibility...")
    # 使用物理质量值进行背景拟合
    SB_masses_single = np.concatenate([all_masses_physical["SBL"], all_masses_physical["SBH"]])
    
    popt_single, pcov_single, chi2_single = fit_background(
        SB_masses_single,
        fit_degree=5,
        SR_left=SR_left,
        SR_right=SR_right,
        plot_bins_left=bins_left,
        plot_bins_right=bins_right,
        plot_centers_SB=centers_SB,
    )
    
    centers_SR = 0.5 * (bins_SR[1:] + bins_SR[:-1])
    background_expectation = parametric_fit(centers_SR, *popt_single)
    
    # Bump hunt（使用物理质量值）
    logger.info("Performing bump hunt...")
    # 注意：hunt_bump 需要物理质量值，而不是预处理后的值
    SR_masses_for_bump = SR_masses  # 使用之前获取的物理质量值
    
    # 调试：打印一些统计信息
    logger.debug(f"SR masses shape: {SR_masses_for_bump.shape}")
    logger.debug(f"SR masses range: [{SR_masses_for_bump.min():.2f}, {SR_masses_for_bump.max():.2f}]")
    logger.debug(f"Background expectation shape: {background_expectation.shape}")
    logger.debug(f"Background expectation sum: {background_expectation.sum():.2f}")
    
    results = bump_hunter.hunt_bump(
        SR_masses_for_bump, SR_scores, bins_SR, background_expectation
    )
    
    logger.info(f"Max significance: {results['max_significance']:.2f} sigma")
    logger.info(f"Max significance bin: {results['max_significance_bin']}")
    logger.debug(f"Observed counts: {results['observed']}")
    logger.debug(f"Expected counts: {results['expected']}")
    logger.debug(f"All significances: {results['significances']}")
    
    # 计算特征显著性数据（用于 plot_sig）
    logger.info("Computing feature significances...")
    # 细粒度 FPR 阈值（与原代码对齐：从 1.0 到 0.001，25 个点）
    fpr_thresholds_finegrained = np.logspace(0, -3, 25)
    
    # 获取用于绘图的特征集合（排除 dimu_mass）
    feature_set_for_plot = [f for f in feature_set if f != "dimu_mass"]
    
    # 定义 flip 和 abs_val 字典（与原始代码一致）
    flip_features_dict = {
        "dimu_pt": False,
        "dimu_eta": True,
        "mu0_pt": False,
        "mu1_pt": False,
        "mu0_eta": True,
        "mu1_eta": True,
        "mu0_ip3d": True,
        "mu1_ip3d": True,
        "mu0_iso04": True,
        "mu1_iso04": True,
        "mumu_deltapT": True,
        "mumu_deltaR": False,
    }
    
    abs_features_dict = {
        "dimu_pt": False,
        "dimu_eta": True,
        "mu0_pt": False,
        "mu1_pt": False,
        "mu0_eta": True,
        "mu1_eta": True,
        "mu0_ip3d": False,
        "mu1_ip3d": False,
        "mu0_iso04": False,
        "mu1_iso04": False,
        "mumu_deltapT": False,
        "mumu_deltaR": False,
    }
    
    # 计算 CATHODE 的显著性（使用已有的 save_data，但需要重新计算以使用 calculate_test_statistic）
    cathode_sigs = np.zeros((len(fpr_thresholds_finegrained), 1))
    for i, fpr_fine in enumerate(fpr_thresholds_finegrained):
        # 找到最接近的 FPR 阈值
        best_idx = np.argmin(np.abs(fpr_thresholds - fpr_fine))
        if best_idx < len(save_data["filtered_masses"]):
            filtered_masses_cathode = save_data["filtered_masses"][best_idx]
            popt_cathode = save_data["popts"][best_idx]
            try:
                S, B, q0 = calculate_test_statistic(
                    filtered_masses_cathode,
                    SR_left=SR_left,
                    SR_right=SR_right,
                    SB_left=SB_left,
                    SB_right=SB_right,
                    num_bins=num_bins_SR,
                    weights=None,
                    degree=5,
                    starting_guess=popt_cathode,
                    verbose_plot=False,
                    max_iter_nelder=args.max_iter,  # 原代码使用 15090，可通过 --max-iter 或 --fast 调整
                )
                cathode_sigs[i, 0] = np.sqrt(q0) if q0 > 0 else 0.0
            except Exception as e:
                logger.warning(f"Error computing CATHODE significance at FPR {fpr_fine:.4f}: {e}")
                cathode_sigs[i, 0] = save_data["significances"][best_idx] if best_idx < len(save_data["significances"]) else 0.0
    
    feature_sigs = {"CATHODE": cathode_sigs}
    
    # 计算其他特征的显著性
    logger.info(f"Computing significances for {len(feature_set_for_plot[:3])} features...")
    for feat_idx_plot, feat_name in enumerate(feature_set_for_plot[:3]):  # 只计算前3个特征（dimu_pt, mu0_ip3d, mu1_ip3d）
        if feat_name in feature_set:
            logger.info(f"Processing feature {feat_idx_plot+1}/{len(feature_set_for_plot[:3])}: {feat_name}")
            feat_idx = feature_set.index(feat_name)
            flip_feat = flip_features_dict.get(feat_name, False)
            abs_val_feat = abs_features_dict.get(feat_name, False)
            
            # 从所有区域数据中提取特征值（预处理后的）
            all_feature_values = np.concatenate([
                banded_data["SBL"][:, feat_idx],
                SR_data[:, feat_idx],
                banded_data["SBH"][:, feat_idx]
            ])
            # 获取物理质量值（用于区域划分）
            SBL_masses_physical = region_data_original["SBL"]["dimu_mass"]
            SBH_masses_physical = region_data_original["SBH"]["dimu_mass"]
            SR_masses_physical_feat = region_data_original["SR"]["dimu_mass"]  # 使用不同的变量名避免冲突
            if hasattr(SBL_masses_physical, 'to_numpy'):
                SBL_masses_physical = SBL_masses_physical.to_numpy()
            elif not isinstance(SBL_masses_physical, np.ndarray):
                SBL_masses_physical = np.array(SBL_masses_physical)
            if hasattr(SBH_masses_physical, 'to_numpy'):
                SBH_masses_physical = SBH_masses_physical.to_numpy()
            elif not isinstance(SBH_masses_physical, np.ndarray):
                SBH_masses_physical = np.array(SBH_masses_physical)
            if hasattr(SR_masses_physical_feat, 'to_numpy'):
                SR_masses_physical_feat = SR_masses_physical_feat.to_numpy()
            elif not isinstance(SR_masses_physical_feat, np.ndarray):
                SR_masses_physical_feat = np.array(SR_masses_physical_feat)
            
            all_masses_feat = np.concatenate([
                SBL_masses_physical,
                SR_masses_physical_feat,
                SBH_masses_physical
            ])
            
            # 应用 flip 和 abs_val
            feature_of_interest = all_feature_values.copy()
            if abs_val_feat:
                feature_of_interest = np.abs(feature_of_interest)
            if flip_feat:
                feature_of_interest = -feature_of_interest
            
            # 分割到不同区域
            feature_SBL = feature_of_interest[all_masses_feat < SR_left]
            feature_SR = feature_of_interest[(all_masses_feat >= SR_left) & (all_masses_feat <= SR_right)]
            feature_SBH = feature_of_interest[all_masses_feat > SR_right]
            
            mass_SBL_feat = all_masses_feat[all_masses_feat < SR_left]
            mass_SR_feat = all_masses_feat[(all_masses_feat >= SR_left) & (all_masses_feat <= SR_right)]
            mass_SBH_feat = all_masses_feat[all_masses_feat > SR_right]
            
            # 计算特征切割点
            feature_cut_points = np.linspace(
                np.min(feature_of_interest),
                np.max(feature_of_interest),
                10000
            )
            
            # 计算每个切割点对应的 FPR
            FPR_values_feat = []
            for cut in feature_cut_points:
                fpr = (np.sum(feature_SBH >= cut) + np.sum(feature_SBL >= cut)) / (len(feature_SBH) + len(feature_SBL))
                FPR_values_feat.append(fpr)
            FPR_values_feat = np.array(FPR_values_feat)
            
            # 对每个细粒度 FPR 阈值计算显著性
            # 为了加快速度，跳过 FPR=1.0 附近的阈值（通常不用于分析）
            fpr_thresholds_to_compute = fpr_thresholds_finegrained[fpr_thresholds_finegrained < 0.99]
            feat_sigs = np.zeros((len(fpr_thresholds_finegrained), 1))
            logger.info(f"  Computing significances for {len(fpr_thresholds_to_compute)} FPR thresholds (skipping FPR>=0.99)...")
            
            # 创建映射：fpr_thresholds_finegrained 索引到 fpr_thresholds_to_compute 索引
            compute_idx = 0
            for i, fpr_fine in enumerate(fpr_thresholds_finegrained):
                if fpr_fine >= 0.99:
                    feat_sigs[i, 0] = 0.0  # 跳过 FPR>=0.99
                    continue
                
                if compute_idx % 3 == 0:  # 每3个打印一次进度
                    logger.info(f"    Processing FPR threshold {compute_idx+1}/{len(fpr_thresholds_to_compute)}: {fpr_fine:.4f}")
                
                # 找到对应的特征切割点
                best_cut_idx = np.argmin(np.abs(FPR_values_feat - fpr_fine))
                feature_cutoff = feature_cut_points[best_cut_idx]
                
                # 过滤数据
                mass_SBL_cut = mass_SBL_feat[feature_SBL >= feature_cutoff]
                mass_SR_cut = mass_SR_feat[feature_SR >= feature_cutoff]
                mass_SBH_cut = mass_SBH_feat[feature_SBH >= feature_cutoff]
                
                filtered_masses_feat = np.concatenate([mass_SBL_cut, mass_SR_cut, mass_SBH_cut])
                
                # 如果过滤后的数据太少，跳过
                if len(filtered_masses_feat) < 100:
                    feat_sigs[i, 0] = 0.0
                    compute_idx += 1
                    continue
                
                # 拟合背景
                try:
                    popt_feat, pcov_feat, chi2_feat = fit_background(
                        filtered_masses_feat,
                        fit_degree=5,
                        SR_left=SR_left,
                        SR_right=SR_right,
                        plot_bins_left=bins_left,
                        plot_bins_right=bins_right,
                        plot_centers_SB=centers_SB,
                    )
                    
                    # 使用 calculate_test_statistic 计算显著性（与原代码一致）
                    # 原代码使用 calculate_test_statistic，而不是简化的Poisson计算
                    S_feat, B_feat, q0_feat = calculate_test_statistic(
                        filtered_masses_feat,
                        SR_left=SR_left,
                        SR_right=SR_right,
                        SB_left=SB_left,
                        SB_right=SB_right,
                        num_bins=num_bins_SR,
                        weights=None,
                        degree=5,
                        starting_guess=popt_feat,
                        verbose_plot=False,
                        max_iter_nelder=args.max_iter,
                    )
                    feat_sigs[i, 0] = np.sqrt(q0_feat) if q0_feat > 0 else 0.0
                    
                    # 添加调试输出（每3个FPR阈值打印一次）
                    if compute_idx % 3 == 0:
                        s_over_b_feat = S_feat/B_feat if B_feat > 0 else 0
                        logger.info(f"      [DEBUG] {feat_name} FPR={fpr_fine:.4f}: Total={len(filtered_masses_feat):,}, SR={len(mass_SR_cut):,}, S={S_feat:.2f}, B={B_feat:.2f}, S/B={s_over_b_feat:.4f}, sqrt(q0)={feat_sigs[i, 0]:.2f}σ")
                except Exception as e:
                    logger.warning(f"      Error computing significance for {feat_name} at FPR {fpr_fine:.4f}: {e}")
                    feat_sigs[i, 0] = 0.0
                
                compute_idx += 1
            
            feature_sigs[feat_name] = feat_sigs
            logger.info(f"  Completed feature {feat_name}")
    
    # 计算随机切割的显著性
    logger.info(f"Computing random cut significances for {len(fpr_thresholds_finegrained)} FPR thresholds...")
    random_sigs = np.zeros((len(fpr_thresholds_finegrained), 1))
    # 使用物理质量值（用于随机切割）
    SBL_masses_physical_rand = region_data_original["SBL"]["dimu_mass"]
    SBH_masses_physical_rand = region_data_original["SBH"]["dimu_mass"]
    SR_masses_physical_rand = SR_masses  # 使用之前定义的 SR_masses
    if hasattr(SBL_masses_physical_rand, 'to_numpy'):
        SBL_masses_physical_rand = SBL_masses_physical_rand.to_numpy()
    elif not isinstance(SBL_masses_physical_rand, np.ndarray):
        SBL_masses_physical_rand = np.array(SBL_masses_physical_rand)
    if hasattr(SBH_masses_physical_rand, 'to_numpy'):
        SBH_masses_physical_rand = SBH_masses_physical_rand.to_numpy()
    elif not isinstance(SBH_masses_physical_rand, np.ndarray):
        SBH_masses_physical_rand = np.array(SBH_masses_physical_rand)
    
    all_masses_all = np.concatenate([
        SBL_masses_physical_rand,
        SR_masses_physical_rand,
        SBH_masses_physical_rand
    ])
    
    for i, fpr_fine in enumerate(fpr_thresholds_finegrained):
        if (i + 1) % 5 == 0 or i == 0:
            logger.info(f"  Processing random cut FPR threshold {i+1}/{len(fpr_thresholds_finegrained)}: {fpr_fine:.4f}")
        
        # 随机切割：随机生成特征值
        random_feature = np.random.uniform(0, 1, len(all_masses_all))
        
        # 分割到不同区域
        random_SBL = random_feature[all_masses_all < SR_left]
        random_SR = random_feature[(all_masses_all >= SR_left) & (all_masses_all <= SR_right)]
        random_SBH = random_feature[all_masses_all > SR_right]
        
        mass_SBL_rand = all_masses_all[all_masses_all < SR_left]
        mass_SR_rand = all_masses_all[(all_masses_all >= SR_left) & (all_masses_all <= SR_right)]
        mass_SBH_rand = all_masses_all[all_masses_all > SR_right]
        
        # 计算随机切割点
        random_cut_points = np.linspace(np.min(random_feature), np.max(random_feature), 10000)
        FPR_values_rand = []
        for cut in random_cut_points:
            fpr = (np.sum(random_SBH >= cut) + np.sum(random_SBL >= cut)) / (len(random_SBH) + len(random_SBL))
            FPR_values_rand.append(fpr)
        FPR_values_rand = np.array(FPR_values_rand)
        
        # 找到对应的切割点
        best_cut_idx = np.argmin(np.abs(FPR_values_rand - fpr_fine))
        random_cutoff = random_cut_points[best_cut_idx]
        
        # 过滤数据
        mass_SBL_cut = mass_SBL_rand[random_SBL >= random_cutoff]
        mass_SR_cut = mass_SR_rand[random_SR >= random_cutoff]
        mass_SBH_cut = mass_SBH_rand[random_SBH >= random_cutoff]
        
        filtered_masses_rand = np.concatenate([mass_SBL_cut, mass_SR_cut, mass_SBH_cut])
        
        # 拟合背景并计算显著性
        try:
            popt_rand, pcov_rand, chi2_rand = fit_background(
                filtered_masses_rand,
                fit_degree=5,
                SR_left=SR_left,
                SR_right=SR_right,
                plot_bins_left=bins_left,
                plot_bins_right=bins_right,
                plot_centers_SB=centers_SB,
            )
            
            # 对于随机切割，也使用简化的显著性计算以加快速度
            centers_SR_rand = 0.5 * (bins_SR[1:] + bins_SR[:-1])
            background_expectation_rand = parametric_fit(centers_SR_rand, *popt_rand)
            SR_counts_rand, _ = np.histogram(mass_SR_cut, bins=bins_SR)
            significances_rand = []
            for j in range(len(SR_counts_rand)):
                obs = SR_counts_rand[j]
                exp = background_expectation_rand[j]
                if exp > 0:
                    sig = (obs - exp) / np.sqrt(exp)
                else:
                    sig = 0
                significances_rand.append(sig)
            random_sigs[i, 0] = np.max(significances_rand) if len(significances_rand) > 0 else 0.0
            
            # 添加调试输出（每5个FPR阈值打印一次）
            if (i + 1) % 5 == 0 or i == 0:
                logger.info(f"      [DEBUG] Random FPR={fpr_fine:.4f}: Total={len(filtered_masses_rand):,}, SR={len(mass_SR_cut):,}, max_sig={random_sigs[i, 0]:.2f}σ")
        except Exception as e:
            logger.warning(f"  Error computing random significance at FPR {fpr_fine:.4f}: {e}")
            random_sigs[i, 0] = 0
    
    # 计算 full_q0（使用 likelihood reweighting，在 FPR=1.0 时）
    # 注意：这个计算可能很慢，因为需要多次优化
    logger.info("Computing full_q0 with likelihood reweighting (this may take a while)...")
    full_q0 = None
    try:
        # 使用所有数据（FPR=1.0）- 使用物理质量值
        SBL_masses_physical_full = all_masses_physical["SBL"]
        SBH_masses_physical_full = all_masses_physical["SBH"]
        SR_masses_physical_full = SR_masses  # 使用之前定义的 SR_masses
        all_masses_full = np.concatenate([
            SBL_masses_physical_full,
            SR_masses_physical_full,
            SBH_masses_physical_full
        ])
        all_scores_full = np.concatenate([
            all_scores["SBL"],
            SR_scores,
            all_scores["SBH"]
        ])
        
        # 计算权重（likelihood reweighting）
        # 首先拟合背景以估计 mu
        popt_full, pcov_full, chi2_full = fit_background(
            all_masses_full,
            fit_degree=5,
            SR_left=SR_left,
            SR_right=SR_right,
            plot_bins_left=bins_left,
            plot_bins_right=bins_right,
            plot_centers_SB=centers_SB,
        )
        
        # 计算 S 和 B（使用减少的迭代次数）
        S_full, B_full, _ = calculate_test_statistic(
            all_masses_full,
            SR_left=SR_left,
            SR_right=SR_right,
            SB_left=SB_left,
            SB_right=SB_right,
            num_bins=num_bins_SR,
            weights=None,
            degree=5,
            starting_guess=popt_full,
            verbose_plot=False,
            max_iter_nelder=args.max_iter,  # 使用命令行参数
        )
        
        mu = S_full / (S_full + B_full) if (S_full + B_full) > 0 else 0.0
        
        # 计算 likelihood ratios 和 weights（与原代码对齐）
        # 原代码：likelihood_ratios = (all_scores) / (1 - all_scores)
        # 注意：原代码没有添加 1e-10，但如果 all_scores 接近 1 可能会有数值问题
        # 我们保留 1e-10 以提高数值稳定性，但应该不影响结果
        likelihood_ratios = all_scores_full / (1 - all_scores_full + 1e-10)
        # 原代码：weights = (likelihood_ratios - (1-mu)) / mu
        # 我们添加 mu > 0 检查以提高稳健性
        if mu > 0:
            weights_full = (likelihood_ratios - (1 - mu)) / mu
        else:
            weights_full = np.ones_like(all_scores_full)
        # 原代码：weights = np.clip(weights, 0, 1e9)
        weights_full = np.clip(weights_full, 0, 1e9)
        
        # 使用权重重新拟合
        popt_weighted, pcov_weighted, chi2_weighted = fit_background(
            all_masses_full,
            fit_degree=5,
            SR_left=SR_left,
            SR_right=SR_right,
            plot_bins_left=bins_left,
            plot_bins_right=bins_right,
            plot_centers_SB=centers_SB,
            weights=weights_full,
        )
        
        # 使用权重计算 full_q0（使用减少的迭代次数）
        s_weighted, b_weighted, q0_weighted, popt_final = calculate_test_statistic(
            all_masses_full,
            SR_left=SR_left,
            SR_right=SR_right,
            SB_left=SB_left,
            SB_right=SB_right,
            num_bins=num_bins_SR,
            weights=weights_full,
            degree=5,
            starting_guess=popt_weighted,
            verbose_plot=False,
            return_popt=True,
            max_iter_nelder=args.max_iter,  # 使用命令行参数
        )
        
        full_q0 = q0_weighted
        logger.info(f"Full likelihood fit: S={s_weighted:.2f}, B={b_weighted:.2f}, sqrt(q0)={np.sqrt(full_q0):.2f}")
    except Exception as e:
        logger.warning(f"Error computing full_q0: {e}, using max significance squared")
        full_q0 = results.get('max_significance', 0.0) ** 2
    
    # 计算所有变体的显著性（如果启用）
    variations_data = {}
    if args.all_variations:
        logger.info("Computing significances for all variations (fit types × bin numbers)...")
        fits = ["cubic", "quintic", "septic"]
        degrees = [3, 5, 7]
        bins_list = [8, 12, 16]
        
        # 使用 finegrained FPR 阈值
        for i, (fit_name, degree) in enumerate(zip(fits, degrees)):
            for bin_num in bins_list:
                logger.info(f"  Computing {fit_name} (degree {degree}), {bin_num} bins...")
                
                # 重新计算 bins
                _, bins_SR_var, bins_left_var, bins_right_var, _, _, centers_SB_var = get_bins(
                    SR_left, SR_right, SB_left, SB_right, num_bins_SR=bin_num, binning="linear"
                )
                
                # 计算所有 finegrained FPR 阈值下的显著性
                sigs_list = []
                for t, fpr_thresh in enumerate(fpr_thresholds_finegrained):
                    if fpr_thresh >= 0.99:
                        continue  # 跳过 FPR >= 0.99
                    
                    # 找到对应的 score cutoff
                    best_cut_idx = np.argmin(np.abs(FPR_values - fpr_thresh))
                    score_cutoff = score_cut_points[best_cut_idx]
                    
                    # 过滤数据
                    filtered_masses_SBL_var = all_masses_physical["SBL"][all_scores["SBL"] >= score_cutoff]
                    filtered_masses_SR_var = all_masses_physical["SR"][all_scores["SR"] >= score_cutoff]
                    filtered_masses_SBH_var = all_masses_physical["SBH"][all_scores["SBH"] >= score_cutoff]
                    filtered_masses_var = np.concatenate([filtered_masses_SBL_var, filtered_masses_SR_var, filtered_masses_SBH_var])
                    
                    # 拟合背景
                    popt_var, pcov_var, chi2_var = fit_background(
                        filtered_masses_var,
                        fit_degree=degree,
                        SR_left=SR_left,
                        SR_right=SR_right,
                        plot_bins_left=bins_left_var,
                        plot_bins_right=bins_right_var,
                        plot_centers_SB=centers_SB_var,
                    )
                    
                    # 计算显著性
                    try:
                        S_var, B_var, q0_var = calculate_test_statistic(
                            filtered_masses_var,
                            SR_left=SR_left,
                            SR_right=SR_right,
                            SB_left=SB_left,
                            SB_right=SB_right,
                            num_bins=bin_num,
                            weights=None,
                            degree=degree,
                            starting_guess=popt_var,
                            verbose_plot=False,
                            max_iter_nelder=args.max_iter,
                        )
                        sig_var = np.sqrt(q0_var) if q0_var > 0 else 0.0
                    except Exception as e:
                        logger.warning(f"    Error computing significance for {fit_name}, {bin_num} bins, FPR={fpr_thresh:.4f}: {e}")
                        sig_var = 0.0
                    
                    sigs_list.append(sig_var)
                
                # 保存到字典（键格式："{degree}_{num_bins}"）
                key = f"{degree}_{bin_num}"
                variations_data[key] = np.array(sigs_list)
                logger.info(f"    Completed {fit_name}, {bin_num} bins: {len(sigs_list)} FPR points")
        
        logger.info(f"Completed all variations: {len(variations_data)} combinations")
    
    # 提取特征数据（用于 plot_features）
    logger.info("Extracting feature data...")
    feature_data = {
        "fpr_thresholds": fpr_thresholds,
        "features": []
    }
    
    # 对每个 FPR 阈值提取特征
    for t, threshold in enumerate(fpr_thresholds):
        best_cut_idx = np.argmin(np.abs(FPR_values - threshold))
        score_cutoff = score_cut_points[best_cut_idx]
        
        # 过滤 SR 数据
        SR_filtered = SR_data[SR_scores >= score_cutoff]
        
        # 提取特征值（排除 dimu_mass）
        feat_dict = {}
        for feat_name in feature_set_for_plot[:3]:  # 只提取前3个特征
            if feat_name in feature_set:
                feat_idx = feature_set.index(feat_name)
                feat_dict[feat_name] = SR_filtered[:, feat_idx]
        
        feature_data["features"].append(feat_dict)
    
    # 提取 isolation 数据（反隔离切割前的数据）
    isolation_data = {}
    for feat_name in feature_set_for_plot[:3]:
        if feat_name in feature_set:
            feat_idx = feature_set.index(feat_name)
            isolation_data[feat_name] = SR_data[:, feat_idx]
    
    # 保存结果
    output_dir = config.output_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = output_dir / f"bump_hunt_results_seed{args.seed}.pkl"
    logger.info(f"Saving results to: {results_path}")
    
    with open(results_path, "wb") as f:
        pickle.dump({
            "results": results,
            "SR_scores": SR_scores,
            "background_params": popt_single,
            "background_cov": pcov_single,
            "chi2": chi2_single,
            # 多个 FPR 阈值的结果（用于 plot_histograms_with_fits）
            "multi_fpr_data": save_data,
            "num_bins_SR": num_bins_SR,
            # 特征显著性数据（用于 plot_sig）
            "feature_sigs": feature_sigs,
            "random_sigs": random_sigs,
            "fpr_thresholds_finegrained": fpr_thresholds_finegrained,
            "full_q0": full_q0,
            # 特征分布数据（用于 plot_features）
            "feature_data": feature_data,
            "isolation_data": isolation_data,
            # 变体数据（用于 plot_variations）
            "variations_data": variations_data if args.all_variations else None,
        }, f)
    
    logger.info("Evaluation completed successfully")


if __name__ == "__main__":
    main()


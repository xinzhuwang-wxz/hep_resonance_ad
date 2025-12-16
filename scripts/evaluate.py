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
from resonance_ad.analysis import BumpHunter, fit_background
from resonance_ad.data import assemble_banded_datasets
from resonance_ad.physics.binning import get_bins

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate CATHODE and perform bump hunt")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--model-dir", type=str, default=None, help="Model directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--input", type=str, default=None, help="Input region data file")
    args = parser.parse_args()
    
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
        region_data = pickle.load(f)
    
    # 获取特征集合
    training_config = config.raw_config.get("training", {})
    feature_set_name = training_config.get("feature_set", "mix_0")
    feature_set = config.get_feature_set(feature_set_name)
    
    logger.info(f"Using feature set: {feature_set_name}")
    
    # 组装数据
    bands = ["SBL", "SR", "SBH"]
    banded_data = assemble_banded_datasets(region_data, feature_set, bands)
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    
    # 计算 anomaly scores
    logger.info("Computing anomaly scores...")
    bump_hunter = BumpHunter(config)
    
    SR_data = banded_data["SR"]
    SR_scores = bump_hunter.compute_anomaly_scores(cathode.model, SR_data, device)
    
    logger.info(f"SR data shape: {SR_data.shape}")
    logger.info(f"SR scores: mean={SR_scores.mean():.4f}, std={SR_scores.std():.4f}")
    
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
    
    # 拟合背景
    logger.info("Fitting background...")
    SB_data = np.vstack([banded_data["SBL"], banded_data["SBH"]])
    SB_masses = SB_data[:, -1]
    
    popt, pcov, chi2 = fit_background(
        SB_masses,
        fit_degree=5,  # 从配置读取
        SR_left=SR_left,
        SR_right=SR_right,
        plot_bins_left=bins_left,
        plot_bins_right=bins_right,
        plot_centers_SB=centers_SB,
    )
    
    logger.info(f"Background fit chi2: {chi2:.2f}")
    
    # 计算 SR 的背景期望值
    centers_SR = 0.5 * (bins_SR[1:] + bins_SR[:-1])
    from resonance_ad.analysis.bump_hunt import parametric_fit
    background_expectation = parametric_fit(centers_SR, *popt)
    
    # Bump hunt
    logger.info("Performing bump hunt...")
    results = bump_hunter.hunt_bump(
        SR_data, SR_scores, bins_SR, background_expectation
    )
    
    logger.info(f"Max significance: {results['max_significance']:.2f} sigma")
    logger.info(f"Max significance bin: {results['max_significance_bin']}")
    
    # 保存结果
    output_dir = config.output_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = output_dir / f"bump_hunt_results_seed{args.seed}.pkl"
    logger.info(f"Saving results to: {results_path}")
    
    with open(results_path, "wb") as f:
        pickle.dump({
            "results": results,
            "SR_scores": SR_scores,
            "background_params": popt,
            "chi2": chi2,
        }, f)
    
    logger.info("Evaluation completed successfully")


if __name__ == "__main__":
    main()


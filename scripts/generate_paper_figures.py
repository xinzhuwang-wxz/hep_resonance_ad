#!/usr/bin/env python
"""
生成论文图

生成论文中的所有关键图。
"""

import argparse
import pickle
import numpy as np
from pathlib import Path

from resonance_ad.core.config import load_config
from resonance_ad.core.logging import setup_logging, get_logger
from resonance_ad.plotting import PaperFigureGenerator
from resonance_ad.physics.binning import get_bins
from resonance_ad.analysis.bump_hunt import parametric_fit

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--input", type=str, default=None, help="Input region data file")
    parser.add_argument("--evaluation-results", type=str, default=None, help="Evaluation results file")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for figures")
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    setup_logging(log_dir=config.output_dir / "logs")
    
    logger.info(f"Generating paper figures with config: {args.config}")
    
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
    
    # 创建图生成器
    output_dir = Path(args.output_dir) if args.output_dir else config.output_dir / "figures"
    fig_gen = PaperFigureGenerator(config, output_dir=output_dir)
    
    # 获取窗口定义
    window = config.get_window()
    region_config = config.raw_config.get("region_selection", {})
    num_bins_SR = region_config.get("num_bins_SR", 12)
    
    _, bins_SR, bins_left, bins_right, _, _, centers_SB = get_bins(
        window["SR_left"],
        window["SR_right"],
        window["SB_left"],
        window["SB_right"],
        num_bins_SR,
        binning="linear",
    )
    bins_all = np.concatenate([bins_left[:-1], bins_SR, bins_right[1:]])
    
    # 1. 质量谱
    logger.info("Generating mass spectrum plot...")
    fig_gen.plot_mass_spectrum(
        region_data=region_data,
        bins=bins_all,
        window=window,
        save_path=output_dir / "mass_spectrum.pdf",
    )
    
    # 2. Score 分布（如果有评估结果）
    if args.evaluation_results:
        eval_path = Path(args.evaluation_results)
        if eval_path.exists():
            logger.info("Generating score distribution plot...")
            with open(eval_path, "rb") as f:
                eval_results = pickle.load(f)
            
            scores = {"SR": eval_results.get("SR_scores", np.array([]))}
            fig_gen.plot_anomaly_score_distribution(
                scores=scores,
                save_path=output_dir / "score_distribution.pdf",
            )
            
            # 3. 显著性图
            logger.info("Generating significance plot...")
            results = eval_results.get("results", {})
            if "significances" in results and "bins" in results:
                bins = results["bins"]
                centers = 0.5 * (bins[1:] + bins[:-1])
                significances = results["significances"]
                
                fig_gen.plot_significance(
                    mass_centers=centers,
                    significances=significances,
                    bins=bins,
                    save_path=output_dir / "significance.pdf",
                )
            
            # 4. Score vs Mass
            logger.info("Generating score vs mass plot...")
            if "SR_scores" in eval_results and "SR" in region_data:
                SR_masses = region_data["SR"]["dimu_mass"]
                SR_scores = eval_results["SR_scores"]
                
                fig_gen.plot_score_vs_mass(
                    mass=SR_masses,
                    score=SR_scores,
                    region="SR",
                    save_path=output_dir / "score_vs_mass.pdf",
                )
    
    # 5. 背景拟合图
    logger.info("Generating background fit plot...")
    if "SBL" in region_data and "SBH" in region_data:
        SB_masses = np.concatenate([
            region_data["SBL"]["dimu_mass"],
            region_data["SBH"]["dimu_mass"],
        ])
        
        # 如果有评估结果，使用拟合参数
        if args.evaluation_results:
            eval_path = Path(args.evaluation_results)
            if eval_path.exists():
                with open(eval_path, "rb") as f:
                    eval_results = pickle.load(f)
                
                if "background_params" in eval_results:
                    fit_params = eval_results["background_params"]
                    fig_gen.plot_background_fit(
                        masses=SB_masses,
                        bins=np.concatenate([bins_left, bins_right]),
                        fit_params=fit_params,
                        fit_function=parametric_fit,
                        window=window,
                        save_path=output_dir / "background_fit.pdf",
                    )
    
    logger.info(f"All figures saved to: {output_dir}")
    logger.info("Figure generation completed successfully")


if __name__ == "__main__":
    main()

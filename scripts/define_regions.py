#!/usr/bin/env python
"""
脚本 02: 定义区域（Sideband 和 Signal Region）

根据质量窗口定义，将数据划分为不同的区域。
"""

import argparse
import pickle
from pathlib import Path

from resonance_ad.core.config import load_config
from resonance_ad.core.logging import setup_logging, get_logger
from resonance_ad.data import RegionSelector

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Define sideband and signal regions")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input data file path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path",
    )
    parser.add_argument(
        "--charge-filter",
        type=str,
        default=None,
        choices=["OS", "SS"],
        help="Charge filter: OS (opposite-sign) or SS (same-sign)",
    )
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置日志
    setup_logging(log_dir=config.output_dir / "logs")
    logger.info(f"Defining regions with config: {args.config}")
    
    # 确定输入文件
    if args.input is None:
        data_config = config.raw_config.get("data", {})
        data_id = data_config.get("data_id", config.dataset_id)
        input_path = config.output_dir / "processed_data" / f"{data_id}_raw.pkl"
    else:
        input_path = Path(args.input)
    
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return
    
    # 加载数据
    logger.info(f"Loading data from: {input_path}")
    with open(input_path, "rb") as f:
        data = pickle.load(f)
    
    # 初始化区域选择器
    region_config = config.raw_config.get("region_selection", {})
    use_inner_bands = region_config.get("use_inner_bands", False)
    selector = RegionSelector(config, use_inner_bands=use_inner_bands)
    
    # 应用电荷筛选
    charge_filter = args.charge_filter or config.raw_config.get("data", {}).get("charge_filter")
    
    # 选择区域
    logger.info("Selecting regions...")
    region_data = selector.select_regions(data, charge_filter=charge_filter)
    
    # 添加派生特征
    logger.info("Adding derived features...")
    region_data = selector.add_derived_features(region_data)
    
    # 保存结果
    if args.output is None:
        charge_suffix = f"_{charge_filter}" if charge_filter else ""
        output_path = config.output_dir / "processed_data" / f"region_data{charge_suffix}.pkl"
    else:
        output_path = Path(args.output)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving region data to: {output_path}")
    with open(output_path, "wb") as f:
        pickle.dump(region_data, f)
    
    # 打印统计信息
    logger.info("Region selection completed successfully")
    logger.info("Region statistics:")
    import numpy as np
    for band in selector.bands:
        if band in region_data and "dimu_mass" in region_data[band]:
            mass_array = region_data[band]["dimu_mass"]
            # 确保是 numpy array（如果是 awkward array，转换为 numpy）
            if hasattr(mass_array, 'to_numpy'):
                mass_array = mass_array.to_numpy()
            elif not isinstance(mass_array, np.ndarray):
                mass_array = np.array(mass_array)
            
            n_events = len(mass_array)
            mass_mean = np.mean(mass_array)
            mass_std = np.std(mass_array)
            logger.info(
                f"  {band}: {n_events} events, "
                f"mass = {mass_mean:.3f} ± {mass_std:.3f} GeV"
            )


if __name__ == "__main__":
    main()


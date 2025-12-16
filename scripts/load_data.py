#!/usr/bin/env python
"""
脚本 01: 数据加载

从 pickle 文件加载数据，应用事件筛选，并保存处理后的数据。
"""

import argparse
import pickle
from pathlib import Path

from resonance_ad.core.config import load_config
from resonance_ad.core.logging import setup_logging, get_logger
from resonance_ad.data import DataLoader

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Load and preprocess data")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--data-id",
        type=str,
        default=None,
        help="Data ID (overrides config)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (overrides config)",
    )
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置日志
    setup_logging(log_dir=config.output_dir / "logs")
    logger.info(f"Loading data with config: {args.config}")
    
    # 初始化数据加载器
    loader = DataLoader(config)
    
    # 确定数据路径
    data_config = config.raw_config.get("data", {})
    data_id = args.data_id or data_config.get("data_id", config.dataset_id)
    
    data_path = config.data_dir / "precompiled_data" / data_id
    
    if not data_path.exists():
        logger.error(f"Data path not found: {data_path}")
        logger.info("Please ensure data files are available at the specified path")
        return
    
    # 加载数据
    logger.info(f"Loading data from: {data_path}")
    data = loader.load_from_pickle(data_path, data_id=data_id)
    
    # 保存数据
    output_path = args.output or (config.output_dir / "processed_data" / f"{data_id}_raw.pkl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving data to: {output_path}")
    with open(output_path, "wb") as f:
        pickle.dump(data, f)
    
    logger.info("Data loading completed successfully")
    logger.info(f"Data summary:")
    for key, value in data.items():
        if isinstance(value, (list, tuple)):
            logger.info(f"  {key}: {len(value)} entries")
        elif hasattr(value, 'shape'):
            logger.info(f"  {key}: shape {value.shape}")


if __name__ == "__main__":
    main()


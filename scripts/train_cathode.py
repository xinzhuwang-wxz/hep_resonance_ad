#!/usr/bin/env python
"""
训练 CATHODE 模型

从预处理的数据训练 CATHODE normalizing flow 模型。
"""

import argparse
import pickle
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split

from resonance_ad.core.config import load_config
from resonance_ad.core.logging import setup_logging, get_logger
from resonance_ad.models import DensityEstimator
from resonance_ad.models.training import train_cathode
from resonance_ad.data import assemble_banded_datasets
from resonance_ad.data.preprocessor import DataPreprocessor

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train CATHODE model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--input", type=str, default=None, help="Input region data file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs (overrides config)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (overrides config)")
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    setup_logging(log_dir=config.output_dir / "logs")
    
    logger.info(f"Training CATHODE with config: {args.config}")
    
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
    logger.info(f"Features: {feature_set}")
    
    # 组装数据
    bands = ["SBL", "SR", "SBH"]
    banded_data = assemble_banded_datasets(region_data, feature_set, bands)
    
    # 合并 sideband 数据用于训练
    SBL_data = banded_data["SBL"]
    SBH_data = banded_data["SBH"]
    SB_data = np.vstack([SBL_data, SBH_data])
    
    logger.info(f"SBL data shape: {SBL_data.shape}")
    logger.info(f"SBH data shape: {SBH_data.shape}")
    logger.info(f"SB data shape: {SB_data.shape}")
    
    # 预处理数据（按照原代码的方式）
    logger.info("Preprocessing data...")
    preprocessor = DataPreprocessor(cushion=0.02)  # 与原代码一致
    
    # 准备用于预处理的字典（从 SB 数据中提取）
    SB_data_dict = {}
    for i, feat in enumerate(feature_set):
        SB_data_dict[feat] = SB_data[:, i]
    
    # 预处理 SB 数据以计算预处理参数
    SB_processed, preproc_info = preprocessor.preprocess_features(
        SB_data_dict, feature_set, mass_key="dimu_mass"
    )
    
    # 应用预处理到所有区域的数据
    banded_data_processed = {}
    for band in bands:
        if band in banded_data:
            banded_data_processed[band] = preprocessor.apply_preprocessing(
                banded_data[band],
                feature_set,
                preproc_info,
                mass_scaler=preproc_info["dimu_mass"]["scaler"],
                mass_key="dimu_mass",
            )
    
    # 使用预处理后的数据
    SBL_data_processed = banded_data_processed["SBL"]
    SBH_data_processed = banded_data_processed["SBH"]
    
    # 训练/验证分割
    SBL_train, SBL_val = train_test_split(SBL_data_processed, test_size=0.2, random_state=args.seed)
    SBH_train, SBH_val = train_test_split(SBH_data_processed, test_size=0.2, random_state=args.seed)
    
    train_data = np.vstack([SBL_train, SBH_train])
    val_data = np.vstack([SBL_val, SBH_val])
    
    logger.info(f"Train data shape: {train_data.shape}")
    logger.info(f"Val data shape: {val_data.shape}")
    
    # 创建数据加载器
    batch_size = args.batch_size or training_config.get("batch_size", 256)
    train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(train_data))
    val_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(val_data))
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 创建模型
    model_config_path = config.working_dir / "configs" / training_config.get("config_file", "CATHODE_8.yml")
    num_features = len(feature_set) - 1  # 不包括 mass (条件输入)
    
    logger.info(f"Creating CATHODE model with {num_features} features")
    logger.info(f"Model config: {model_config_path}")
    
    cathode = DensityEstimator(
        config_path=model_config_path,
        num_inputs=num_features,
        device=device,
        verbose=True,
    )
    
    # 训练配置
    epochs = args.epochs or training_config.get("num_epochs", 100)
    
    # 保存目录
    save_dir = config.output_dir / "models" / f"seed{args.seed}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存特征集合信息
    with open(save_dir / "feature_set.txt", "w") as f:
        f.write(f"feature_set = {feature_set}\n")
    
    # 保存预处理信息（与原代码一致）
    processed_data_dir = config.output_dir / "processed_data"
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存 preprocessing_info（与原代码格式一致）
    preprocessing_info = {}
    mass_scaler = None
    for feat in feature_set:
        if feat == "dimu_mass":
            mass_scaler = preproc_info[feat]["scaler"]
        else:
            preprocessing_info[feat] = {
                "mean": preproc_info[feat]["mean"],
                "std": preproc_info[feat]["std"],
                "min": preproc_info[feat]["min"],
                "max": preproc_info[feat]["max"],
            }
    
    bootstrap_seed = args.seed  # 使用 seed 作为 bootstrap ID
    preprocessing_info_path = processed_data_dir / f"preprocessing_info_bootstrap{bootstrap_seed}"
    mass_scaler_path = processed_data_dir / f"mass_scaler_bootstrap{bootstrap_seed}"
    
    logger.info(f"Saving preprocessing info to: {preprocessing_info_path}")
    with open(preprocessing_info_path, "wb") as f:
        pickle.dump(preprocessing_info, f)
    
    logger.info(f"Saving mass scaler to: {mass_scaler_path}")
    with open(mass_scaler_path, "wb") as f:
        pickle.dump(mass_scaler, f)
    
    # 开始训练
    logger.info("Starting training...")
    train_cathode(
        model=cathode.model,
        optimizer=cathode.optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        savedir=save_dir,
        device=device,
        verbose=True,
    )
    
    logger.info("Training completed successfully")


if __name__ == "__main__":
    main()


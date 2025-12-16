"""
配置管理系统

从 YAML 文件加载配置，并提供类型安全的配置访问接口。
参考 bambooML 和 Made-With-ML 的设计风格。
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class Config:
    """配置类，提供类型安全的配置访问"""
    
    # 文件路径
    working_dir: Path
    data_dir: Path
    output_dir: Path
    
    # 分析关键词
    analysis_name: str
    particle: str
    dataset_id: str
    
    # 窗口定义（sideband 和 signal region）
    window_definitions: Dict[str, Dict[str, float]]
    
    # 特征集合
    feature_sets: Dict[str, list]
    
    # 分析切割
    analysis_cuts: Dict[str, Any] = field(default_factory=dict)
    
    # 其他配置
    raw_config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """确保路径是 Path 对象"""
        self.working_dir = Path(self.working_dir)
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_window(self, particle: Optional[str] = None) -> Dict[str, float]:
        """获取指定粒子的窗口定义"""
        particle = particle or self.particle
        if particle not in self.window_definitions:
            raise ValueError(f"Window definition for particle '{particle}' not found")
        return self.window_definitions[particle]
    
    def get_feature_set(self, name: str) -> list:
        """获取指定名称的特征集合"""
        if name not in self.feature_sets:
            raise ValueError(f"Feature set '{name}' not found")
        return self.feature_sets[name]


def load_config(config_path: str | Path) -> Config:
    """
    从 YAML 文件加载配置
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        Config 对象
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        raw_config = yaml.safe_load(f)
    
    # 提取配置
    file_paths = raw_config.get("file_paths", {})
    analysis_keywords = raw_config.get("analysis_keywords", {})
    
    working_dir = Path(file_paths.get("working_dir", "."))
    data_dir = Path(file_paths.get("data_storage_dir", "./data"))
    output_dir = working_dir / "outputs" / analysis_keywords.get("name", "default")
    
    config = Config(
        working_dir=working_dir,
        data_dir=data_dir,
        output_dir=output_dir,
        analysis_name=analysis_keywords.get("name", "default"),
        particle=analysis_keywords.get("particle", "upsilon"),
        dataset_id=analysis_keywords.get("dataset_id", "lowmass"),
        window_definitions=raw_config.get("window_definitions", {}),
        feature_sets=raw_config.get("feature_sets", {}),
        analysis_cuts=analysis_keywords.get("analysis_cuts", {}),
        raw_config=raw_config,
    )
    
    return config


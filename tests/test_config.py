"""
测试配置系统

验证配置加载和访问是否正常工作。
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from resonance_ad.core.config import load_config


def test_config_loading():
    """测试配置加载"""
    config_path = project_root / "configs" / "upsilon_reproduction.yaml"
    
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        print("Skipping config test")
        return
    
    config = load_config(config_path)
    
    # 验证基本属性
    assert config.analysis_name == "upsilon_iso_12_03"
    assert config.particle == "upsilon"
    assert config.dataset_id == "lowmass"
    
    # 验证窗口定义
    window = config.get_window()
    assert "SB_left" in window
    assert "SR_left" in window
    assert "SR_right" in window
    assert "SB_right" in window
    
    # 验证特征集合
    feature_set = config.get_feature_set("mix_0")
    assert "dimu_mass" in feature_set
    assert "dimu_pt" in feature_set
    
    print("✓ Config loading test passed")


if __name__ == "__main__":
    test_config_loading()
    print("All tests passed!")


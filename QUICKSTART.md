# 快速开始指南

## 项目概述

这是一个用于复现和扩展 CATHODE（dimuon anomaly detection）相关工作的研究工程平台。

## 安装

```bash
cd /Users/physicsboy/Documents/GitHub/hep_resonance_ad
pip install -e .
```

## 项目结构

```
hep_resonance_ad/
├── configs/              # 配置文件（YAML）
│   ├── upsilon_reproduction.yaml  # 主配置文件
│   └── CATHODE_8.yml     # CATHODE 模型配置
├── resonance_ad/         # 主代码包
│   ├── core/            # 核心功能（配置、日志）
│   ├── data/            # 数据加载和处理
│   ├── physics/         # 物理函数
│   ├── models/          # ML 模型（待实现）
│   ├── analysis/        # 分析流程（待实现）
│   └── plotting/        # 画图模块
├── scripts/             # 可执行脚本
│   ├── 01_load_data.py
│   ├── 02_define_regions.py
│   └── 05_generate_paper_figures.py
└── outputs/             # 输出目录（自动创建）
```

## Phase 1 使用流程

### 1. 准备数据

确保数据文件位于配置文件中指定的路径：
```yaml
file_paths:
  data_storage_dir: "./data"
```

数据应该按照以下结构组织：
```
data/
└── precompiled_data/
    └── {data_id}/
        ├── all_mu_{file_index}
        └── all_jet_{file_index}
```

### 2. 加载数据

```bash
python scripts/load_data.py --config configs/upsilon_reproduction.yaml
```

这将：
- 从 pickle 文件加载数据
- 应用事件筛选（至少 2 个 tight muon）
- 计算 dimuon 运动学量
- 保存处理后的数据到 `outputs/{analysis_name}/processed_data/`

### 3. 定义区域

```bash
python scripts/define_regions.py --config configs/upsilon_reproduction.yaml
```

这将：
- 根据质量窗口定义划分 sideband 和 signal region
- 应用电荷筛选（OS 或 SS）
- 添加派生特征（deltaR, deltaPT）
- 保存区域数据到 `outputs/{analysis_name}/processed_data/region_data_{OS|SS}.pkl`

### 4. 训练 CATHODE

```bash
python scripts/train_cathode.py --config configs/upsilon_reproduction.yaml --seed 42
```

这将：
- 加载区域数据
- 训练 CATHODE normalizing flow 模型
- 保存模型和训练历史到 `outputs/{analysis_name}/models/seed{seed}/`

### 5. 评估和 Bump Hunt

```bash
python scripts/evaluate.py --config configs/upsilon_reproduction.yaml --seed 42
```

这将：
- 加载训练好的模型
- 计算 anomaly scores
- 拟合背景并执行 bump hunt
- 保存结果到 `outputs/{analysis_name}/evaluation/`

### 6. 生成论文图

```bash
python scripts/generate_paper_figures.py --config configs/upsilon_reproduction.yaml --evaluation-results outputs/{analysis_name}/evaluation/bump_hunt_results_seed42.pkl
```

这将生成：
- Mass spectrum
- Anomaly score 分布
- Significance 图
- Score vs mass 图
- 背景拟合图

所有图保存到 `outputs/{analysis_name}/figures/`

## 配置说明

配置文件使用 YAML 格式，主要包含：

1. **文件路径**: 数据目录、工作目录等
2. **分析参数**: 粒子类型、数据集 ID、分析名称
3. **窗口定义**: Sideband 和 Signal Region 的质量范围
4. **特征集合**: 用于 ML 训练的特征列表
5. **模型配置**: CATHODE 模型参数
6. **训练配置**: 训练超参数、随机种子等

## 设计原则

- ✅ **Research-first**: 代码优先，notebook 仅用于探索
- ✅ **配置驱动**: 所有参数通过 YAML 配置
- ✅ **模块化**: 物理逻辑与 ML 逻辑分离
- ✅ **可扩展**: 易于添加新方法、新特征
- ✅ **可复现**: 显式设置随机种子，清晰的日志

## 当前状态

**Phase 1 已完成部分**:
- ✅ 项目基础结构
- ✅ 配置管理系统
- ✅ 数据加载模块
- ✅ 区域选择模块
- ✅ 数据预处理模块框架

**Phase 1 待完成**:
- 🚧 CATHODE 模型实现
- 🚧 训练脚本
- 🚧 Bump hunt 和显著性计算
- 🚧 论文图生成

详见 `PROJECT_STATUS.md`。

## 参考

- 原始仓库: `/Users/physicsboy/Documents/GitHub/dimuonAD`
- 设计参考: `bambooML`, `Made-With-ML`


# 脚本使用说明

## Phase 1: 论文复现流程

### 1. 数据加载

```bash
python scripts/load_data.py --config configs/upsilon_reproduction.yaml
```

从 pickle 文件加载原始数据，应用基本的事件筛选。

**输出**: `outputs/{analysis_name}/processed_data/{data_id}_raw.pkl`

### 2. 区域定义

```bash
python scripts/define_regions.py --config configs/upsilon_reproduction.yaml
```

根据质量窗口定义，将数据划分为 sideband 和 signal region。

**输出**: `outputs/{analysis_name}/processed_data/region_data_{OS|SS}.pkl`

### 3. 数据预处理

```bash
python scripts/preprocess_data.py --config configs/upsilon_reproduction.yaml
```

对数据进行预处理（logit transform, scaling等）。

### 4. 训练 CATHODE

```bash
python scripts/train_cathode.py --config configs/upsilon_reproduction.yaml
```

训练 CATHODE 模型。

### 5. 评估和 Bump Hunt

```bash
python scripts/evaluate.py --config configs/upsilon_reproduction.yaml
```

评估模型并进行 bump hunt。

### 6. 生成论文图

```bash
python scripts/generate_figures.py --config configs/upsilon_reproduction.yaml
```

生成论文中的所有关键图。

## 配置说明

所有脚本都接受 `--config` 参数，指定配置文件路径。

配置文件使用 YAML 格式，包含：
- 文件路径
- 分析参数
- 窗口定义
- 特征集合
- 模型配置
等。

## 输出目录结构

```
outputs/
└── {analysis_name}/
    ├── logs/              # 日志文件
    ├── processed_data/    # 处理后的数据
    ├── models/           # 保存的模型
    ├── predictions/       # 预测结果
    └── figures/           # 生成的图
```


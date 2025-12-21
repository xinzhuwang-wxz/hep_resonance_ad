# HEP Resonance Anomaly Detection

一个用于复现和扩展 CATHODE（dimuon anomaly detection）相关工作的研究工程平台。

## 项目目标

1. **完全复现论文结果** - Phase 1（最高优先级）
2. **代码结构清晰、模块化、可扩展**
3. **为后续研究预留接口** - Phase 2
4. **系统性诊断工具** - Phase 3
5. **通用化设计** - Phase 4

## 项目结构

```
hep_resonance_ad/
├── configs/              # 配置文件（YAML）
├── resonance_ad/         # 主代码包
│   ├── core/            # 核心功能（配置、日志、注册）
│   ├── data/            # 数据加载和处理
│   ├── physics/         # 物理相关函数
│   ├── models/          # ML 模型（Flow, Classifier等）
│   ├── analysis/        # 分析流程（bump hunt, significance等）
│   └── plotting/        # 画图模块
├── scripts/             # 可执行脚本
│   └── sub/             # SLURM作业提交脚本（云端集群）
├── outputs/             # 输出目录（自动创建）
├── requirements.txt     # Python依赖列表
└── tests/               # 测试代码
```

## 快速开始

详细使用说明请参考 [QUICKSTART.md](QUICKSTART.md)

### Phase 1: 论文复现

```bash
# 数据加载和预处理
python scripts/load_data.py --config configs/upsilon_reproduction.yaml

# Sideband 划分
python scripts/define_regions.py --config configs/upsilon_reproduction.yaml

# 训练 CATHODE
python scripts/train_cathode.py --config configs/upsilon_reproduction.yaml --seed 42

# 评估和 bump hunt
python scripts/evaluate.py --config configs/upsilon_reproduction.yaml --seed 42

# 生成论文图
python scripts/generate_figures.py --config configs/upsilon_reproduction.yaml \
    --evaluation-results outputs/{analysis_name}/evaluation/bump_hunt_results_seed42.pkl
```

## 脚本使用说明

### 本地运行

#### Phase 1: 论文复现流程

##### 1. 数据加载

```bash
python scripts/load_data.py --config configs/upsilon_reproduction.yaml
```

从 pickle 文件加载原始数据，应用基本的事件筛选。

**输出**: `outputs/{analysis_name}/processed_data/{data_id}_raw.pkl`

##### 2. 区域定义

```bash
python scripts/define_regions.py --config configs/upsilon_reproduction.yaml
```

根据质量窗口定义，将数据划分为 sideband 和 signal region。

**输出**: `outputs/{analysis_name}/processed_data/region_data_{OS|SS}.pkl`

##### 3. 数据预处理

```bash
python scripts/preprocess_data.py --config configs/upsilon_reproduction.yaml
```

对数据进行预处理（logit transform, scaling等）。

##### 4. 训练 CATHODE

```bash
python scripts/train_cathode.py --config configs/upsilon_reproduction.yaml
```

训练 CATHODE 模型。

##### 5. 评估和 Bump Hunt

```bash
python scripts/evaluate.py --config configs/upsilon_reproduction.yaml
```

评估模型并进行 bump hunt。

##### 6. 生成论文图

```bash
python scripts/generate_figures.py --config configs/upsilon_reproduction.yaml
```

生成论文中的所有关键图。

#### 配置说明

所有脚本都接受 `--config` 参数，指定配置文件路径。

配置文件使用 YAML 格式，包含：
- 文件路径
- 分析参数
- 窗口定义
- 特征集合
- 模型配置
等。

#### 输出目录结构

```
outputs/
└── {analysis_name}/
    ├── logs/              # 日志文件
    ├── processed_data/    # 处理后的数据
    ├── models/           # 保存的模型
    ├── predictions/       # 预测结果
    └── figures/           # 生成的图
```

### 云端集群运行 (SLURM)

项目支持在GPU集群上运行完整的pipeline。SLURM作业提交脚本位于 `scripts/sub/` 目录。

#### 可用脚本

##### 1. `scripts/sub/train_cathode.sh`
训练 CATHODE normalizing flow 模型。
- **GPU**: 1 GPU required
- **Time**: ~24 hours
- **Memory**: 32GB
- **CPUs**: 8

##### 2. `scripts/sub/evaluate.sh`
执行模型评估和 bump hunt 分析。
- **GPU**: 1 GPU required
- **Time**: ~12 hours
- **Memory**: 16GB
- **CPUs**: 4

##### 3. `scripts/sub/generate_figures.sh`
从评估结果生成所有论文图。
- **GPU**: 1 GPU required (用于ROC曲线生成)
- **Time**: ~6 hours
- **Memory**: 16GB
- **CPUs**: 4

#### 集群使用方法

1. **修改路径**: 更新脚本中的conda环境路径和工作目录：
   ```bash
   source /hpcfs/cepc/higgsgpu/wangxinzhu/miniconda3/bin/activate
   conda activate strange
   cd /hpcfs/cepc/higgsgpu/wangxinzhu/hep_resonance_ad
   ```

2. **调整参数**: 根据需要修改脚本中的命令行参数。

3. **提交作业**:
   ```bash
   sbatch scripts/sub/train_cathode.sh
   sbatch scripts/sub/evaluate.sh
   sbatch scripts/sub/generate_figures.sh
   ```

4. **监控作业**:
   ```bash
   squeue -u $USER  # 检查作业状态
   scancel <job_id>  # 如需要取消作业
   ```

#### 集群依赖

确保conda环境包含 `requirements.txt` 中的所有依赖：

```bash
conda create -n strange python=3.10
conda activate strange
pip install -r requirements.txt
```

#### 集群输出

- 日志保存在 `./logs/` 目录
- 结果按时间戳保存在 `./outputs/` 目录
- 检查日志文件获取详细进度和错误信息

#### 注意事项

- 脚本配置用于 `gpu` 分区和 `higgsgpu` 账户
- GPU内存分配针对具体工作负载优化
- 环境变量设置为最佳PyTorch性能
- CUDA设备检测由Python脚本自动处理

## 设计原则

- **Research-first, not notebook-first**
- **强配置驱动（YAML）**
- **物理逻辑与 ML 逻辑分离**
- **不硬编码路径、参数、超参数**
- **每一张论文图都能通过一个脚本命令生成**
- **易于 debug、审查、扩展**

## 开发阶段

- [x] **Phase 1: 论文复现** - ✅ 已完成
- [ ] **Phase 2: 方法扩展** - 🚧 进行中
- [ ] **Phase 3: 系统性诊断** - 📋 计划中
- [ ] **Phase 4: 通用化** - 📋 计划中

详细开发计划请参考 [DEVELOPMENT.md](DEVELOPMENT.md)

## 文档

- [快速开始指南](QUICKSTART.md) - 快速上手使用
- [开发文档](DEVELOPMENT.md) - 详细的开发计划和验收标准（807行）
- [项目状态](PROJECT_STATUS.md) - 当前项目状态
- [Phase 1 完成总结](PHASE1_COMPLETE.md) - Phase 1 完成情况

## 文档

- [快速开始指南](QUICKSTART.md) - 快速上手使用
- [开发文档](DEVELOPMENT.md) - 详细的开发计划和验收标准（807行）
- [项目状态](PROJECT_STATUS.md) - 当前项目状态
- [Phase 1 完成总结](PHASE1_COMPLETE.md) - Phase 1 完成情况


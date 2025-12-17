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
├── outputs/             # 输出目录（自动创建）
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


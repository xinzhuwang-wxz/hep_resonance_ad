# 项目状态

## Phase 1: 论文复现（当前阶段）

### ✅ 已完成

1. **项目基础结构**
   - 创建了清晰的项目目录结构
   - 实现了配置管理系统（YAML 驱动）
   - 实现了日志系统

2. **核心模块**
   - `resonance_ad/core/`: 配置和日志管理
   - `resonance_ad/physics/`: 物理函数（运动学、binning）
   - `resonance_ad/data/`: 数据加载和处理
     - `DataLoader`: 从 pickle 文件加载数据
     - `RegionSelector`: Sideband 和 Signal Region 划分
     - `DataPreprocessor`: 数据预处理（logit transform, scaling）

3. **脚本框架**
   - `scripts/01_load_data.py`: 数据加载脚本
   - `scripts/02_define_regions.py`: 区域定义脚本
   - `scripts/05_generate_paper_figures.py`: 图生成脚本框架

4. **配置文件**
   - `configs/upsilon_reproduction.yaml`: 主配置文件
   - `configs/CATHODE_8.yml`: CATHODE 模型配置

### ✅ Phase 1 已完成

1. **数据预处理模块**
   - `resonance_ad/data/preprocessor.py`: 完整的数据预处理流程

2. **模型模块**
   - `resonance_ad/models/flows.py`: Normalizing Flow 实现
   - `resonance_ad/models/cathode.py`: CATHODE 架构
   - `resonance_ad/models/training.py`: 训练循环

3. **训练脚本**
   - `scripts/train_cathode.py`: CATHODE 训练脚本

4. **分析模块**
   - `resonance_ad/analysis/bump_hunt.py`: Bump hunt 实现
   - `resonance_ad/analysis/significance.py`: 显著性计算

5. **评估脚本**
   - `scripts/evaluate.py`: 模型评估和 bump hunt

6. **画图模块**
   - `resonance_ad/plotting/paper_figures.py`: 实现所有论文图的绘制
   - `scripts/generate_paper_figures.py`: 图生成脚本

### 📋 下一步工作

1. **完善数据加载**
   - 测试数据加载流程
   - 处理边界情况（缺失文件、空数据等）

2. **实现 CATHODE 模型**
   - 参考原始仓库的 `helpers/flows.py` 和 `helpers/DNN.py`
   - 实现 MAF (Masked Autoregressive Flow)
   - 实现训练循环

3. **实现 Bump Hunt**
   - 背景拟合（多项式拟合）
   - Anomaly score 计算
   - 显著性计算

4. **实现画图功能**
   - Mass spectrum
   - Score 分布
   - Significance 图
   - 参考原始仓库的 `helpers/plotting.py`

## Phase 2-4: 未来扩展

- Phase 2: 方法扩展（CWoLa, SALAD, LaCATHODE, Diffusion）
- Phase 3: 系统性诊断工具
- Phase 4: 通用化设计

## 设计原则遵守情况

- ✅ Research-first, not notebook-first
- ✅ 强配置驱动（YAML）
- ✅ 物理逻辑与 ML 逻辑分离
- ✅ 不硬编码路径、参数、超参数
- 🚧 每一张论文图都能通过一个脚本命令生成（框架已就绪）
- ✅ 易于 debug、审查、扩展


# SLURM 作业提交脚本

本目录包含用于在集群上运行完整分析流程的 SLURM 脚本。

## 文件说明

### 1. `run_pipeline.sh` - 完整流程脚本（提交到集群）

这个脚本完成从数据加载到评估的完整流程：

- **数据下载**（可选，如果数据已存在可跳过）
- **数据加载** (`load_data.py`)
- **区域定义** (`define_regions.py`)
- **CATHODE 训练** (`train_cathode.py`)
- **评估和 Bump Hunt** (`evaluate.py`)

**使用方法：**

```bash
# 推荐：从项目根目录提交（确保路径正确）
cd /path/to/hep_resonance_ad
sbatch sub/run_pipeline.sh

# 或者：从 sub 目录提交（也可以工作，但日志路径可能不同）
cd /path/to/hep_resonance_ad/sub
sbatch run_pipeline.sh
# 注意：如果从 sub 目录提交，日志会写到 sub/logs/ 而不是 logs/

# 查看作业状态
squeue -u $USER

# 查看作业输出（根据提交位置选择）
tail -f logs/run_pipeline_<JOB_ID>.log
# 或
tail -f sub/logs/run_pipeline_<JOB_ID>.log
```

**重要提示：**

- **推荐从项目根目录提交**：`sbatch sub/run_pipeline.sh`
  - 日志会正确写到 `logs/` 目录
  - 所有相对路径都正确
  
- **如果从 sub 目录提交**：`sbatch run_pipeline.sh`
  - 脚本会自动切换到项目根目录执行
  - 但 SLURM 日志会写到 `sub/logs/`（因为 `#SBATCH --output` 在提交时解析）
  - 功能不受影响，只是日志位置不同

**配置修改：**

在提交前，请根据实际情况修改脚本中的以下参数：

1. **Conda 环境路径**（第 28-30 行）：
   ```bash
   CONDA_BASE="/hpcfs/cepc/higgsgpu/wangxinzhu/miniconda3"
   conda activate strange
   ```

2. **SLURM 资源请求**（第 5-12 行）：
   - `--partition`: 分区名称
   - `--account`: 账户名称
   - `--gres`: GPU 资源（如 `gpu:v100:1`）
   - `--time`: 作业时间限制

3. **运行参数**（第 40-60 行）：
   - `SEED`: 随机种子（默认 42）
   - `FAST_MODE`: 是否使用快速模式（`true`/`false`）
   - `ALL_VARIATIONS`: 是否计算所有variations（`true`/`false`）
   - `EPOCHS`: 训练轮数（留空使用配置文件默认值）
   - `MAX_ITER`: 评估最大迭代次数（留空使用默认值）

### 2. `plot.sh` - 画图脚本（手动运行，不需要提交）

在 `run_pipeline.sh` 完成后，使用此脚本生成所有论文图。

**使用方法：**

```bash
# 在服务器上直接运行（不需要提交到SLURM）
bash sub/plot.sh
```

**配置修改：**

在运行前，请根据实际情况修改脚本中的以下参数：

1. **Conda 环境路径**（第 10-11 行）：
   ```bash
   CONDA_BASE="/hpcfs/cepc/higgsgpu/wangxinzhu/miniconda3"
   conda activate strange
   ```

2. **评估结果路径**（第 25 行）：
   ```bash
   EVAL_RESULTS=""  # 留空自动检测，或手动指定路径
   ```
   如果留空，脚本会自动查找 `outputs/*/evaluation/bump_hunt_results_seed{SEED}.pkl`

3. **画图参数**（第 28-35 行）：
   - `SEED`: 随机种子（必须与 `run_pipeline.sh` 中的一致）
   - `FIT_TYPE`: 拟合类型（`cubic`, `quintic`, `septic`）
   - `NUM_BINS_SR`: SR bin 数量（8, 12, 16）
   - `INCLUDE_TRAINING_PLOTS`: 是否包含训练损失图（`true`/`false`）

## 完整工作流程

### 步骤 1: 提交完整流程作业

```bash
# 修改 run_pipeline.sh 中的配置参数
vim sub/run_pipeline.sh

# 提交作业
sbatch sub/run_pipeline.sh

# 查看作业状态
squeue -u $USER
```

### 步骤 2: 等待作业完成

作业完成后，检查输出文件：

```bash
# 查看日志
cat logs/run_pipeline_<JOB_ID>.log

# 检查评估结果是否生成
ls -lh outputs/*/evaluation/bump_hunt_results_seed*.pkl
```

### 步骤 3: 生成图片

```bash
# 修改 plot.sh 中的配置参数（如果需要）
vim sub/plot.sh

# 运行画图脚本
bash sub/plot.sh
```

## 输出文件

### `run_pipeline.sh` 输出：

- **日志文件**: `logs/run_pipeline_<JOB_ID>.log`
- **错误日志**: `logs/run_pipeline_<JOB_ID>.err`
- **评估结果**: `outputs/{analysis_name}/evaluation/bump_hunt_results_seed{SEED}.pkl`
- **模型文件**: `outputs/{analysis_name}/models/seed{SEED}/`

**注意**: 输出路径基于 `analysis_keywords.name`（如 `upsilon_iso_12_21`），而不是 `dataset_id`（`lowmass`）。`dataset_id` 仅用于数据输入路径。

### `plot.sh` 输出：

- **所有论文图**: `outputs/{analysis_name}/figures/`
  - `histogram_cuts.pdf` - Cut histograms
  - `features.pdf` - Feature distributions
  - `significance.pdf` - Significance vs FPR
  - `training_losses.pdf` - Training/validation losses
  - `roc_curve.pdf` - ROC curve
  - `significance_variations.pdf` - Variations plot（如果使用了 `--all-variations`）

## 注意事项

1. **环境配置**: 确保 `run_pipeline.sh` 和 `plot.sh` 中的 conda 环境路径一致
2. **随机种子**: `plot.sh` 中的 `SEED` 必须与 `run_pipeline.sh` 中的一致
3. **数据路径**: 如果数据不在默认位置，需要修改数据下载或加载步骤
4. **GPU 资源**: 根据实际需求调整 `--gres` 参数
5. **时间限制**: 根据数据量和计算资源调整 `--time` 参数
   - 快速模式（`FAST_MODE=true`）: 约 2-4 小时
   - 完整模式（`FAST_MODE=false`）: 约 6-12 小时

## 故障排除

### 作业失败

1. 检查日志文件：`cat logs/run_pipeline_<JOB_ID>.err`
2. 检查 conda 环境是否正确激活
3. 检查数据文件是否存在
4. 检查 GPU 是否可用：`nvidia-smi`

### 画图失败

1. 检查评估结果文件是否存在
2. 检查 `SEED` 是否与训练时一致
3. 检查输出目录是否有写权限

## 快速测试

如果只想快速测试流程，可以设置：

```bash
# 在 run_pipeline.sh 中
FAST_MODE="true"
ALL_VARIATIONS="false"
EPOCHS="10"  # 减少训练轮数
```

这样可以快速验证流程是否正确，但结果精度会降低。


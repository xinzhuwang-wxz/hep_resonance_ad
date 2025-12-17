# 评估流程对齐检查

## 原代码流程 (`07_significances.py`)

### 1. 数据加载
- 从 pickle 文件加载预处理后的数据 (`all_test_data_splits`)
- 加载 scores (`all_scores_splits`)
- 加载预处理信息 (`mass_scaler`, `preprocessing_info`)

### 2. 质量值转换
```python
all_masses = mass_scalar.inverse_transform(all_data[:,-1].reshape(-1,1))
```
- 将预处理后的质量值转换回物理值
- 使用物理值判断区域（SBL/SR/SBH）

### 3. FPR 阈值计算
```python
fpr_thresholds = [1, 0.25, 0.1, 0.05, 0.01, 0.005]  # 粗粒度
fpr_thresholds_finegrained = np.logspace(0, -3, 25)  # 细粒度
```

### 4. Score Cutoff 计算
```python
feature_cut_points = np.linspace(np.min(all_scores), np.max(all_scores), 10000)
FPR = []
for cut in feature_cut_points:
    FPR.append((np.sum(feature_SBH >= cut)+np.sum(feature_SBL >= cut))/(len(feature_SBH)+len(feature_SBL)))
```
- 基于 SB 区域的 scores 计算 FPR

### 5. 显著性计算流程
对于每个 FPR 阈值：
1. 找到对应的 score cutoff
2. 过滤数据（基于 score cutoff）
3. 拟合背景：`curve_fit_m_inv`
4. 计算显著性：`calculate_test_statistic(filtered_masses, ..., starting_guess=popt)`

### 6. Likelihood Reweighting (FPR=1.0 时)
```python
if threshold == 1.0:
    mu = (S) / (S + B)
    likelihood_ratios = (all_scores) / (1 - all_scores)
    weights = (likelihood_ratios - (1-mu)) / mu
    weights = np.clip(weights, 0, 1e9)
    # 使用 weights 进行 weighted fit
    popt, pcov, chi2, y_vals, n_dof = curve_fit_m_inv(..., weights=weights)
    s, b, bonus_q0, popt = calculate_test_statistic(..., weights=weights, ...)
```

## 我们的代码流程 (`evaluate.py`)

### ✅ 已对齐的部分

1. **数据加载**：从 `region_data_original` 加载原始数据
2. **预处理应用**：使用 `DataPreprocessor` 应用预处理
3. **物理质量值获取**：直接从 `region_data_original` 获取（等价于 `inverse_transform`）
4. **FPR 阈值**：使用相同的粗粒度阈值 `[1.0, 0.25, 0.1, 0.05, 0.01, 0.005]`
5. **Score Cutoff 计算**：基于 SB 区域的 scores 计算 FPR
6. **显著性计算**：使用 `calculate_test_statistic` 和 `fit_background`

### ⚠️ 需要检查的部分

1. **Likelihood Reweighting**：
   - 原代码：`likelihood_ratios = (all_scores) / (1 - all_scores)`
   - 我们的代码：`likelihood_ratios = all_scores_full / (1 - all_scores_full + 1e-10)`
   - **差异**：我们添加了 `1e-10` 防止除零，但原代码没有
   - **影响**：可能在小 scores 时有细微差异

2. **Weights 计算**：
   - 原代码：`weights = (likelihood_ratios - (1-mu)) / mu`
   - 我们的代码：`weights = (likelihood_ratios - (1 - mu)) / (mu + 1e-10) if mu > 0 else np.ones_like(...)`
   - **差异**：我们添加了 `1e-10` 和 `mu > 0` 检查
   - **影响**：更稳健，但可能略有不同

3. **Weights Clipping**：
   - 原代码：`weights = np.clip(weights, 0, 1e9)`
   - 我们的代码：需要检查是否有 clipping

4. **细粒度 FPR 阈值**：
   - 原代码：`fpr_thresholds_finegrained = np.logspace(0, -3, 25)`
   - 我们的代码：需要确认是否使用相同的细粒度阈值

## 建议修复

1. **移除 `1e-10`**：如果原代码没有，我们也应该移除（除非会导致数值问题）
2. **添加 weights clipping**：确保 weights 被 clip 到 `[0, 1e9]`
3. **确认细粒度 FPR 阈值**：使用 `np.logspace(0, -3, 25)`


"""
XGBoost/BDT 训练模块

实现论文中的Step 3：训练分类器区分SR中的真实数据和生成的背景样本

使用XGBoost作为默认分类器（优先，更稳定），支持通过配置选择其他分类器。
"""

import numpy as np
import xgboost as xgb
from sklearn.utils import shuffle
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt

from resonance_ad.core.logging import get_logger

logger = get_logger(__name__)


class BDTTrainer:
    """
    XGBoost/BDT 训练器

    在SR中训练分类器区分真实数据和flow生成的背景样本。
    默认使用XGBoost（优先，更稳定）。
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化BDT训练器

        Args:
            config: BDT配置字典，支持以下选项：
                - classifier_type: "xgboost" (默认，推荐) 或 "bdt"
                - n_estimators: 树的数量
                - max_depth: 树的最大深度
                - learning_rate: 学习率
                - subsample: 子样本比例
                - early_stopping_rounds: 早停轮数
                - n_ensemble: ensemble模型数量
                - num_folds: 交叉验证折数
        """
        self.config = config
        self.logger = get_logger(__name__)

        # 分类器类型（默认XGBoost，优先使用）
        self.classifier_type = config.get("classifier_type", "xgboost").lower()
        if self.classifier_type not in ["xgboost", "bdt"]:
            self.logger.warning(f"Unknown classifier_type '{self.classifier_type}', defaulting to 'xgboost'")
            self.classifier_type = "xgboost"

        if self.classifier_type == "xgboost":
            self.logger.info("Using XGBoost classifier (recommended, more stable)")
        else:
            self.logger.info(f"Using {self.classifier_type} classifier")

        # 默认XGBoost超参数（基于原始论文）
        self.default_params = {
            "n_estimators": 100,
            "max_depth": 3,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "early_stopping_rounds": 10,
            "n_ensemble": 10,  # ensemble训练的树数量
            "num_folds": 5,    # k-fold交叉验证
        }

        # 更新默认参数
        self.hyperparams = {**self.default_params, **config}

    def train_bdt_on_sr(
        self,
        sr_data: np.ndarray,
        sr_generated_background: np.ndarray,
        sb_data: np.ndarray,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, Dict[int, List[Any]]]:
        """
        在SR中训练BDT分类器（使用5-fold交叉验证）
        
        完全匹配原始代码的 run_BDT_bump_hunt 函数逻辑
        
        Args:
            sr_data: SR中的真实数据 (n_events, n_features)
            sr_generated_background: SR中flow生成的背景样本 (n_events, n_features)
            sb_data: SB中的真实数据 (n_events, n_features) - 用于测试集
            random_state: 随机种子
            
        Returns:
            sr_scores: SR数据的BDT概率分数数组，按原始数据顺序排列
            sb_scores: SB数据的BDT概率分数数组，按原始数据顺序排列
            trained_models_by_fold: 按fold组织的训练好的模型列表 {fold_id: [models]}
        """
        self.logger.info(f"开始{self.classifier_type.upper()}训练（5-fold交叉验证）...")
        
        num_folds = self.hyperparams["num_folds"]
        n_ensemble = self.hyperparams["n_ensemble"]
        
        # 1. 打乱数据（与原始代码一致）
        flow_samples_SR = shuffle(sr_generated_background, random_state=random_state)
        data_samples_SR = shuffle(sr_data, random_state=random_state + 1)
        data_samples_SB = shuffle(sb_data, random_state=random_state + 2)
        
        # 记录原始索引（用于后续重新排序）
        flow_indices = np.arange(len(sr_generated_background))
        data_indices = np.arange(len(sr_data))
        sb_indices = np.arange(len(sb_data))
        
        flow_indices_shuffled = shuffle(flow_indices, random_state=random_state)
        data_indices_shuffled = shuffle(data_indices, random_state=random_state + 1)
        sb_indices_shuffled = shuffle(sb_indices, random_state=random_state + 2)
        
        # 2. 分割成fold
        flow_SR_splits = np.array_split(flow_samples_SR, num_folds)
        data_SR_splits = np.array_split(data_samples_SR, num_folds)
        data_SB_splits = np.array_split(data_samples_SB, num_folds)
        
        # 记录每个fold对应的原始索引
        flow_indices_splits = np.array_split(flow_indices_shuffled, num_folds)
        data_indices_splits = np.array_split(data_indices_shuffled, num_folds)
        sb_indices_splits = np.array_split(sb_indices_shuffled, num_folds)
        
        # 3. 存储结果
        scores_splits = {}  # {fold_id: scores_array}
        test_indices_splits = {}  # {fold_id: original_indices}
        trained_models_by_fold = {}  # {fold_id: [models]}
        
        # 4. 对每个fold进行训练
        for i_fold in range(num_folds):
            self.logger.info(f"训练 Fold {i_fold + 1}/{num_folds}...")
            
            # 4.1 组装训练/验证/测试数据
            training_data, training_labels = [], []
            validation_data, validation_labels = [], []
            testing_data = []
            test_indices = []
            
            for ii in range(num_folds):
                if ii == i_fold:
                    # 测试集：SR数据 + SB数据
                    testing_data.append(data_SR_splits[ii])
                    testing_data.append(data_SB_splits[ii])
                    test_indices.append(('SR', data_indices_splits[ii]))
                    test_indices.append(('SB', sb_indices_splits[ii]))
                elif ((ii+1) % num_folds) == i_fold:
                    # 验证集：flow SR + real SR
                    validation_data.append(flow_SR_splits[ii])
                    validation_labels.append(np.zeros((flow_SR_splits[ii].shape[0], 1)))
                    validation_data.append(data_SR_splits[ii])
                    validation_labels.append(np.ones((data_SR_splits[ii].shape[0], 1)))
                else:
                    # 训练集：flow SR + real SR
                    training_data.append(flow_SR_splits[ii])
                    training_labels.append(np.zeros((flow_SR_splits[ii].shape[0], 1)))
                    training_data.append(data_SR_splits[ii])
                    training_labels.append(np.ones((data_SR_splits[ii].shape[0], 1)))
            
            # 4.2 合并数据（排除质量特征，最后一列）
            X_train_fold = np.concatenate(training_data)[:, :-1]
            Y_train_fold = np.concatenate(training_labels).flatten()
            X_val_fold = np.concatenate(validation_data)[:, :-1]
            Y_val_fold = np.concatenate(validation_labels).flatten()
            X_test_fold = np.concatenate(testing_data)[:, :-1]
            
            # 4.3 计算类别权重（与原始代码一致）
            class_weight = {0: 1.0, 1: np.sum(Y_train_fold == 0) / np.sum(Y_train_fold == 1)}
            w_train_fold = np.array([class_weight[int(y)] for y in Y_train_fold])
            w_val_fold = np.array([class_weight[int(y)] for y in Y_val_fold])
            
            # 4.4 打乱数据
            X_train_fold, Y_train_fold, w_train_fold = shuffle(
                X_train_fold, Y_train_fold, w_train_fold, random_state=random_state + i_fold
            )
            X_val_fold, Y_val_fold, w_val_fold = shuffle(
                X_val_fold, Y_val_fold, w_val_fold, random_state=random_state + i_fold + 100
            )
            
            # 4.5 训练ensemble模型
            scores_fold = np.empty((X_test_fold.shape[0], n_ensemble))
            trained_models_fold = []
            
            for i_tree in range(n_ensemble):
                if i_tree % 10 == 0:
                    self.logger.info(f"  Fold {i_fold + 1}, 模型 {i_tree + 1}/{n_ensemble}")
                
                # 随机种子：与原始代码完全一致
                random_seed = i_fold * n_ensemble + i_tree + 1
                
                xgb_params = {
                    'n_estimators': self.hyperparams["n_estimators"],
                    'max_depth': self.hyperparams["max_depth"],
                    'learning_rate': self.hyperparams["learning_rate"],
                    'subsample': self.hyperparams["subsample"],
                    'early_stopping_rounds': self.hyperparams["early_stopping_rounds"],
                    'objective': 'binary:logistic',
                    'random_state': random_seed,
                    'eval_metric': 'logloss',
                    'verbosity': 0,
                }
                
                model = xgb.XGBClassifier(**xgb_params)
                eval_set = [(X_train_fold, Y_train_fold), (X_val_fold, Y_val_fold)]
                
                model.fit(
                    X_train_fold, Y_train_fold,
                    sample_weight=w_train_fold,
                    eval_set=eval_set,
                    sample_weight_eval_set=[w_train_fold, w_val_fold],
                    verbose=False
                )
                
                trained_models_fold.append(model)
                
                # 使用best_iteration进行预测（与原始代码一致）
                best_iteration = model.best_iteration if model.best_iteration is not None else self.hyperparams["n_estimators"]
                scores_fold[:, i_tree] = model.predict_proba(
                    X_test_fold, 
                    iteration_range=(0, best_iteration)
                )[:, 1]
            
            # 4.6 平均ensemble分数（与原始代码一致，take_ensemble_avg=True）
            scores_splits[i_fold] = np.mean(scores_fold, axis=1)
            test_indices_splits[i_fold] = test_indices
            trained_models_by_fold[i_fold] = trained_models_fold
            
            self.logger.info(f"Fold {i_fold + 1} 完成，测试集大小: {len(scores_splits[i_fold])}")
        
        # 5. 合并所有fold的分数，按原始数据顺序排列
        self.logger.info("合并所有fold的分数...")
        
        # 创建完整的分数数组（SR数据 + SB数据）
        n_sr = len(sr_data)
        n_sb = len(sb_data)
        total_scores = np.zeros(n_sr + n_sb)
        
        # 合并SR和SB的分数
        for i_fold in range(num_folds):
            scores_fold = scores_splits[i_fold]
            indices_fold = test_indices_splits[i_fold]
            
            idx_in_fold = 0
            for region_type, indices in indices_fold:
                if region_type == 'SR':
                    # SR数据的分数
                    for orig_idx in indices:
                        total_scores[orig_idx] = scores_fold[idx_in_fold]
                        idx_in_fold += 1
                elif region_type == 'SB':
                    # SB数据的分数（放在SR之后）
                    for orig_idx in indices:
                        total_scores[n_sr + orig_idx] = scores_fold[idx_in_fold]
                        idx_in_fold += 1
        
        # 分离SR和SB的分数
        sr_scores = total_scores[:n_sr]
        sb_scores = total_scores[n_sr:]
        
        self.logger.info(f"XGBoost训练完成，共 {num_folds} folds × {n_ensemble} 模型")
        self.logger.info(f"SR数据分数: mean={sr_scores.mean():.4f}, std={sr_scores.std():.4f}")
        self.logger.info(f"SB数据分数: mean={sb_scores.mean():.4f}, std={sb_scores.std():.4f}")
        
        return sr_scores, sb_scores, trained_models_by_fold

    def predict_proba_ensemble(
        self,
        X: np.ndarray,
        models: List[Any],
        method: str = "mean",
        use_best_iteration: bool = True
    ) -> np.ndarray:
        """
        使用ensemble模型进行概率预测

        Args:
            X: 输入特征 (n_samples, n_features)
            models: 训练好的模型列表
            method: 集成方法 ("mean", "median", 或 "best")
            use_best_iteration: 是否使用best_iteration进行预测（与原始代码一致）

        Returns:
            预测概率数组 (n_samples,) - 返回正类概率
        """
        if not models:
            raise ValueError("没有训练好的模型")

        # 收集所有模型的预测
        all_predictions = []

        for model in models:
            # 使用best_iteration进行预测（与原始代码一致）
            if use_best_iteration and hasattr(model, 'best_iteration') and model.best_iteration is not None:
                proba = model.predict_proba(
                    X, 
                    iteration_range=(0, model.best_iteration)
                )[:, 1]
            else:
                # 如果没有best_iteration，使用所有iteration
                proba = model.predict_proba(X)[:, 1]
            
            all_predictions.append(proba)

        all_predictions = np.array(all_predictions)  # (n_models, n_samples)

        # 根据方法聚合预测
        if method == "mean":
            final_predictions = np.mean(all_predictions, axis=0)
        elif method == "median":
            final_predictions = np.median(all_predictions, axis=0)
        elif method == "best":
            # 选择表现最好的模型（这里简化使用第一个）
            final_predictions = all_predictions[0]
        else:
            raise ValueError(f"不支持的集成方法: {method}")

        return final_predictions

    def compute_likelihood_ratios(
        self,
        classifier_scores: np.ndarray,
        mu: float,
        epsilon: float = 1e-10
    ) -> np.ndarray:
        """
        根据论文公式计算likelihood ratios

        ℓ(x) = [z(x) - (1-μ)(1-z(x))] / [μ(1-z(x))]

        其中 z(x) 是XGBoost分类器概率分数，μ 是信号比例

        Args:
            classifier_scores: XGBoost分类器概率分数 z(x)
            mu: 信号比例 μ = N_sig / (N_sig + N_bkg)
            epsilon: 数值稳定性参数

        Returns:
            likelihood_ratios: 似然比数组
        """
        z = classifier_scores

        # 避免除零错误
        denominator = mu * (1 - z + epsilon)

        # 论文公式：ℓ(x) = [z - (1-μ)(1-z)] / [μ(1-z)]
        numerator = z - (1 - mu) * (1 - z)
        likelihood_ratios = numerator / denominator

        # 处理异常值
        likelihood_ratios = np.clip(likelihood_ratios, -1e9, 1e9)

        return likelihood_ratios

    def compute_weights_from_likelihood_ratios(
        self,
        likelihood_ratios: np.ndarray,
        mu: float
    ) -> np.ndarray:
        """
        从likelihood ratios计算权重（用于soft weighting）

        w_i = ℓ(x_i)

        但需要确保权重为正且合理

        Args:
            likelihood_ratios: 似然比数组
            mu: 信号比例

        Returns:
            weights: 权重数组
        """
        # 基本权重就是likelihood ratios
        weights = likelihood_ratios

        # 确保权重为正（论文中提到要去掉负权重）
        weights = np.maximum(weights, 0)

        # 限制权重上限避免数值问题
        weights = np.clip(weights, 0, 1e9)

        return weights

    def plot_bdt_training_history(
        self,
        models: List[Any],
        save_path: Optional[str] = None
    ):
        """
        绘制BDT训练历史

        Args:
            models: 训练好的模型列表
            save_path: 保存路径（可选）
        """
        if not models:
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        for i, model in enumerate(models[:5]):  # 只绘制前5个模型
            if hasattr(model, 'evals_result_') and model.evals_result_():
                results = model.evals_result_()
                if 'validation_0' in results:
                    train_loss = results['validation_0']['logloss']
                    ax.plot(train_loss, label=f'Model {i+1}', alpha=0.7)

        ax.set_xlabel('Boosting Round')
        ax.set_ylabel('Log Loss')
        ax.set_title('BDT Training History')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"BDT训练历史已保存到: {save_path}")

        plt.close()


def run_bdt_bump_hunt(
    flow_samples_sr: np.ndarray,
    data_samples_sr: np.ndarray,
    data_samples_sb: np.ndarray,
    bdt_config: Dict[str, Any],
    num_folds: int = 5,
    visualize: bool = True,
    save_path: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[int, List[Any]]]:
    """
    运行完整的BDT bump hunt流程（参考原始dimuonAD代码）

    Args:
        flow_samples_sr: SR中flow生成的背景样本
        data_samples_sr: SR中的真实数据
        data_samples_sb: SB中的真实数据（用于测试）
        bdt_config: BDT配置
        num_folds: 交叉验证折数
        visualize: 是否可视化
        save_path: 保存路径

    Returns:
        sr_scores: SR数据的BDT分数
        sb_scores: SB数据的BDT分数
        trained_models_by_fold: 按fold组织的训练好的模型列表
    """
    trainer = BDTTrainer(bdt_config)

    # 训练BDT（使用5-fold交叉验证）
    sr_scores, sb_scores, trained_models_by_fold = trainer.train_bdt_on_sr(
        sr_data=data_samples_sr,
        sr_generated_background=flow_samples_sr,
        sb_data=data_samples_sb,
        random_state=42
    )

    if visualize and trained_models_by_fold:
        # 收集所有模型用于可视化
        all_models = []
        for fold_id in sorted(trained_models_by_fold.keys()):
            all_models.extend(trained_models_by_fold[fold_id])
        trainer.plot_bdt_training_history(all_models, save_path)

    return sr_scores, sb_scores, trained_models_by_fold

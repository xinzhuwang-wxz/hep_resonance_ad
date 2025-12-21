"""
XGBoost/BDT 训练模块

实现论文中的Step 3：训练分类器区分SR中的真实数据和生成的背景样本
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

    在SR中训练分类器区分真实数据和flow生成的背景样本
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化BDT训练器

        Args:
            config: BDT配置字典
        """
        self.config = config
        self.logger = get_logger(__name__)

        # 默认BDT超参数（基于原始论文）
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
        val_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[np.ndarray, List[Any]]:
        """
        在SR中训练BDT分类器

        Args:
            sr_data: SR中的真实数据 (n_events, n_features)
            sr_generated_background: SR中flow生成的背景样本 (n_events, n_features)
            val_size: 验证集比例
            random_state: 随机种子

        Returns:
            bdt_scores: BDT概率分数数组 (n_events,)
            trained_models: 训练好的模型列表
        """
        self.logger.info("开始BDT训练...")

        # 准备训练数据
        # 注意：排除质量特征（最后一列），因为BDT不使用质量信息
        X_data = sr_data[:, :-1]      # 真实SR数据（不含质量）
        X_background = sr_generated_background[:, :-1]  # 生成的背景（不含质量）

        # 标签：1 = 真实数据，0 = 生成背景
        y_data = np.ones(X_data.shape[0])
        y_background = np.zeros(X_background.shape[0])

        # 合并训练数据
        X_train = np.vstack([X_data, X_background])
        y_train = np.hstack([y_data, y_background])

        # 计算类别权重（平衡正负样本）
        class_weight = {0: 1.0, 1: len(X_background) / len(X_data)}
        sample_weights = np.array([class_weight[int(y)] for y in y_train])

        # 打乱数据
        X_train, y_train, sample_weights = shuffle(
            X_train, y_train, sample_weights, random_state=random_state
        )

        # 训练BDT ensemble
        n_ensemble = self.hyperparams["n_ensemble"]
        trained_models = []

        self.logger.info(f"训练 {n_ensemble} 个BDT模型...")

        for i in range(n_ensemble):
            # 为每个模型设置不同的随机种子
            model_seed = random_state + i * 100

            # XGBoost参数
            xgb_params = {
                'n_estimators': self.hyperparams["n_estimators"],
                'max_depth': self.hyperparams["max_depth"],
                'learning_rate': self.hyperparams["learning_rate"],
                'subsample': self.hyperparams["subsample"],
                'early_stopping_rounds': self.hyperparams["early_stopping_rounds"],
                'objective': 'binary:logistic',
                'random_state': model_seed,
                'eval_metric': 'logloss',
                'verbosity': 0,  # 静默模式
            }

            # 创建和训练模型
            model = xgb.XGBClassifier(**xgb_params)

            # 训练（使用验证集进行early stopping）
            eval_set = [(X_train, y_train)]
            if val_size > 0:
                # 简单分割验证集
                n_val = int(len(X_train) * val_size)
                X_train_split = X_train[:-n_val]
                y_train_split = y_train[:-n_val]
                X_val_split = X_train[-n_val:]
                y_val_split = y_train[-n_val:]
                w_train_split = sample_weights[:-n_val]
                w_val_split = sample_weights[-n_val:]

                eval_set = [(X_train_split, y_train_split), (X_val_split, y_val_split)]

                model.fit(
                    X_train_split, y_train_split,
                    sample_weight=w_train_split,
                    eval_set=eval_set,
                    sample_weight_eval_set=[w_train_split, w_val_split],
                    verbose=False
                )
            else:
                model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)

            trained_models.append(model)

            if (i + 1) % 5 == 0:
                self.logger.info(f"已训练 {i + 1}/{n_ensemble} 个模型")

        self.logger.info(f"BDT训练完成，共训练 {len(trained_models)} 个模型")

        # 使用训练好的模型对所有SR数据进行打分
        self.logger.info("计算BDT分数...")

        # 对SR中的所有事件进行打分（包括真实数据和生成的背景）
        X_sr_all = np.vstack([sr_data[:, :-1], sr_generated_background[:, :-1]])

        bdt_scores = self.predict_proba_ensemble(X_sr_all, trained_models)

        self.logger.info(f"BDT打分完成，共处理 {len(bdt_scores)} 个事件")

        return bdt_scores, trained_models

    def predict_proba_ensemble(
        self,
        X: np.ndarray,
        models: List[Any],
        method: str = "mean"
    ) -> np.ndarray:
        """
        使用ensemble模型进行概率预测

        Args:
            X: 输入特征 (n_samples, n_features)
            models: 训练好的模型列表
            method: 集成方法 ("mean", "median", 或 "best")

        Returns:
            预测概率数组 (n_samples,) - 返回正类概率
        """
        if not models:
            raise ValueError("没有训练好的模型")

        # 收集所有模型的预测
        all_predictions = []

        for model in models:
            # predict_proba 返回 [负类概率, 正类概率]，我们需要正类概率
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
        bdt_scores: np.ndarray,
        mu: float,
        epsilon: float = 1e-10
    ) -> np.ndarray:
        """
        根据论文公式计算likelihood ratios

        ℓ(x) = [z(x) - (1-μ)(1-z(x))] / [μ(1-z(x))]

        其中 z(x) 是BDT概率分数，μ 是信号比例

        Args:
            bdt_scores: BDT概率分数 z(x)
            mu: 信号比例 μ = N_sig / (N_sig + N_bkg)
            epsilon: 数值稳定性参数

        Returns:
            likelihood_ratios: 似然比数组
        """
        z = bdt_scores

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
) -> Tuple[np.ndarray, List[Any]]:
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
        bdt_scores: BDT分数
        trained_models: 训练好的模型列表
    """
    trainer = BDTTrainer(bdt_config)

    # 训练BDT
    bdt_scores, trained_models = trainer.train_bdt_on_sr(
        sr_data=data_samples_sr,
        sr_generated_background=flow_samples_sr,
        random_state=42
    )

    if visualize and trained_models:
        trainer.plot_bdt_training_history(trained_models, save_path)

    return bdt_scores, trained_models

"""
CATHODE 模型实现

基于 normalizing flow 的异常检测模型。
"""

import torch
import torch.nn as nn
import yaml
from pathlib import Path
from typing import Optional

from resonance_ad.core.logging import get_logger
from resonance_ad.models.flows import FlowSequential, MADE, BatchNormFlow, Reverse

logger = get_logger(__name__)


class DensityEstimator:
    """
    密度估计器基类
    
    从 YAML 配置文件构建 normalizing flow 模型。
    """
    
    def __init__(
        self,
        config_path: str | Path,
        num_inputs: int,
        eval_mode: bool = False,
        load_path: Optional[str | Path] = None,
        device: torch.device = torch.device("cpu"),
        verbose: bool = False,
        bound: bool = False,
    ):
        """
        初始化密度估计器
        
        Args:
            config_path: 配置文件路径
            num_inputs: 输入特征数（不包括条件输入）
            eval_mode: 是否为评估模式
            load_path: 模型权重路径
            device: 设备
            verbose: 是否打印详细信息
            bound: 是否使用有界分布（Uniform base）
        """
        self.config_path = Path(config_path)
        self.num_inputs = num_inputs
        self.device = device
        self.verbose = verbose
        self.bound = bound
        
        # 加载配置
        with open(self.config_path, 'r') as f:
            self.params = yaml.safe_load(f)
        
        # 构建模型
        self.build(self.params, eval_mode, load_path)
    
    def build(self, params, eval_mode, load_path):
        """构建 flow 模型"""
        modules = []
        
        for layer in range(params['num_layers']):
            for i in range(params['num_blocks']):
                self.build_block(i, modules, params)
        
        # 创建 FlowSequential
        self.model = FlowSequential(*modules)
        
        # 初始化权重
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias.data.fill_(0)
        
        self.model.num_inputs = self.num_inputs
        self.model.to(self.device)
        
        if self.verbose:
            logger.info(f"Model:\n{self.model}")
            total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.info(f"DensityEstimator has {total_params} parameters")
        
        # 加载权重
        if load_path is not None:
            self.load_model(load_path)
        
        # 构建优化器
        self.build_optimizer(params)
        
        if eval_mode:
            self.model.eval()
    
    def build_block(self, i, modules, params):
        """构建一个 flow block"""
        # MADE layer
        modules.append(
            MADE(
                self.num_inputs,
                params['num_hidden'],
                params['num_cond_inputs'],
                act=params['activation_function'],
                pre_exp_tanh=params.get('pre_exp_tanh', False),
            )
        )
        
        # Batch normalization
        if params.get('batch_norm', True):
            modules.append(
                BatchNormFlow(
                    self.num_inputs,
                    momentum=params.get('batch_norm_momentum', 1.0),
                )
            )
        
        # Reverse permutation
        modules.append(Reverse(self.num_inputs))
    
    def load_model(self, load_path):
        """加载模型权重"""
        load_path = Path(load_path)
        if load_path.exists():
            logger.info(f"Loading model parameters from {load_path}")
            self.model.load_state_dict(torch.load(load_path, map_location=self.device))
        else:
            logger.warning(f"Model file not found: {load_path}")
    
    def build_optimizer(self, params):
        """构建优化器"""
        optimizer_name = params['optimizer']['name']
        optimizer_kwargs = {k: v for k, v in params['optimizer'].items() if k != 'name'}
        
        optimizer_class = getattr(torch.optim, optimizer_name)
        self.optimizer = optimizer_class(self.model.parameters(), **optimizer_kwargs)
    
    def log_probs(self, inputs, cond_inputs=None):
        """计算 log probability"""
        return self.model.log_probs(inputs, cond_inputs)
    
    def sample(self, num_samples=None, noise=None, cond_inputs=None):
        """从模型中采样"""
        return self.model.sample(num_samples, noise, cond_inputs)


# 别名
CATHODE = DensityEstimator


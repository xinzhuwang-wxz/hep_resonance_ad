"""
CATHODE 训练工具

提供训练循环和损失计算函数。
"""

import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from typing import Optional

from resonance_ad.core.logging import get_logger

logger = get_logger(__name__)


def compute_loss_over_batches(model, dataloader, device, correct_logit=None):
    """
    计算整个数据集的平均损失
    
    与原始代码对齐：处理NaN和异常值，使用累积平均方式
    
    Args:
        model: CATHODE 模型
        dataloader: 数据加载器
        device: 设备
        correct_logit: 如果使用 no_logit，需要提供 data_std 来校正损失
        
    Returns:
        (corrected_loss, uncorrected_loss) 或 (loss, None)
    """
    model.eval()
    with torch.no_grad():
        if correct_logit is not None:
            corrected_now_loss = 0.
        now_loss = 0
        n_nans = 0
        n_highs = 0
        
        for batch_idx, batch_data in enumerate(dataloader):
            # DataLoader 返回 tuple，需要解包
            if isinstance(batch_data, (tuple, list)):
                batch_data = batch_data[0]
            batch_data = batch_data.to(device).float()
            data = batch_data[:, :-1]
            cond_data = torch.reshape(batch_data[:, -1], (-1, 1))
            
            loss_vals_raw = model.log_probs(data, cond_data)
            
            # 检查NaN（原始代码的行为）
            if torch.isnan(loss_vals_raw).any():
                logger.warning(f"Found NaN in loss_vals_raw at batch {batch_idx}")
                logger.warning(f"Problematic data: {batch_data[torch.isnan(loss_vals_raw).reshape(-1,)]}")
            
            loss_vals = loss_vals_raw.flatten()
            
            if correct_logit is not None:
                mask = (data > 0) & (data < 1)
                data_masked = data[mask.all(dim=1)]
                loss_vals_raw_masked = loss_vals_raw[mask.all(dim=1)]
                corrected_loss_vals_raw = loss_vals_raw_masked.flatten() + \
                    torch.log(correct_logit * data_masked * (1. - data_masked)).sum(dim=1)
                corrected_loss_vals = corrected_loss_vals_raw.flatten()
            
            # 统计并过滤NaN和异常值（与原始代码对齐）
            n_nans += torch.isnan(loss_vals).sum().item()
            n_highs += (torch.abs(loss_vals) >= 1000).sum().item()
            loss_vals = loss_vals[~torch.isnan(loss_vals)]
            loss_vals = loss_vals[torch.abs(loss_vals) < 1000]
            
            loss = -loss_vals.mean()
            loss = loss.item()
            
            if correct_logit is not None:
                corrected_now_loss -= corrected_loss_vals.mean().item()
                corrected_end_loss = corrected_now_loss / (batch_idx + 1)
            
            now_loss += loss
            end_loss = now_loss / (batch_idx + 1)
        
        if n_nans > 0 or n_highs > 0:
            logger.warning(f"Loss statistics: n_nans = {n_nans}, n_highs = {n_highs}")
        
        if correct_logit is not None:
            return (corrected_end_loss, end_loss)
        else:
            return (end_loss,)


def train_epoch(model, optimizer, dataloader, device, verbose=True, data_std=None):
    """
    训练一个 epoch
    
    与原始代码对齐：包括BatchNormFlow的momentum处理和corrected_loss计算
    
    Args:
        model: CATHODE 模型
        optimizer: 优化器
        dataloader: 训练数据加载器
        device: 设备
        verbose: 是否显示进度条
        data_std: 用于 no_logit 校正
        
    Returns:
        (corrected_loss, uncorrected_loss) 或 (loss, None)
    """
    model.train()
    train_loss = 0
    train_loss_avg = []
    if data_std is not None:
        corrected_train_loss_avg = []
    
    if verbose:
        pbar = tqdm(total=len(dataloader.dataset), desc="Training")
    
    for batch_idx, batch_data in enumerate(dataloader):
        # DataLoader 返回 tuple，需要解包
        if isinstance(batch_data, (tuple, list)):
            batch_data = batch_data[0]
        batch_data = batch_data.to(device).float()
        data = batch_data[:, :-1]
        cond_data = torch.reshape(batch_data[:, -1], (-1, 1))
        
        optimizer.zero_grad()
        loss = -model.log_probs(data, cond_data)
        train_loss += loss.mean().item()
        train_loss_avg.extend(loss.tolist())
        
        if data_std is not None:
            # 校正损失（与原始代码对齐）
            mask = (data > 0) & (data < 1)
            data_masked = data[mask.all(dim=1)]
            loss_masked = loss[mask.all(dim=1)]
            corrected_loss = (loss_masked.flatten() - 
                            torch.log(data_std * data_masked * (1. - data_masked)).sum(dim=1)).flatten()
            corrected_train_loss_avg.extend(corrected_loss.tolist())
        
        loss.mean().backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        if verbose:
            pbar.update(data.size(0))
            pbar.set_postfix({'loss': f'{-train_loss / (batch_idx + 1):.6f}'})
    
    if verbose:
        pbar.close()
    
    # BatchNormFlow的momentum处理（原始代码的关键部分）
    from resonance_ad.models.flows import BatchNormFlow
    has_batch_norm = False
    for module in model.modules():
        if isinstance(module, BatchNormFlow):
            has_batch_norm = True
            module.momentum = 0
    
    if has_batch_norm:
        # 对整个数据集做一次forward pass（原始代码的注释说这对BN很重要）
        with torch.no_grad():
            # 获取整个数据集
            if hasattr(dataloader.dataset, 'tensors'):
                # TensorDataset
                loc_data = dataloader.dataset.tensors[0].to(device).float()
            elif isinstance(dataloader.dataset, np.ndarray):
                # 如果直接传递numpy数组（原始代码的方式）
                loc_data = torch.tensor(dataloader.dataset, device=device).float()
            else:
                # 其他情况，尝试直接使用
                loc_data = torch.tensor(dataloader.dataset, device=device).float()
            model(loc_data[:, :-1], torch.reshape(loc_data[:, -1], (-1, 1)))
        
        # 恢复momentum
        for module in model.modules():
            if isinstance(module, BatchNormFlow):
                module.momentum = 1
    
    # 计算平均损失（与原始代码对齐）
    if data_std is not None:
        return (np.array(corrected_train_loss_avg).flatten().mean(),
                np.array(train_loss_avg).flatten().mean())
    else:
        return (np.array(train_loss_avg).flatten().mean(),)


def train_cathode(
    model,
    optimizer,
    train_loader,
    val_loader,
    epochs: int,
    savedir: Path,
    device: torch.device,
    verbose: bool = True,
    no_logit: bool = False,
    data_std: Optional[float] = None,
):
    """
    训练 CATHODE 模型
    
    Args:
        model: CATHODE 模型
        optimizer: 优化器
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        epochs: 训练轮数
        savedir: 保存目录
        device: 设备
        verbose: 是否显示详细信息
        no_logit: 是否不使用 logit transform
        data_std: 用于 no_logit 校正
    """
    savedir = Path(savedir)
    savedir.mkdir(parents=True, exist_ok=True)
    
    # 记录初始损失
    train_loss_return = compute_loss_over_batches(
        model, train_loader, device, correct_logit=data_std if no_logit else None
    )
    val_loss_return = compute_loss_over_batches(
        model, val_loader, device, correct_logit=data_std if no_logit else None
    )
    
    train_loss = train_loss_return[0]
    val_loss = val_loss_return[0]
    
    if no_logit:
        logger.info(f"Uncorrected train_loss = {train_loss_return[1]}")
        logger.info(f"Uncorrected val_loss = {val_loss_return[1]}")
    
    logger.info(f"Initial train_loss = {train_loss:.6f}")
    logger.info(f"Initial val_loss = {val_loss:.6f}")
    
    # 初始化损失数组
    train_losses = np.full(epochs + 1, 1e20, dtype=np.float32)
    val_losses = np.full(epochs + 1, 1e20, dtype=np.float32)
    
    train_losses[0] = train_loss
    val_losses[0] = val_loss
    
    np.save(savedir / "train_losses.npy", train_losses)
    np.save(savedir / "val_losses.npy", val_losses)
    
    best_val_loss = val_loss
    best_epoch = 0
    
    # 训练循环
    for epoch in range(epochs):
        logger.info(f"\nEpoch {epoch + 1}/{epochs}")
        
        # 训练一个 epoch
        train_loss_return = train_epoch(
            model, optimizer, train_loader, device, verbose=verbose, data_std=data_std if no_logit else None
        )
        
        # 验证
        val_loss_return = compute_loss_over_batches(
            model, val_loader, device, correct_logit=data_std if no_logit else None
        )
        
        train_loss = train_loss_return[0]
        val_loss = val_loss_return[0]
        
        if no_logit:
            logger.info(f"Uncorrected train_loss = {train_loss_return[1]}")
            logger.info(f"Uncorrected val_loss = {val_loss_return[1]}")
        
        logger.info(f"Train loss = {train_loss:.6f}")
        logger.info(f"Val loss = {val_loss:.6f}")
        
        train_losses[epoch + 1] = train_loss
        val_losses[epoch + 1] = val_loss
        
        # 保存损失
        np.save(savedir / "train_losses.npy", train_losses)
        np.save(savedir / "val_losses.npy", val_losses)
        
        # 保存模型
        torch.save(model.state_dict(), savedir / f"model_epoch_{epoch}.pt")
        
        # 更新最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), savedir / "best_model.pt")
    
    logger.info(f"\nTraining completed. Best epoch: {best_epoch} (val_loss = {best_val_loss:.6f})")


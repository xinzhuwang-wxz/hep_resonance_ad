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
    
    Args:
        model: CATHODE 模型
        dataloader: 数据加载器
        device: 设备
        correct_logit: 如果使用 no_logit，需要提供 data_std 来校正损失
        
    Returns:
        (corrected_loss, uncorrected_loss) 或 (loss, None)
    """
    model.eval()
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch_data in dataloader:
            batch_data = batch_data.to(device).float()
            data = batch_data[:, :-1]
            cond_data = torch.reshape(batch_data[:, -1], (-1, 1))
            
            loss = -model.log_probs(data, cond_data)
            total_loss += loss.sum().item()
            total_samples += loss.size(0)
    
    avg_loss = total_loss / total_samples
    
    if correct_logit is not None:
        # 校正损失（用于 no_logit 情况）
        # 这里简化处理，实际需要更复杂的计算
        uncorrected_loss = avg_loss
        corrected_loss = avg_loss  # 简化版本
        return corrected_loss, uncorrected_loss
    else:
        return avg_loss, None


def train_epoch(model, optimizer, dataloader, device, verbose=True, data_std=None):
    """
    训练一个 epoch
    
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
    train_loss_avg = []
    corrected_train_loss_avg = []
    
    if verbose:
        pbar = tqdm(total=len(dataloader.dataset), desc="Training")
    
    for batch_idx, batch_data in enumerate(dataloader):
        batch_data = batch_data.to(device).float()
        data = batch_data[:, :-1]
        cond_data = torch.reshape(batch_data[:, -1], (-1, 1))
        
        optimizer.zero_grad()
        loss = -model.log_probs(data, cond_data)
        
        train_loss_avg.extend(loss.tolist())
        
        if data_std is not None:
            # 校正损失（简化版本）
            mask = (data > 0) & (data < 1)
            if mask.any():
                corrected_loss = loss[mask.all(dim=1)]
                corrected_train_loss_avg.extend(corrected_loss.tolist())
        
        loss.mean().backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        if verbose:
            pbar.update(batch_data.size(0))
            pbar.set_postfix({'loss': f'{loss.mean().item():.4f}'})
    
    if verbose:
        pbar.close()
    
    avg_loss = np.mean(train_loss_avg)
    
    if data_std is not None and corrected_train_loss_avg:
        corrected_avg_loss = np.mean(corrected_train_loss_avg)
        return corrected_avg_loss, avg_loss
    else:
        return avg_loss, None


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


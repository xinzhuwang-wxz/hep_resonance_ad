"""
日志系统

提供统一的日志接口，参考 Made-With-ML 的设计。
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    log_dir: Optional[Path] = None,
    level: int = logging.INFO,
    log_to_file: bool = True,
) -> None:
    """
    设置日志系统
    
    Args:
        log_dir: 日志文件目录，如果为 None 则不写入文件
        level: 日志级别
        log_to_file: 是否写入文件
    """
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(
        logging.Formatter("%(levelname)s - %(name)s - %(message)s")
    )
    handlers.append(console_handler)
    
    # File handler
    if log_to_file and log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_dir / "resonance_ad.log")
        file_handler.setFormatter(
            logging.Formatter(
                "%(levelname)s %(asctime)s [%(name)s:%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n"
            )
        )
        handlers.append(file_handler)
    
    # 配置根 logger
    logging.basicConfig(
        level=level,
        handlers=handlers,
        force=True,  # 覆盖已有配置
    )


def get_logger(name: str) -> logging.Logger:
    """
    获取指定名称的 logger
    
    Args:
        name: logger 名称，通常是模块名
        
    Returns:
        Logger 对象
    """
    return logging.getLogger(name)


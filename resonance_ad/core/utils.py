"""
Utility functions for device management and other common operations.
"""

import torch
import logging

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """
    Get the best available device for PyTorch computations.

    Priority order:
    1. CUDA (NVIDIA GPU)
    2. MPS (Apple Silicon GPU)
    3. CPU

    Returns:
        torch.device: The best available device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS device (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")

    return device


def get_device_info() -> str:
    """
    Get detailed information about available devices.

    Returns:
        str: Device information summary
    """
    info_lines = ["Device Information:"]

    # CUDA info
    if torch.cuda.is_available():
        info_lines.append(f"  CUDA: Available ({torch.cuda.device_count()} device(s))")
        for i in range(torch.cuda.device_count()):
            info_lines.append(f"    Device {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            info_lines.append(f"      Memory: {props.total_memory / 1024**3:.1f} GB")
    else:
        info_lines.append("  CUDA: Not available")

    # MPS info
    if torch.backends.mps.is_available():
        info_lines.append("  MPS: Available (Apple Silicon)")
    else:
        info_lines.append("  MPS: Not available")

    info_lines.append("  CPU: Available")

    return "\n".join(info_lines)

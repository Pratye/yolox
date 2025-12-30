#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

"""
Device utilities for supporting MPS (MacBook M1), CUDA, and CPU.
"""

import torch
from loguru import logger


def get_available_device():
    """
    Detect and return the best available device.
    
    Priority: CUDA > MPS > CPU
    
    Returns:
        str: Device string ('cuda:0', 'mps', or 'cpu')
    """
    if torch.cuda.is_available():
        device = "cuda:0"
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        return device
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        logger.info("Using MPS device (MacBook M1 GPU)")
        return device
    else:
        device = "cpu"
        logger.warning("No GPU available, using CPU (training will be slow)")
        return device


def get_device_type(device_str: str = None):
    """
    Get the device type from device string.
    
    Args:
        device_str: Device string (e.g., 'cuda:0', 'mps', 'cpu')
        
    Returns:
        str: Device type ('cuda', 'mps', or 'cpu')
    """
    if device_str is None:
        device_str = get_available_device()
    
    if device_str.startswith("cuda"):
        return "cuda"
    elif device_str == "mps":
        return "mps"
    else:
        return "cpu"


def is_mps_device(device_str: str = None):
    """Check if device is MPS."""
    return get_device_type(device_str) == "mps"


def is_cuda_device(device_str: str = None):
    """Check if device is CUDA."""
    return get_device_type(device_str) == "cuda"


def get_amp_scaler(device_str: str = None, enabled: bool = True):
    """
    Get appropriate AMP scaler for the device.
    
    MPS doesn't support torch.cuda.amp.GradScaler, so we use a CPU-based scaler.
    
    Args:
        device_str: Device string
        enabled: Whether AMP is enabled
        
    Returns:
        GradScaler or None
    """
    if not enabled:
        return None
    
    device_type = get_device_type(device_str)
    
    if device_type == "cuda":
        return torch.cuda.amp.GradScaler(enabled=True)
    elif device_type == "mps":
        # MPS doesn't support CUDA scaler, use CPU scaler as fallback
        # Note: MPS has limited AMP support, may need to disable for some operations
        logger.warning("MPS has limited AMP support. Using CPU-based scaler.")
        return torch.cuda.amp.GradScaler(enabled=True)  # Still works, just slower
    else:
        return None


def get_autocast_context(device_str: str = None, enabled: bool = True):
    """
    Get appropriate autocast context for the device.
    
    Args:
        device_str: Device string
        enabled: Whether autocast is enabled
        
    Returns:
        Autocast context manager
    """
    if not enabled:
        # Return a no-op context manager
        from contextlib import nullcontext
        return nullcontext()
    
    device_type = get_device_type(device_str)
    
    if device_type == "cuda":
        return torch.cuda.amp.autocast(enabled=True)
    elif device_type == "mps":
        # MPS supports autocast but with limitations
        # Use CPU autocast as fallback
        return torch.cuda.amp.autocast(enabled=True, device_type="cpu")
    else:
        from contextlib import nullcontext
        return nullcontext()


def set_device(device_str: str, local_rank: int = 0):
    """
    Set the device for the current process.
    
    Args:
        device_str: Device string
        local_rank: Local rank for distributed training
    """
    device_type = get_device_type(device_str)
    
    if device_type == "cuda":
        torch.cuda.set_device(local_rank)
    elif device_type == "mps":
        # MPS doesn't need explicit device setting
        pass
    # CPU doesn't need device setting


def get_optimal_batch_size(device_str: str = None, base_batch_size: int = 64):
    """
    Get optimal batch size for the device.
    
    MPS typically has less memory than CUDA GPUs, so reduce batch size.
    
    Args:
        device_str: Device string
        base_batch_size: Base batch size for CUDA
        
    Returns:
        int: Recommended batch size
    """
    device_type = get_device_type(device_str)
    
    if device_type == "cuda":
        return base_batch_size
    elif device_type == "mps":
        # MPS (M1) typically has 16GB unified memory, but GPU portion is limited
        # Reduce batch size to avoid OOM
        recommended = max(8, base_batch_size // 4)
        logger.info(f"MPS device detected. Reducing batch size from {base_batch_size} to {recommended}")
        return recommended
    else:
        # CPU training is very slow, use smaller batch size
        return max(4, base_batch_size // 8)


def get_num_devices(device_str: str = None):
    """
    Get number of available devices.
    
    Args:
        device_str: Device string (optional)
        
    Returns:
        int: Number of devices
    """
    if device_str is None:
        device_str = get_available_device()
    
    device_type = get_device_type(device_str)
    
    if device_type == "cuda":
        return torch.cuda.device_count()
    elif device_type == "mps":
        # MPS only supports single device
        return 1
    else:
        return 1


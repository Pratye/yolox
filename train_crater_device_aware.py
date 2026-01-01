#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Device-aware training script for YOLOX crater detection.

Automatically detects device (CUDA/MPS/CPU) and optimizes settings.

Usage:
    python train_crater_device_aware.py -c weights/yolox_m.pth
"""

import sys
import os

# Add YOLOX to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from loguru import logger

# Import device utilities
try:
    from yolox.utils.device import (
        get_available_device,
        get_device_type,
        get_optimal_batch_size,
        is_mps_device,
        is_cuda_device,
    )
except ImportError:
    # Fallback if device utils not available
    def get_available_device():
        if torch.cuda.is_available():
            return "cuda:0"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    def get_device_type(device_str):
        if device_str.startswith("cuda"):
            return "cuda"
        elif device_str == "mps":
            return "mps"
        return "cpu"
    
    def get_optimal_batch_size(device_str, base_batch_size=64):
        device_type = get_device_type(device_str)
        if device_type == "mps":
            return max(8, base_batch_size // 4)
        elif device_type == "cpu":
            return max(4, base_batch_size // 8)
        return base_batch_size
    
    def is_mps_device(device_str):
        return get_device_type(device_str) == "mps"
    
    def is_cuda_device(device_str):
        return get_device_type(device_str) == "cuda"


def main():
    """Main entry point with device detection and optimization."""
    from tools.train import make_parser, main as train_main
    from yolox.core import launch
    from yolox.exp import get_exp, check_exp_value
    from yolox.utils import configure_module, get_num_devices
    
    # Detect device
    device_str = get_available_device()
    device_type = get_device_type(device_str)
    
    logger.info(f"Detected device: {device_str} ({device_type.upper()})")
    
    # Parse arguments
    configure_module()
    parser = make_parser()
    args = parser.parse_args()
    
    # Set default experiment file
    if args.exp_file is None:
        args.exp_file = "exps/example/custom/crater_yolox_s.py"
    
    # Auto-optimize batch size if not specified
    if args.batch_size is None or args.batch_size == 64:  # Default
        optimal_batch = get_optimal_batch_size(device_str, 64)
        if args.batch_size != optimal_batch:
            logger.info(f"Auto-optimizing batch size: {args.batch_size} -> {optimal_batch} for {device_type.upper()}")
            args.batch_size = optimal_batch
    
    # Handle device count
    if args.devices is None:
        if device_type == "cuda":
            args.devices = get_num_devices()
        else:
            args.devices = 1  # MPS and CPU only support single device

    # Adjust batch size for MPS - use conservative size with periodic memory clearing
    if device_type == "mps" and args.batch_size > 2:
        original_batch = args.batch_size
        args.batch_size = 2  # Conservative batch size for memory stability
        logger.warning(f"Reducing batch size from {original_batch} to {args.batch_size} for MPS memory stability")

    # Force disable caching for MPS to prevent memory accumulation
    if device_type == "mps" and args.cache is not None:
        logger.warning("Disabling caching for MPS - using aggressive memory clearing instead")
        args.cache = None

    # Warn about MPS limitations
    if is_mps_device(device_str):
        logger.warning("=" * 60)
        logger.warning("MPS (MacBook M1 GPU) detected")
        logger.warning("Note: MPS has limited support in PyTorch")
        logger.warning("- Distributed training is not supported")
        logger.warning("- Mixed precision (FP16) may be slower")
        logger.warning("- Some operations may fall back to CPU")
        logger.warning("- Image caching available (use --cache for faster training)")
        logger.warning(f"- Recommended batch size: {get_optimal_batch_size(device_type)} (currently {args.batch_size})")
        logger.warning("=" * 60)
        
        # Disable distributed training for MPS
        if args.devices > 1:
            logger.warning("MPS does not support multi-device. Setting devices=1")
            args.devices = 1
    
    # Get experiment
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)
    check_exp_value(exp)
    
    if not args.experiment_name:
        args.experiment_name = exp.exp_name
    
    # Caching is now handled in MPS patch with memory management
    if args.cache is not None:
        logger.info(f"Enabling dataset caching (type: {args.cache})")
    
    # For MPS/CPU, we need to modify the trainer to use the correct device
    if is_mps_device(device_str):
        logger.info("Applying MPS compatibility patches...")
        # We'll patch the trainer after it's created
    
    # Launch training
    if device_type == "cuda" and args.devices > 1:
        # Distributed training for CUDA
        dist_url = "auto" if args.dist_url is None else args.dist_url
        launch(
            train_main,
            args.devices,
            args.num_machines,
            args.machine_rank,
            backend=args.dist_backend,
            dist_url=dist_url,
            args=(exp, args),
        )
    else:
        # Single device training
        # Patch trainer for MPS if needed
        if is_mps_device(device_str):
            # Import and apply MPS patches
            from yolox.utils.mps_patch import patch_trainer_for_mps
            
            # We need to patch after trainer is created, so modify train_main
            original_train_main = train_main
            
            def patched_train_main(exp, args):
                # Skip CUDA-specific setup
                import torch
                import torch.backends.cudnn as cudnn
                
                if exp.seed is not None:
                    import random
                    random.seed(exp.seed)
                    torch.manual_seed(exp.seed)
                    # Skip cudnn.deterministic for MPS
                
                # Skip CUDA-specific environment setup
                # configure_nccl() and configure_omp() are CUDA-specific
                
                trainer = exp.get_trainer(args)
                # Set device to MPS
                trainer.device = "mps"
                # Patch before training starts
                patch_trainer_for_mps(trainer)
                trainer.train()
            
            train_main = patched_train_main
        
        train_main(exp, args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
MPS (MacBook M1) compatibility patches for YOLOX trainer.

This module patches the trainer to work with MPS devices.
"""

import torch
from loguru import logger


def patch_trainer_for_mps(trainer):
    """
    Patch trainer instance to work with MPS device.
    
    This patches methods that use CUDA-specific functions.
    """
    device_str = trainer.device
    
    # Check if we're using MPS
    if device_str == "mps" or (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and "mps" in str(device_str)):
        logger.info("Patching trainer for MPS device...")
        
        # Fix scaler - MPS doesn't support CUDA scaler, use CPU-based or disable
        if trainer.amp_training:
            logger.warning("MPS has limited AMP support. Disabling AMP scaler.")
            trainer.amp_training = False
            trainer.scaler = None
        
        # Don't patch get_data_loader on exp object to avoid __repr__ recursion
        # Instead, we'll handle pin_memory in before_train by recreating the dataloader if needed
        
        # Patch before_train to skip CUDA device setting
        original_before_train = trainer.before_train
        
        def mps_before_train(self):
            logger.info("args: {}".format(self.args))
            # Skip printing exp value to avoid recursion issues with patched methods
            logger.info("Starting training with MPS device...")
            
            # Skip CUDA device setting for MPS
            # torch.cuda.set_device() doesn't exist for MPS
            logger.info("Skipping CUDA device setup (using MPS)")
            
            model = self.exp.get_model()
            from yolox.utils import get_model_info
            logger.info(
                "Model Summary: {}".format(get_model_info(model, self.exp.test_size))
            )
            model.to(self.device)
            
            # Rest of original before_train
            self.optimizer = self.exp.get_optimizer(self.args.batch_size)
            model = self.resume_train(model)
            self.no_aug = self.start_epoch >= self.max_epoch - self.exp.no_aug_epochs
            
            # Get data loader - temporarily patch get_data_loader to disable pin_memory
            # We need to do this inline to avoid __repr__ recursion issues
            original_get_data_loader = self.exp.get_data_loader
            
            def mps_get_data_loader_wrapper(batch_size, is_distributed, no_aug=False, cache_img=None):
                """Wrapper that calls original but with pin_memory=False."""
                from yolox.data import (
                    TrainTransform,
                    YoloBatchSampler,
                    DataLoader,
                    InfiniteSampler,
                    MosaicDetection,
                    worker_init_reset_seed,
                )
                from yolox.utils import wait_for_the_master
                import torch.distributed as dist
                
                exp = self.exp
                if exp.dataset is None:
                    with wait_for_the_master():
                        assert cache_img is None, \
                            "cache_img must be None if you didn't create exp.dataset before launch"
                        exp.dataset = exp.get_dataset(cache=False, cache_type=cache_img)
                
                exp.dataset = MosaicDetection(
                    dataset=exp.dataset,
                    mosaic=not no_aug,
                    img_size=exp.input_size,
                    preproc=TrainTransform(
                        max_labels=120,
                        flip_prob=exp.flip_prob,
                        hsv_prob=exp.hsv_prob),
                    degrees=exp.degrees,
                    translate=exp.translate,
                    mosaic_scale=exp.mosaic_scale,
                    mixup_scale=exp.mixup_scale,
                    shear=exp.shear,
                    enable_mixup=exp.enable_mixup,
                    mosaic_prob=exp.mosaic_prob,
                    mixup_prob=exp.mixup_prob,
                )
                
                if is_distributed:
                    batch_size = batch_size // dist.get_world_size()
                
                sampler = InfiniteSampler(len(exp.dataset), seed=exp.seed if exp.seed else 0)
                batch_sampler = YoloBatchSampler(
                    sampler=sampler,
                    batch_size=batch_size,
                    drop_last=False,
                    mosaic=not no_aug,
                )
                
                # Disable pin_memory for MPS
                dataloader_kwargs = {"num_workers": exp.data_num_workers, "pin_memory": False}
                dataloader_kwargs["batch_sampler"] = batch_sampler
                dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed
                
                train_loader = DataLoader(exp.dataset, **dataloader_kwargs)
                return train_loader
            
            # Temporarily replace method
            self.exp.get_data_loader = mps_get_data_loader_wrapper
            try:
                self.train_loader = self.exp.get_data_loader(
                    batch_size=self.args.batch_size,
                    is_distributed=self.is_distributed,
                    no_aug=self.no_aug,
                    cache_img=self.args.cache,
                )
            finally:
                # Restore original method immediately to avoid __repr__ issues
                self.exp.get_data_loader = original_get_data_loader
            logger.info("init prefetcher, this might take one minute or less...")
            # Use MPS-compatible prefetcher
            from yolox.data.data_prefetcher_mps import DataPrefetcherMPS
            self.prefetcher = DataPrefetcherMPS(self.train_loader, device=self.device)
            self.max_iter = len(self.train_loader)
            
            self.lr_scheduler = self.exp.get_lr_scheduler(
                self.exp.basic_lr_per_img * self.args.batch_size, self.max_iter
            )
            # Skip occupy_mem for MPS (CUDA-specific)
            
            # Skip DDP for MPS (not supported)
            if self.is_distributed:
                logger.warning("MPS does not support distributed training. Disabling DDP.")
                self.is_distributed = False
            
            if self.use_model_ema:
                from yolox.utils import ModelEMA
                self.ema_model = ModelEMA(model, 0.9998)
                self.ema_model.updates = self.max_iter * self.start_epoch
            
            self.model = model
            
            self.evaluator = self.exp.get_evaluator(
                batch_size=self.args.batch_size, is_distributed=self.is_distributed
            )
            
            # Tensorboard and Wandb loggers
            import os
            if self.rank == 0:
                from yolox.utils import WandbLogger
                if self.args.logger == "wandb":
                    self.wandb_logger = WandbLogger.initialize_wandb_logger(
                        self.args, self.exp, self.eval_interval
                    )
                    self.wandb_logger.log_stuff_dict = {"lr": self.lr_scheduler.lr}
                else:
                    self.wandb_logger = None
                
                from torch.utils.tensorboard import SummaryWriter
                os.makedirs(self.file_name, exist_ok=True)
                self.tblogger = SummaryWriter(self.file_name)
            else:
                self.tblogger = None
                self.wandb_logger = None
        
        trainer.before_train = mps_before_train.__get__(trainer, type(trainer))
        
        # Patch train_one_iter to use device-aware autocast
        original_train_one_iter = trainer.train_one_iter
        
        def mps_train_one_iter(self):
            import time
            iter_start_time = time.time()
            
            inps, targets = self.prefetcher.next()
            inps = inps.to(self.data_type)
            targets = targets.to(self.data_type)
            targets.requires_grad = False
            inps, targets = self.exp.preprocess(inps, targets, self.input_size)
            data_end_time = time.time()
            
            # Use device-aware autocast
            # MPS has limited AMP support, but we can try
            if self.amp_training:
                # For MPS, we'll use CPU autocast as fallback or disable
                # MPS doesn't fully support torch.cuda.amp.autocast
                try:
                    # Try using autocast with device_type='cpu' as fallback
                    with torch.cuda.amp.autocast(enabled=False):  # Disable for MPS
                        outputs = self.model(inps, targets)
                except:
                    # If that fails, just run without autocast
                    outputs = self.model(inps, targets)
            else:
                outputs = self.model(inps, targets)
            
            loss = outputs["total_loss"]
            
            self.optimizer.zero_grad()
            if self.amp_training and self.scaler is not None:
                try:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                except Exception as e:
                    logger.warning(f"AMP scaler failed on MPS: {e}. Falling back to regular training.")
                    loss.backward()
                    self.optimizer.step()
            else:
                loss.backward()
                self.optimizer.step()
            
            if self.use_model_ema:
                self.ema_model.update(self.model)
            
            lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
            
            iter_end_time = time.time()
            self.meter.update(
                iter_time=iter_end_time - iter_start_time,
                data_time=data_end_time - iter_start_time,
                lr=lr,
                **outputs,
            )
        
        trainer.train_one_iter = mps_train_one_iter.__get__(trainer, type(trainer))
        
        logger.info("Trainer patched for MPS device")
    
    return trainer


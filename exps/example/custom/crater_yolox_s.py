#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        
        # Model configuration - YOLOX-M variant
        self.depth = 0.67
        self.width = 0.75
        self.num_classes = 5  # Crater classifications 0-4
        
        # Experiment name
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        
        # Pre-trained weights path (optional)
        # Download YOLOX-m weights from: https://github.com/Megvii-BaseDetection/YOLOX/releases
        # Place the .pth file in YOLOX/weights/ directory
        # Then use: -c weights/yolox_m.pth when training
        
        # Dataset configuration
        # Path relative to YOLOX root directory (go up 4 levels from exps/example/custom/ to YOLOX root)
        yolox_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        # Go up one more level to workspace root
        workspace_root = os.path.dirname(yolox_root)
        self.data_dir = workspace_root  # Parent directory containing data/

        # COCO annotation files (will be created by convert_csv_to_coco.py)
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"

        # Training configuration
        self.max_epoch = 300
        self.data_num_workers = 4
        self.input_size = (640, 640)  # (height, width)
        self.test_size = (640, 640)
        
        # Transform configuration
        self.mosaic_prob = 1.0
        self.mixup_prob = 1.0
        self.hsv_prob = 1.0
        self.flip_prob = 0.5
        
        # Learning rate configuration
        self.warmup_epochs = 5
        self.min_lr_ratio = 0.05
        self.basic_lr_per_img = 0.01 / 64.0
        
        # Device-specific optimizations
        # These will be auto-detected, but you can override if needed
        # MPS (MacBook M1): Use smaller batch size, disable some optimizations
        # CUDA: Full optimizations enabled
        self.enable_mps_optimizations = True  # Auto-detect MPS and optimize

    def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
        """
        Get training dataset.
        
        Args:
            cache: Whether to cache images
            cache_type: "ram" or "disk"
        """
        from yolox.data import COCODataset, TrainTransform

        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            name="",  # Empty name since our paths are relative to data_dir
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=120,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob
            ),
            cache=cache,
            cache_type=cache_type,
        )

    def get_eval_dataset(self, **kwargs):
        """
        Get validation dataset.
        """
        from yolox.data import COCODataset, ValTransform
        legacy = kwargs.get("legacy", False)

        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann,
            name="",  # Empty name since our paths are relative to data_dir
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        """
        Get evaluator for validation.
        Uses custom CraterEvaluator for proper crater detection metrics.
        """
        from yolox.evaluators import COCOEvaluator

        # Use COCOEvaluator with our COCO format data
        return COCOEvaluator(
            dataloader=self.get_eval_loader(batch_size, is_distributed, testdev=testdev, legacy=legacy),
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )


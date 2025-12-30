#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
import csv
import copy
import random
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np

from .datasets_wrapper import CacheDataset, cache_read_img


class CraterDataset(CacheDataset):
    """
    Crater Detection Dataset for YOLOX.
    
    Loads images from data/train/altitudeXX/longitudeYY/orientationZZ_lightWW.png
    and annotations from data/train/altitudeXX/longitudeYY/truth/detections.csv
    """

    def __init__(
        self,
        data_dir: str,
        img_size: Tuple[int, int] = (640, 640),
        preproc=None,
        cache: bool = False,
        cache_type: str = "ram",
        split: str = "train",
        train_ratio: float = 0.8,
        seed: int = 42,
    ):
        """
        Args:
            data_dir: Root directory containing train data (e.g., "data/train")
            img_size: Target image size (height, width)
            preproc: Preprocessing function
            cache: Whether to cache images
            cache_type: "ram" or "disk"
            split: "train" or "val"
            train_ratio: Ratio of data to use for training (rest for validation)
            seed: Random seed for train/val split
        """
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.preproc = preproc
        self.split = split
        
        # Scan directories and collect all images
        self.image_paths = self._scan_images()
        
        # Split into train/val
        random.seed(seed)
        random.shuffle(self.image_paths)
        split_idx = int(len(self.image_paths) * train_ratio)
        
        if split == "train":
            self.image_paths = self.image_paths[:split_idx]
        else:  # val
            self.image_paths = self.image_paths[split_idx:]
        
        self.num_imgs = len(self.image_paths)
        
        # Load all annotations
        self.annotations = self._load_all_annotations()
        
        # For caching, we need to set data_dir to a common parent
        # and path_filename relative to that parent
        # Use the workspace root (parent of data_dir) as the base
        workspace_root = self.data_dir.parent
        
        # Prepare path_filename for caching (relative to workspace root)
        path_filename = [
            str(img_path.relative_to(workspace_root)) 
            for img_path in self.image_paths
        ]
        
        super().__init__(
            input_dimension=img_size,
            num_imgs=self.num_imgs,
            data_dir=str(workspace_root),
            cache_dir_name=f"cache_crater_{split}",
            path_filename=path_filename,
            cache=cache,
            cache_type=cache_type,
        )
        
        print(f"Loaded {self.num_imgs} images for {split} split")
        total_craters = sum(len(annos) for annos in self.annotations.values())
        print(f"Total craters: {total_craters}")

    def _scan_images(self) -> List[Path]:
        """Scan data directory for all PNG images."""
        image_paths = []
        
        # Look for images in altitudeXX/longitudeYY/ directories
        for altitude_dir in sorted(self.data_dir.glob("altitude*")):
            if not altitude_dir.is_dir():
                continue
            for longitude_dir in sorted(altitude_dir.glob("longitude*")):
                if not longitude_dir.is_dir():
                    continue
                # Find all PNG files (excluding truth directory)
                for img_file in sorted(longitude_dir.glob("*.png")):
                    if img_file.parent.name != "truth":  # Skip truth directory
                        image_paths.append(img_file)
        
        return image_paths

    def _load_all_annotations(self) -> dict:
        """Load annotations from all CSV files."""
        annotations = {}
        
        for img_path in self.image_paths:
            # Find corresponding CSV file
            csv_path = img_path.parent / "truth" / "detections.csv"
            
            if not csv_path.exists():
                # No annotations for this image
                annotations[str(img_path)] = []
                continue
            
            # Load annotations for this image
            img_filename = img_path.name
            img_annotations = self._load_annotations_from_csv(csv_path, img_filename)
            annotations[str(img_path)] = img_annotations
        
        return annotations

    def _load_annotations_from_csv(self, csv_path: Path, img_filename: str) -> List[np.ndarray]:
        """
        Load annotations from CSV file for a specific image.
        
        Returns:
            List of annotations as [class, xmin, ymin, xmax, ymax]
        """
        annotations = []
        
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Check if this row belongs to the current image
                    if row.get('inputImage', '') != img_filename:
                        continue
                    
                    # Skip invalid detections
                    try:
                        xmin = float(row.get('boundingBoxMinX(px)', -1))
                        ymin = float(row.get('boundingBoxMinY(px)', -1))
                        xmax = float(row.get('boundingBoxMaxX(px)', -1))
                        ymax = float(row.get('boundingBoxMaxY(px)', -1))
                        class_id = int(row.get('crater_classification', -1))
                    except (ValueError, KeyError):
                        continue
                    
                    # Skip invalid entries
                    if xmin < 0 or ymin < 0 or xmax <= xmin or ymax <= ymin:
                        continue
                    
                    # Skip invalid class - ensure class_id is in range [0, num_classes-1]
                    # With num_classes=5, valid classes are 0-4
                    # Also check for negative values and non-integer types
                    try:
                        class_id = int(float(row.get('crater_classification', -1)))
                        if class_id < 0 or class_id >= 5:
                            print(f"Warning: Skipping invalid class_id {class_id} in {csv_path}")
                            continue
                    except (ValueError, TypeError):
                        print(f"Warning: Could not parse class_id '{row.get('crater_classification', 'N/A')}' in {csv_path}")
                        continue
                    
                    # Convert to YOLOX format: [class, xmin, ymin, xmax, ymax]
                    # YOLOX expects class first, then bbox coordinates
                    annotation = np.array([class_id, xmin, ymin, xmax, ymax], dtype=np.float32)
                    annotations.append(annotation)
        except Exception as e:
            print(f"Warning: Failed to load annotations from {csv_path}: {e}")
        
        return annotations

    def __len__(self):
        return self.num_imgs

    def load_anno(self, index):
        """Load annotations for a given index."""
        img_path = self.image_paths[index]
        annotations = self.annotations.get(str(img_path), [])
        
        if len(annotations) == 0:
            return np.zeros((0, 5), dtype=np.float32)
        
        # Stack all annotations
        return np.vstack(annotations)

    def load_image(self, index):
        """Load original image without resizing."""
        img_path = self.image_paths[index]
        img = cv2.imread(str(img_path))
        
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        return img

    def load_resized_img(self, index):
        """Load and resize image."""
        img = self.load_image(index)
        height, width = img.shape[:2]
        
        r = min(self.img_size[0] / height, self.img_size[1] / width)
        resized_img = cv2.resize(
            img,
            (int(width * r), int(height * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        
        return resized_img

    @cache_read_img(use_cache=True)
    def read_img(self, index):
        """Read image (cached)."""
        return self.load_resized_img(index)

    def pull_item(self, index):
        """
        Returns the original image and target at an index.
        
        Returns:
            img: numpy array (BGR, resized)
            target: numpy array [N, 5] with [class, xmin, ymin, xmax, ymax]
            img_info: tuple (height, width) of original image
            img_id: image identifier
        """
        img_path = self.image_paths[index]
        img = self.read_img(index)
        
        # Get original image size
        orig_img = self.load_image(index)
        orig_height, orig_width = orig_img.shape[:2]
        img_info = (orig_height, orig_width)
        
        # Get annotations
        annotations = self.annotations.get(str(img_path), [])
        
        if len(annotations) == 0:
            target = np.zeros((0, 5), dtype=np.float32)
        else:
            # Stack annotations
            target = np.vstack(annotations)
            
            # Validate class IDs before scaling (extra safety check)
            # Format is [class, xmin, ymin, xmax, ymax]
            valid_mask = (target[:, 0] >= 0) & (target[:, 0] < 5)
            if not valid_mask.all():
                invalid_count = (~valid_mask).sum()
                print(f"Warning: Found {invalid_count} annotations with invalid class IDs in {img_path}")
                print(f"  Invalid class IDs: {target[~valid_mask, 0]}")
                target = target[valid_mask]

                if len(target) == 0:
                    print(f"Warning: All annotations filtered out for {img_path}")
                    target = np.zeros((0, 5), dtype=np.float32)
            
            if len(target) == 0:
                target = np.zeros((0, 5), dtype=np.float32)
            else:
                # Scale bounding boxes to resized image
                # Format is [class, xmin, ymin, xmax, ymax], so scale columns 1-4
                r = min(self.img_size[0] / orig_height, self.img_size[1] / orig_width)
                target[:, 1:5] *= r  # Scale xmin, ymin, xmax, ymax (class is in column 0)
                
                # Clip bounding boxes to image boundaries
                target[:, 1] = np.clip(target[:, 1], 0, self.img_size[1] - 1)  # xmin
                target[:, 2] = np.clip(target[:, 2], 0, self.img_size[0] - 1)  # ymin
                target[:, 3] = np.clip(target[:, 3], target[:, 1] + 1, self.img_size[1])  # xmax
                target[:, 4] = np.clip(target[:, 4], target[:, 2] + 1, self.img_size[0])  # ymax
        
        img_id = np.array([index])
        
        return img, copy.deepcopy(target), img_info, img_id

    @CacheDataset.mosaic_getitem
    def __getitem__(self, index):
        """
        Get item with preprocessing.
        
        Returns:
            img: preprocessed image
            target: preprocessed labels [max_labels, 5]
            img_info: (height, width)
            img_id: image identifier
        """
        img, target, img_info, img_id = self.pull_item(index)
        
        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)
        
        return img, target, img_info, img_id


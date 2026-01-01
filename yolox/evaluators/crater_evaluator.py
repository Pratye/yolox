#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Crater Detection Evaluator
Custom evaluator for NASA crater detection dataset with CSV annotations.
"""

import csv
import json
import os
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from loguru import logger
try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False
from tqdm import tqdm

from yolox.utils import (
    gather,
    is_main_process,
    synchronize,
    time_synchronized,
    xyxy2xywh
)
from yolox.utils.boxes import postprocess


class CraterEvaluator:
    """
    Crater Detection Evaluation class.
    Evaluates crater detection performance using CSV ground truth annotations.
    """

    def __init__(
        self,
        dataloader,
        img_size: int,
        confthre: float,
        nmsthre: float,
        num_classes: int,
        testdev: bool = False,
        per_class_metrics: bool = True,
    ):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size: image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre: confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre: IoU threshold of non-max supression ranging from 0 to 1.
            num_classes: number of classes to evaluate.
            testdev: if True, save results to json file.
            per_class_metrics: Show per class metrics during evaluation.
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.testdev = testdev
        self.per_class_metrics = per_class_metrics

        # Load ground truth annotations for validation set
        self.ground_truth = self._load_ground_truth()

    def _load_ground_truth(self):
        """
        Load ground truth annotations from CSV files for the validation dataset.

        Returns:
            dict: Ground truth annotations keyed by image index
                  Format: {img_index: [{'bbox': [x1,y1,x2,y2], 'class': class_id}, ...]}
        """
        ground_truth = defaultdict(list)

        # Get the dataset from dataloader
        dataset = self.dataloader.dataset

        # For validation, we need to load annotations for all images in the dataset
        # The dataset should have image_paths and annotations
        if hasattr(dataset, 'image_paths') and hasattr(dataset, 'annotations'):
            for idx, img_path in enumerate(dataset.image_paths):
                if str(img_path) in dataset.annotations:
                    annos = dataset.annotations[str(img_path)]
                    for anno in annos:
                        # anno format: [class_id, x1, y1, x2, y2]
                        if len(anno) >= 5:
                            class_id, x1, y1, x2, y2 = anno[:5]
                            ground_truth[idx].append({
                                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                'class': int(class_id)
                            })

        logger.info(f"Loaded ground truth for {len(ground_truth)} validation images")
        total_gt = sum(len(annos) for annos in ground_truth.values())
        logger.info(f"Total ground truth craters: {total_gt}")

        return ground_truth

    def evaluate(
        self, model, distributed=False, half=False, trt_file=None,
        decoder=None, test_size=None, return_outputs=False
    ):
        """
        Crater Detection Evaluation. Iterate inference on the test dataset
        and compute precision, recall, F1, and mAP metrics.

        Args:
            model: model to evaluate.

        Returns:
            ap50_95 (float): mean Average Precision across IoU thresholds 50:95
            ap50 (float): Average Precision at IoU=50
            summary (dict): summary info of evaluation with per-class metrics
        """
        # Get model device for tensor operations
        model_device = next(model.parameters()).device

        # Set tensor type based on device and precision
        if model_device.type == "cuda":
            tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        elif model_device.type == "mps":
            tensor_type = torch.HalfTensor if half else torch.FloatTensor
        else:  # CPU
            tensor_type = torch.HalfTensor if half else torch.FloatTensor

        model = model.eval()
        if half:
            model = model.half()

        ids = []
        data_list = []
        output_data = defaultdict(list)
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        nms_time = 0
        n_samples = max(len(self.dataloader) - 1, 1)

        if trt_file is not None:
            from torch2trt import TRTModule
            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).to(model_device)
            model(x)
            model = model_trt

        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                imgs = imgs.type(tensor_type)
                imgs = imgs.to(model_device)

                # Skip the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)

                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

                outputs = postprocess(
                    outputs, self.num_classes, self.confthre,
                    self.nmsthre, class_agnostic=True
                )

                if is_time_record:
                    nms_end = time_synchronized()
                    nms_time += nms_end - infer_end

            data_list_elem, image_wise_data = self.convert_to_coco_format(
                outputs, info_imgs, ids, return_outputs
            )
            data_list.extend(data_list_elem)
            output_data.update(image_wise_data)

        statistics = torch.FloatTensor([inference_time, nms_time, n_samples]).to(model_device)
        if distributed:
            synchronize()
            data_list = gather(data_list, dst=0)
            output_data = gather(output_data, dst=0)
            data_list = list(itertools.chain(*data_list))
            output_data = dict(itertools.chain(*output_data.items()))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_predictions(data_list, statistics)

        synchronize()

        if return_outputs:
            return eval_results, output_data
        return eval_results

    def convert_to_coco_format(self, outputs, info_imgs, ids, return_outputs=False):
        """
        Convert model outputs to COCO-like format for evaluation.
        """
        data_list = []
        image_wise_data = defaultdict(list)

        for (output, img_h, img_w, img_id) in zip(
            outputs, info_imgs[0], info_imgs[1], ids
        ):
            if output is None:
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]

            # preprocessing: resize
            scale = min(
                self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
            )
            bboxes /= scale

            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]

            for ind in range(bboxes.shape[0]):
                label = int(cls[ind])  # For crater dataset, class IDs are 0-4
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "bbox": [round(x, 3) for x in bboxes[ind].tolist()],
                    "score": round(scores[ind].item(), 5),
                    "segmentation": [],
                }
                data_list.append(pred_data)

            if return_outputs:
                image_wise_data.update({img_id: output})

        return data_list, image_wise_data

    def evaluate_predictions(self, predictions, statistics):
        """
        Evaluate predictions against ground truth and compute metrics.

        Args:
            predictions: List of prediction dictionaries
            statistics: Timing statistics tensor

        Returns:
            tuple: (mAP, AP50, summary_dict)
        """
        inference_time = statistics[0].item()
        nms_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_nms_time = 1000 * nms_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join([
            f"Average forward time: {a_infer_time:.2f} ms",
            f"Average NMS time: {a_nms_time:.2f} ms",
            f"Average inference time: {a_infer_time + a_nms_time:.2f} ms"
        ])

        logger.info(time_info)

        if not predictions:
            logger.warning("No predictions to evaluate!")
            return 0.0, 0.0, {"bbox": {"mAP": 0.0, "AP50": 0.0, "AP75": 0.0}}

        # Group predictions by image
        pred_by_image = defaultdict(list)
        for pred in predictions:
            img_id = pred['image_id']
            pred_by_image[img_id].append({
                'bbox': pred['bbox'],
                'score': pred['score'],
                'class': pred['category_id']
            })

        # Compute metrics
        metrics = self._compute_metrics(pred_by_image)

        # Create summary
        summary = {
            "bbox": {
                "mAP": metrics['mAP'],
                "AP50": metrics['AP50'],
                "AP75": metrics['AP75']
            }
        }

        if self.per_class_metrics and 'per_class' in metrics:
            logger.info("Per-class AP:")
            if HAS_TABULATE:
                per_class_table = []
                for class_id, ap in metrics['per_class'].items():
                    per_class_table.append([f"Class {class_id}", ".3f"])
                logger.info(tabulate(per_class_table, headers=["Class", "AP"], tablefmt="pipe"))
            else:
                for class_id, ap in metrics['per_class'].items():
                    logger.info(f"  Class {class_id}: AP = {ap:.3f}")

        return metrics['mAP'], metrics['AP50'], summary

    def _compute_metrics(self, pred_by_image, iou_thresholds=[0.5, 0.75]):
        """
        Compute precision, recall, and AP metrics.

        Args:
            pred_by_image: Predictions grouped by image ID
            iou_thresholds: IoU thresholds for evaluation

        Returns:
            dict: Computed metrics
        """
        all_tp = defaultdict(list)  # True positives for each IoU threshold
        all_fp = defaultdict(list)  # False positives for each IoU threshold
        all_scores = defaultdict(list)  # Confidence scores
        all_gt_counts = defaultdict(int)  # Ground truth counts per class

        # Process each image
        for img_id, preds in pred_by_image.items():
            # img_id should correspond to dataset index
            if img_id not in self.ground_truth:
                continue

            gt_annos = self.ground_truth[img_id]

            # Group predictions and ground truth by class
            for class_id in range(self.num_classes):
                pred_class = [p for p in preds if p['class'] == class_id]
                gt_class = [gt for gt in gt_annos if gt['class'] == class_id]

                if not gt_class:
                    # No ground truth for this class in this image
                    continue

                all_gt_counts[class_id] += len(gt_class)

                # Sort predictions by confidence (descending)
                pred_class.sort(key=lambda x: x['score'], reverse=True)

                # Match predictions to ground truth
                matched_gt = set()
                for pred in pred_class:
                    best_iou = 0
                    best_gt_idx = -1

                    # Find best matching ground truth
                    for gt_idx, gt in enumerate(gt_class):
                        if gt_idx in matched_gt:
                            continue

                        iou = self._compute_iou(pred['bbox'], gt['bbox'])
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx

                    # Check if match meets IoU threshold
                    for iou_thresh in iou_thresholds:
                        if best_iou >= iou_thresh and best_gt_idx != -1:
                            if best_gt_idx not in matched_gt:
                                all_tp[(class_id, iou_thresh)].append(1)
                                all_fp[(class_id, iou_thresh)].append(0)
                                matched_gt.add(best_gt_idx)
                            else:
                                all_tp[(class_id, iou_thresh)].append(0)
                                all_fp[(class_id, iou_thresh)].append(1)
                        else:
                            all_tp[(class_id, iou_thresh)].append(0)
                            all_fp[(class_id, iou_thresh)].append(1)

                        all_scores[(class_id, iou_thresh)].append(pred['score'])

        # Compute AP for each class and IoU threshold
        per_class_ap = {}
        all_ap_50 = []
        all_ap_75 = []

        for class_id in range(self.num_classes):
            if all_gt_counts[class_id] == 0:
                continue

            for iou_thresh in iou_thresholds:
                key = (class_id, iou_thresh)
                if key not in all_tp or not all_tp[key]:
                    continue

                tp = np.array(all_tp[key])
                fp = np.array(all_fp[key])
                scores = np.array(all_scores[key])

                # Sort by confidence score (already sorted, but ensure)
                sort_idx = np.argsort(scores)[::-1]
                tp = tp[sort_idx]
                fp = fp[sort_idx]

                # Compute precision and recall
                tp_cumsum = np.cumsum(tp)
                fp_cumsum = np.cumsum(fp)

                precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
                recall = tp_cumsum / all_gt_counts[class_id]

                # Compute AP using 11-point interpolation (VOC style)
                ap = self._compute_ap(recall, precision)

                per_class_ap[f"{class_id}_{iou_thresh}"] = ap

                if iou_thresh == 0.5:
                    all_ap_50.append(ap)
                elif iou_thresh == 0.75:
                    all_ap_75.append(ap)

        # Compute mean AP
        mAP = np.mean(list(per_class_ap.values())) if per_class_ap else 0.0
        AP50 = np.mean(all_ap_50) if all_ap_50 else 0.0
        AP75 = np.mean(all_ap_75) if all_ap_75 else 0.0

        return {
            'mAP': mAP,
            'AP50': AP50,
            'AP75': AP75,
            'per_class': per_class_ap
        }

    def _compute_iou(self, bbox1, bbox2):
        """
        Compute IoU between two bounding boxes.

        Args:
            bbox1, bbox2: [x1, y1, x2, y2] format

        Returns:
            float: IoU value
        """
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _compute_ap(self, recall, precision):
        """
        Compute Average Precision using 11-point interpolation (VOC style).

        Args:
            recall: Array of recall values
            precision: Array of precision values

        Returns:
            float: Average Precision
        """
        # Add sentinel values
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([0.0], precision, [0.0]))

        # Make precision monotonically decreasing
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # Find points where recall changes
        indices = np.where(mrec[1:] != mrec[:-1])[0] + 1

        # Compute AP
        ap = 0.0
        for i in indices:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]

        return ap

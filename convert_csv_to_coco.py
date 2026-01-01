#!/usr/bin/env python3
"""
Convert NASA Crater Detection CSV annotations to COCO format.

This script converts the CSV-based crater annotations to COCO format
so that YOLOX's built-in COCO evaluator can be used directly.
"""

import csv
import json
import os
from pathlib import Path
from PIL import Image
import argparse


def get_image_size(image_path):
    """Get image dimensions."""
    try:
        with Image.open(image_path) as img:
            return img.size  # (width, height)
    except Exception as e:
        print(f"Warning: Could not get size for {image_path}: {e}")
        return None


def convert_csv_to_coco(data_dir, output_dir, split="train"):
    """
    Convert CSV annotations to COCO format.

    Args:
        data_dir: Root directory containing the data (e.g., "data/train")
        output_dir: Directory to save COCO format files
        split: "train" or "val"
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # COCO format structure
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Define categories (crater classifications 0-4)
    category_names = [f"crater_{i}" for i in range(5)]
    for i, name in enumerate(category_names):
        coco_data["categories"].append({
            "id": i,
            "name": name,
            "supercategory": "crater"
        })

    # Collect all image paths and determine train/val split
    all_images = []
    for altitude_dir in sorted(data_dir.glob("altitude*")):
        if not altitude_dir.is_dir():
            continue
        for longitude_dir in sorted(altitude_dir.glob("longitude*")):
            if not longitude_dir.is_dir():
                continue
            # Find all PNG files (excluding truth directory)
            for img_file in sorted(longitude_dir.glob("*.png")):
                if img_file.parent.name != "truth":
                    all_images.append(img_file)

    # Split train/val (80/20)
    import random
    random.seed(42)  # For reproducible splits
    random.shuffle(all_images)
    split_idx = int(len(all_images) * 0.8)

    if split == "train":
        images_to_process = all_images[:split_idx]
    else:  # val
        images_to_process = all_images[split_idx:]

    print(f"Processing {len(images_to_process)} images for {split} split")

    # Process images and annotations
    image_id = 0
    annotation_id = 0

    for img_path in images_to_process:
        # Calculate relative path from COCO data_dir to actual image location
        # COCO data_dir: /content/yolox/datasets/coco_crater/
        # Images at: /content/data/train/...
        coco_data_dir = Path(output_dir)  # This will be /content/yolox/datasets/coco_crater/
        try:
            file_name = os.path.relpath(str(img_path), str(coco_data_dir))
        except ValueError:
            # If relative path calculation fails, use absolute path
            file_name = str(img_path.resolve())

        # Get image dimensions
        img_size = get_image_size(img_path)
        if img_size is None:
            print(f"Skipping {img_path} due to size error")
            continue

        width, height = img_size

        # Add image to COCO data
        coco_data["images"].append({
            "id": image_id,
            "file_name": file_name,
            "width": width,
            "height": height
        })

        # Find corresponding CSV file
        csv_path = img_path.parent / "truth" / "detections.csv"
        if not csv_path.exists():
            print(f"Warning: No CSV found for {img_path}")
            image_id += 1
            continue

        # Parse CSV annotations
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Check if this row belongs to current image
                    if row.get('inputImage', '') != img_path.name:
                        continue

                    # Extract bounding box coordinates
                    try:
                        x1 = float(row.get('boundingBoxMinX(px)', -1))
                        y1 = float(row.get('boundingBoxMinY(px)', -1))
                        x2 = float(row.get('boundingBoxMaxX(px)', -1))
                        y2 = float(row.get('boundingBoxMaxY(px)', -1))
                        class_id = int(float(row.get('crater_classification', -1)))
                    except (ValueError, KeyError):
                        continue

                    # Validate coordinates and class
                    if (x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1 or
                        class_id < 0 or class_id > 4):
                        continue

                    # Convert to COCO bbox format: [x, y, width, height]
                    bbox = [x1, y1, x2 - x1, y2 - y1]
                    area = bbox[2] * bbox[3]

                    # Add annotation
                    coco_data["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": class_id,
                        "bbox": bbox,
                        "area": area,
                        "iscrowd": 0
                    })

                    annotation_id += 1

        except Exception as e:
            print(f"Error processing CSV {csv_path}: {e}")

        image_id += 1

    # Save COCO format JSON
    # Create annotations subdirectory for COCO format
    annotations_dir = output_dir / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)

    output_file = annotations_dir / f"instances_{split}2017.json"
    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=2)

    print(f"Saved {len(coco_data['images'])} images and {len(coco_data['annotations'])} annotations to {output_file}")
    print(f"Categories: {len(coco_data['categories'])}")

    return coco_data


def main():
    parser = argparse.ArgumentParser(description="Convert CSV crater annotations to COCO format")
    parser.add_argument("--data_dir", type=str, default="../data/train",
                       help="Root directory containing crater data")
    parser.add_argument("--output_dir", type=str, default="datasets/coco_crater",
                       help="Output directory for COCO format files")
    parser.add_argument("--split", type=str, choices=["train", "val", "both"], default="both",
                       help="Which split to convert")

    args = parser.parse_args()

    if args.split == "both":
        # Convert both train and val
        print("Converting train split...")
        convert_csv_to_coco(args.data_dir, args.output_dir, "train")
        print("Converting val split...")
        convert_csv_to_coco(args.data_dir, args.output_dir, "val")
    else:
        convert_csv_to_coco(args.data_dir, args.output_dir, args.split)

    print("Conversion complete!")


if __name__ == "__main__":
    main()

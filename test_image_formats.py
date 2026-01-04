#!/usr/bin/env python3
"""
Test script to verify image format handling for non-RGB imagery.
"""

import cv2
import numpy as np
from PIL import Image
import tempfile
import os

def create_test_images():
    """Create test images of different formats."""
    test_images = {}

    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()

    # 1. Grayscale image
    gray_img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    gray_path = os.path.join(temp_dir, 'gray.png')
    cv2.imwrite(gray_path, gray_img)
    test_images['grayscale'] = gray_path

    # 2. RGB image
    rgb_img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    rgb_path = os.path.join(temp_dir, 'rgb.png')
    cv2.imwrite(rgb_path, rgb_img)
    test_images['rgb'] = rgb_path

    # 3. RGBA image
    rgba_img = np.random.randint(0, 256, (100, 100, 4), dtype=np.uint8)
    rgba_path = os.path.join(temp_dir, 'rgba.png')
    cv2.imwrite(rgba_path, rgba_img)
    test_images['rgba'] = rgba_path

    # 4. Single channel saved as 3-channel but identical
    identical_channels = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    identical_channels[:, :, 1] = identical_channels[:, :, 0]  # Make all channels identical
    identical_channels[:, :, 2] = identical_channels[:, :, 0]
    identical_path = os.path.join(temp_dir, 'identical_channels.png')
    cv2.imwrite(identical_path, identical_channels)
    test_images['identical_channels'] = identical_path

    return test_images, temp_dir

def test_image_loading():
    """Test the robust image loading function."""
    from yolox.data.datasets.coco import COCODataset

    # Create a dummy COCODataset just to access the method
    # We'll call the method directly
    test_images, temp_dir = create_test_images()

    print("Testing image loading robustness...")

    for img_type, img_path in test_images.items():
        print(f"\nTesting {img_type} image: {img_path}")

        # Test our robust loading
        try:
            # Create a dummy dataset to access the method
            dataset = COCODataset.__new__(COCODataset)  # Create without __init__
            loaded_img = dataset._load_image_robust(img_path)

            if loaded_img is not None:
                print(f"  ✓ Successfully loaded: shape={loaded_img.shape}, dtype={loaded_img.dtype}")
                if len(loaded_img.shape) == 3:
                    print(f"  ✓ Has {loaded_img.shape[2]} channels")
                else:
                    print(f"  ✗ Still not 3-channel: shape={loaded_img.shape}")
            else:
                print(f"  ✗ Failed to load")

        except Exception as e:
            print(f"  ✗ Error: {e}")

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)
    print("\nTest completed.")

if __name__ == "__main__":
    test_image_loading()

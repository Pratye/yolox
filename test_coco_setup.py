#!/usr/bin/env python3
"""
Test script to verify COCO format crater dataset setup.
"""

import sys
import os
sys.path.insert(0, '.')

def test_coco_dataset():
    """Test that COCO dataset can be loaded and used."""
    try:
        from yolox.exp import get_exp
        from yolox.data import COCODataset, TrainTransform, ValTransform

        # Get experiment
        exp = get_exp("exps/example/custom/crater_yolox_s.py")

        print("✓ Experiment loaded successfully")
        print(f"  Data dir: {exp.data_dir}")
        print(f"  Train ann: {exp.train_ann}")
        print(f"  Val ann: {exp.val_ann}")

        # Test training dataset
        print("\nTesting training dataset...")
        train_dataset = exp.get_dataset(cache=False)
        print(f"✓ Train dataset: {len(train_dataset)} samples")

        # Test validation dataset
        print("Testing validation dataset...")
        val_dataset = exp.get_eval_dataset()
        print(f"✓ Val dataset: {len(val_dataset)} samples")

        # Test evaluator
        print("Testing evaluator...")
        evaluator = exp.get_evaluator(batch_size=4, is_distributed=False)
        print(f"✓ Evaluator created: {type(evaluator).__name__}")

        # Test data loading
        print("\nTesting data loading...")
        sample = train_dataset[0]
        print(f"✓ Sample data type: {type(sample)}")
        if isinstance(sample, tuple):
            print(f"  Tuple length: {len(sample)}")
            if len(sample) >= 2:
                print(f"  Image shape: {sample[0].shape}")
                print(f"  Target shape: {sample[1].shape}")
        else:
            print(f"✓ Sample data keys: {list(sample.keys())}")
            print(f"  Image shape: {sample['img'].shape}")
            print(f"  Target shape: {sample['target'].shape}")

        print("\n🎉 All tests passed! Ready for training with COCO format.")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    test_coco_dataset()

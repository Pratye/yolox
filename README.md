# YOLOX for Crater Detection

This repository contains a customized YOLOX implementation for detecting craters in NASA imagery. The codebase is based on YOLOX with modifications for crater detection, including custom dataset loading, MPS (MacBook M1 GPU) support, and device-aware training.

## Features

- **Custom Crater Dataset Support**: Handles nested directory structure with CSV annotations
- **MPS (MacBook M1) Compatibility**: Full support for training on Apple Silicon GPUs
- **CUDA Support**: Optimized for NVIDIA GPUs
- **Pre-trained Model Fine-tuning**: Support for YOLOX-m pre-trained weights
- **Device-Aware Training**: Automatic device detection and optimization

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.12+ (with MPS support for MacBook M1)
- CUDA toolkit (for NVIDIA GPUs)

### Setup

```bash
# Clone the repository
git clone https://github.com/Pratye/yolo-scratch.git
cd yolo-scratch/YOLOX

# Install dependencies
pip install -r requirements.txt

# Fix environment dependencies (if needed)
bash fix_dependencies.sh

# Install YOLOX
pip install -v -e .
```

## Dataset Structure

The crater detection dataset should be organized as follows:

```
data/
└── train/
    └── altitudeXX/
        └── longitudeYY/
            ├── orientationZZ_lightWW.png  # Images
            └── truth/
                └── detections.csv          # Annotations
```

The CSV file should contain columns:
- `inputImage`: Image filename
- `boundingBoxMinX(px)`, `boundingBoxMinY(px)`: Top-left corner
- `boundingBoxMaxX(px)`, `boundingBoxMaxY(px)`: Bottom-right corner
- `crater_classification`: Class ID (0-4)

## Quick Start

### 1. Download Pre-trained Weights

Download YOLOX-m weights and place them in the `weights/` directory:

```bash
cd weights
wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth
```

See `weights/README.md` for more details.

### 2. Train on Crater Dataset

**For MacBook M1 (MPS):**
```bash
python train_crater_device_aware.py -b 8 -c weights/yolox_m.pth
```

**For CUDA:**
```bash
python train_crater.py -b 16 -d 1 -c weights/yolox_m.pth
```

**Using standard training script:**
```bash
python tools/train.py -f exps/example/custom/crater_yolox_s.py -b 16 -d 1 -c weights/yolox_m.pth
```

### 3. Evaluate

```bash
python tools/eval.py -f exps/example/custom/crater_yolox_s.py -c YOLOX_outputs/crater_yolox_s/best_ckpt.pth -b 64 -d 1
```

## Configuration

The experiment configuration is in `exps/example/custom/crater_yolox_s.py`. Key settings:

- `num_classes = 5`: Crater classification classes (0-4)
- `data_dir`: Path to your dataset
- `max_epoch = 300`: Training epochs
- `input_size = (640, 640)`: Input image size

## Device Support

### MacBook M1 (MPS)
- Automatic device detection
- MPS compatibility patches applied
- Recommended batch size: 8-16
- Mixed precision (FP16) may be slower

### CUDA
- Full CUDA support
- Distributed training available
- Recommended batch size: 16-64 (depending on GPU)

### CPU
- Falls back to CPU if no GPU available
- Slower training speed

## Documentation

- `DEVICE_OPTIMIZATION.md`: Device-specific optimization guide
- `FIX_ENVIRONMENT.md`: Environment dependency fixes
- `MPS_FIX_SUMMARY.md`: MPS compatibility details
- `CLEANUP_SUMMARY.md`: Repository cleanup information

## Project Structure

```
YOLOX/
├── exps/example/custom/
│   └── crater_yolox_s.py          # Crater detection experiment config
├── yolox/
│   ├── data/datasets/
│   │   └── crater.py               # Custom crater dataset loader
│   ├── utils/
│   │   ├── device.py               # Device detection utilities
│   │   └── mps_patch.py            # MPS compatibility patches
│   └── ...
├── train_crater.py                 # Convenience training script
├── train_crater_device_aware.py    # Device-aware training script
└── weights/
    └── README.md                   # Weights download instructions
```

## Troubleshooting

### Environment Issues
If you encounter dependency conflicts (NumPy, Protobuf, TensorFlow):
```bash
bash fix_dependencies.sh
```

### MPS Issues
- Ensure PyTorch 1.12+ with MPS support
- Use `train_crater_device_aware.py` for automatic MPS patches
- Check `MPS_FIX_SUMMARY.md` for known limitations

### Training Issues
- Reduce batch size if running out of memory
- Check dataset path in experiment config
- Verify CSV annotation format

## License

This project is based on YOLOX and is licensed under the Apache License 2.0. See LICENSE file for details.

## Acknowledgments

- Based on [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) by Megvii
- Adapted for NASA Crater Detection Challenge

## Citation

This project is based on YOLOX. For academic use, please refer to the original YOLOX paper: [arXiv:2107.08430](https://arxiv.org/abs/2107.08430)

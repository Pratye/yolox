# Fix Environment Dependencies

Your environment has dependency conflicts. Here's how to fix them:

## Quick Fix

Run these commands in your conda environment:

```bash
# Activate your environment
conda activate TF_MPS

# Fix NumPy (TensorFlow requires NumPy < 2.0)
pip install "numpy<2.0.0,>=1.23.5"

# Fix Protobuf (TensorFlow requires protobuf < 5.0)
pip install "protobuf<5.0.0,>=3.20.3"

# Fix ml-dtypes (TensorFlow requires specific version)
pip install "ml-dtypes~=0.3.1"

# Reinstall tensorboard to fix compatibility
pip install --upgrade tensorboard
```

## Complete Environment Reset (Recommended)

If the quick fix doesn't work, create a fresh environment:

```bash
# Create new environment
conda create -n yolox python=3.10 -y
conda activate yolox

# Install PyTorch with MPS support (for MacBook M1)
pip install torch torchvision torchaudio

# Install YOLOX requirements
cd YOLOX
pip install -r requirements.txt

# Fix version conflicts
pip install "numpy<2.0.0,>=1.23.5"
pip install "protobuf<5.0.0,>=3.20.3"
pip install "ml-dtypes~=0.3.1"

# Install additional dependencies
pip install loguru tqdm scipy pycocotools tabulate thop ninja
pip install tensorboard onnx onnx-simplifier
```

## Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available() if hasattr(torch.backends, \"mps\") else False}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import protobuf; print(f'Protobuf: {protobuf.__version__}')" 2>/dev/null || python -c "import google.protobuf; print(f'Protobuf: {google.protobuf.__version__}')"
```

## After Fixing

Try training again:

```bash
cd YOLOX
python train_crater_device_aware.py -b 16 -c weights/yolox_m.pth
```

## Alternative: Use Standard Training Script

If device-aware script still has issues, use the standard script:

```bash
cd YOLOX
python tools/train.py -f exps/example/custom/crater_yolox_s.py -b 16 -d 1 -c weights/yolox_m.pth
```

## Note About TensorFlow

If you don't need TensorFlow, you can uninstall it to avoid conflicts:

```bash
pip uninstall tensorflow tensorflow-macos tensorflow-metal -y
```

YOLOX only needs TensorBoard (which has its own TensorFlow dependency), but the versions should be compatible.


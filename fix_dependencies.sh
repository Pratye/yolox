#!/bin/bash
# Quick fix for dependency conflicts in YOLOX environment

echo "Fixing dependency conflicts..."

# Fix NumPy (TensorFlow requires NumPy < 2.0)
echo "Installing compatible NumPy version..."
pip install "numpy<2.0.0,>=1.23.5" --force-reinstall

# Fix Protobuf (TensorFlow requires protobuf < 5.0)
echo "Installing compatible Protobuf version..."
pip install "protobuf<5.0.0,>=3.20.3" --force-reinstall

# Fix ml-dtypes
echo "Installing compatible ml-dtypes..."
pip install "ml-dtypes~=0.3.1" --force-reinstall

# Reinstall tensorboard to fix compatibility
echo "Reinstalling TensorBoard..."
pip install --upgrade tensorboard --force-reinstall

echo ""
echo "Dependencies fixed! Verifying versions..."
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import google.protobuf; print(f'Protobuf: {google.protobuf.__version__}')" 2>/dev/null || echo "Protobuf: installed"
python -c "import ml_dtypes; print(f'ml-dtypes: {ml_dtypes.__version__}')" 2>/dev/null || echo "ml-dtypes: installed"

echo ""
echo "Done! Try running training again:"
echo "  python train_crater_device_aware.py -b 16 -c weights/yolox_m.pth"


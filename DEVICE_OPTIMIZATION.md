# Device Optimization Guide

This guide explains how YOLOX training is optimized for different devices.

## Supported Devices

### 1. CUDA (NVIDIA GPUs)
- **Full Support**: All features enabled
- **Distributed Training**: Supported
- **Mixed Precision (FP16)**: Fully supported
- **Recommended Batch Size**: 64 (adjust based on GPU memory)

### 2. MPS (MacBook M1/M2/M3 GPU)
- **Limited Support**: Some features may be slower or unavailable
- **Distributed Training**: Not supported (single device only)
- **Mixed Precision (FP16)**: Limited support (may be slower)
- **Recommended Batch Size**: 16-32 (MPS has unified memory constraints)
- **Note**: MPS is still experimental in PyTorch, some operations may fall back to CPU

### 3. CPU
- **Fallback Only**: Very slow, not recommended for training
- **Recommended Batch Size**: 4-8

## Usage

### Automatic Device Detection

Use the device-aware training script:

```bash
cd YOLOX
python train_crater_device_aware.py -c weights/yolox_m.pth
```

This script will:
- Automatically detect your device (CUDA > MPS > CPU)
- Optimize batch size for your device
- Adjust training settings accordingly

### Manual Device Selection

#### For CUDA:
```bash
python tools/train.py -f exps/example/custom/crater_yolox_s.py -b 64 -d 1 -c weights/yolox_m.pth
```

#### For MPS (MacBook M1):
```bash
# Use device-aware script (recommended)
python train_crater_device_aware.py -b 16 -c weights/yolox_m.pth

# Or use standard script with reduced batch size
python tools/train.py -f exps/example/custom/crater_yolox_s.py -b 16 -d 1 -c weights/yolox_m.pth
```

## MPS-Specific Considerations

1. **Batch Size**: Start with 16, increase if you have enough memory
2. **Mixed Precision**: May be slower than CUDA, consider disabling with `--no-fp16`
3. **Memory**: MPS uses unified memory, monitor usage carefully
4. **Performance**: MPS is typically 2-4x slower than equivalent CUDA GPU

## CUDA Optimizations

1. **Batch Size**: Can use larger batches (64+)
2. **Mixed Precision**: Fully supported and recommended (`--fp16`)
3. **Distributed Training**: Supported for multi-GPU setups
4. **CUDNN**: Enabled by default for optimal performance

## Troubleshooting

### MPS Issues

If you encounter errors with MPS:
1. Try reducing batch size further (8 or 4)
2. Disable mixed precision: remove `--fp16` flag
3. Use CPU fallback: set `CUDA_VISIBLE_DEVICES=""` (not applicable, but shows fallback)

### CUDA Out of Memory

1. Reduce batch size: `-b 32` or `-b 16`
2. Use gradient accumulation (modify training script)
3. Enable image caching: `--cache ram` or `--cache disk`

## Performance Benchmarks

Approximate training speeds (YOLOX-M, batch size 16, 640x640 images):

- **CUDA (RTX 3090)**: ~0.5-1.0 seconds per iteration
- **MPS (M1 Max)**: ~1.5-3.0 seconds per iteration
- **CPU (M1)**: ~10-20 seconds per iteration

*Note: Actual speeds vary based on model size, batch size, and data loading*


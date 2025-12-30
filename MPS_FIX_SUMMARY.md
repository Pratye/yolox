# MPS (MacBook M1) Training Fix Summary

## Problem

The YOLOX trainer is hardcoded to use CUDA functions, which fail on MPS devices:
- `torch.cuda.set_device()` doesn't exist for MPS
- `torch.cuda.amp.GradScaler` may not work properly
- Distributed training (DDP) is not supported

## Solution

Created MPS compatibility patches that:

1. **Skip CUDA device setup** - MPS doesn't need `torch.cuda.set_device()`
2. **Disable AMP scaler** - MPS has limited mixed precision support
3. **Disable distributed training** - MPS only supports single device
4. **Patch autocast** - Use CPU fallback or disable for MPS

## Files Created

1. **`yolox/utils/mps_patch.py`** - MPS compatibility patches
2. **`train_crater_device_aware.py`** - Updated to apply patches automatically

## Usage

After fixing NumPy/Protobuf dependencies (see `FIX_ENVIRONMENT.md`), run:

```bash
cd YOLOX
python train_crater_device_aware.py -b 16 -c weights/yolox_m.pth
```

The script will:
1. Detect MPS device
2. Auto-optimize batch size (16 for MPS)
3. Apply MPS compatibility patches
4. Start training

## Important Notes

1. **Fix Dependencies First**: You must fix NumPy/Protobuf issues first:
   ```bash
   bash fix_dependencies.sh
   ```

2. **MPS Limitations**:
   - No distributed training
   - Limited AMP support (disabled by default)
   - Slower than CUDA (2-4x)
   - Some operations may fall back to CPU

3. **Batch Size**: Start with 16, increase if you have enough memory

4. **Mixed Precision**: Disabled by default for MPS (use `--fp16` at your own risk)

## Troubleshooting

If you still get CUDA-related errors:
1. Make sure you're using `train_crater_device_aware.py` (not `tools/train.py`)
2. Check that MPS is available: `python -c "import torch; print(torch.backends.mps.is_available())"`
3. Verify device detection: The script should log "Using MPS device"

## Alternative: Use Standard Script with Manual Patches

If the device-aware script doesn't work, you can manually patch:

```python
from yolox.utils.mps_patch import patch_trainer_for_mps
trainer = exp.get_trainer(args)
trainer.device = "mps"
patch_trainer_for_mps(trainer)
trainer.train()
```


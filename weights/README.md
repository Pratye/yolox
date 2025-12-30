# Pre-trained Weights Directory

Place your pre-trained YOLOX model weights (.pth files) in this directory.

## Download YOLOX-m Pre-trained Weights

You can download the official YOLOX-m weights from:

**Official Release:**
- https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth

**Direct Download:**
```bash
cd YOLOX/weights
wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth
```

Or download manually and place the file here as `yolox_m.pth`.

## Usage

After placing the weights file in this directory, use the `-c` flag when training:

```bash
cd YOLOX
python tools/train.py -f exps/example/custom/crater_yolox_s.py -b 64 -d 1 -c weights/yolox_m.pth
```

Or using the convenience script:

```bash
cd YOLOX
python train_crater.py -b 64 -d 1 -c weights/yolox_m.pth
```

## Note

The checkpoint file should be a PyTorch `.pth` file containing a dictionary with a "model" key. The training script will automatically:
- Load the pre-trained weights
- Skip incompatible layers (e.g., the final classification head since we have 5 classes instead of 80)
- Fine-tune the model on your crater detection dataset


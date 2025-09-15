# Detailed Setup Guide

## Prerequisites
- Python 3.9+ (3.11 recommended)
- CUDA GPU with 12GB+ VRAM
- Google Drive account (for Colab)

## Setup Options

### Colab Training (Recommended)
1. Open [colab_train.ipynb](colab_train.ipynb) in Google Colab
2. Upload `idd20k_lite` dataset to `/content/drive/MyDrive/idd20k_lite/`
3. Run all cells sequentially

**Runtime:** 6-8 hours total on T4 GPU

### Local Training
```bash
git clone --recursive https://github.com/harshvardhanchand/idd-semantic-segmentation.git
cd idd-semantic-segmentation
pip install -r requirements.txt
make setup-submodule

# Place dataset in data/idd20k_lite/
make train-pspnet
make train-deeplabv3
make train-deeplabv3plus
```

## Model Details
- **PSPNet ResNet50**: 24.3M params, pyramid pooling
- **DeepLabV3 ResNet50**: 42M params, atrous convolution  
- **DeepLabV3+ MobileNet**: 5.2M params, mobile-optimized

## TensorBoard
```bash
# Local
make tensorboard  # http://localhost:6006

# Colab
%load_ext tensorboard
%tensorboard --logdir runs/
```

## Advanced Commands

### Individual Training
```bash
python train.py src/configs/pspnet.yaml --export-predictions
python train.py src/configs/deeplabv3.yaml --export-predictions
python train.py src/configs/deeplabv3plus.yaml --export-predictions
```

### Evaluation
```bash
make eval-official PREDS=runs/model_name/predictions/val
make export-preds CHECKPOINT=path/to/best.pt CONFIG=src/configs/model.yaml
```

### Visualization
```bash
make visualize-pspnet
make visualize-deeplabv3
make visualize-deeplabv3plus
```

### Domain Gap Analysis
```bash
make domain-gap-full
# OR manually:
make cityscapes-inference
make eval-domain-gap
```

## Troubleshooting
```bash
# GPU memory issues
clear_memory()  # In Colab

# TensorBoard issues
make tensorboard RELOAD=1

# Dataset path issues
# Ensure dataset is in correct location
```

## Memory Management
- Sequential training prevents OOM errors
- Automatic cleanup between models
- Memory monitoring included

## Results Backup
- Automatic Google Drive backup in Colab
- Timestamped results folders
- TensorBoard logs preserved 
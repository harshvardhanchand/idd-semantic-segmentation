# IDD Semantic Segmentation Training

Clean, modular training pipeline for IDD semantic segmentation.

## Quick Start

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Train Models
```bash
# DeepLabV3+ ResNet-50
python train.py src/configs/deeplabv3.yaml

# FCN ResNet-50  
python train.py src/configs/fcn.yaml
```

## Features
- Full pre-trained model weights (DeepLabV3+, FCN)
- SGD + Polynomial LR + Mixed Precision (AMP)
- Differential learning rates (backbone 1x, classifier 10x)
- mIoU tracking with best model saving
- Configuration-driven with YAML inheritance
- Reproducible training with seed control

## Output Structure
```
runs/
├── idd_lite_deeplabv3_r50/
│   ├── checkpoints/best.pt
│   └── experiment_info.json
└── idd_lite_fcn_r50/
    └── ...
```

## Environment
Developed for **Local + Colab** hybrid workflow:
- Local development in Cursor
- Training on Colab T4/A100 GPUs
- Easy transfer via modular Python files 
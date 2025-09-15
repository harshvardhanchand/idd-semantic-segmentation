# 🚗 IDD Semantic Segmentation: Professional Training Suite

<p align="center">
  <a href="https://arxiv.org/abs/1811.10200"><img src="https://img.shields.io/badge/📝-Paper-blue" height="25"></a>
  <a href="https://github.com/AutoNUE/public-code"><img src="https://img.shields.io/badge/🏆-AutoNUE-green" height="25"></a>
  <a href="colab_train.ipynb"><img src="https://img.shields.io/badge/🚀-Colab-orange" height="25"></a>
</p>

## 🌎 Overview

 Semantic segmentation training suite for Indian Driving Dataset with three state-of-the-art models: PSPNet, DeepLabV3, and DeepLabV3+.

| Model | Parameters | mIoU |
|-------|------------|------|
| DeepLabV3 ResNet50 | 42M | **69.23%** |
| DeepLabV3+ MobileNet | 5.2M | **66.65%** |
| PSPNet ResNet50 | 24.3M | **66.18%** |
| DeepLabV3+ Cityscapes Zero-Shot | 5.2M | **37.55%** |

## 🔧 Setup

Clone the repository with submodules and install dependencies:
```bash
git clone --recursive https://github.com/harshvardhanchand/idd-semantic-segmentation.git
cd idd-semantic-segmentation
pip install -r requirements.txt
make setup-submodule
```

Dataset preparation:
- **Colab**: Upload `idd20k_lite` to `/content/drive/MyDrive/idd20k_lite/`
- **Local**: Place dataset in `data/idd20k_lite/`

## 🚀 Quick Start

### Colab Training (Recommended)
```python
# Open colab_train.ipynb and run all cells
# Complete automated pipeline: training → evaluation → visualization → backup
```

### Local Training
```bash
# Train all three models
make train-pspnet
make train-deeplabv3  
make train-deeplabv3plus



# Run domain gap analysis
make domain-gap-full
```

## 📊 Results & Analysis

Generate comprehensive analysis:
```bash
# Create visualizations
make visualize-pspnet
make visualize-deeplabv3
make visualize-deeplabv3plus


```

See [DETAILED_SETUP.md](DETAILED_SETUP.md) for advanced configuration options.

## 🎯 Citation

```bibtex
@article{varma2019idd,
  title={IDD: A Dataset for Exploring Problems of Autonomous Navigation in Unconstrained Environments},
  author={Varma, Girish and Subramanian, Anbumani and Namboodiri, Anoop and Chandraker, Manmohan and Jawahar, CV},
  journal={arXiv preprint arXiv:1811.10200},
  year={2019}
}
``` 
## Acknowledgments
This work utilizes code from the following public repository. Many thanks to the author for their contribution.
https://github.com/VainF/DeepLabV3Plus-Pytorch
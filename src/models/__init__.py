"""Models package for semantic segmentation."""

from .base_model import BaseSegmentationModel
from .deeplabv3plus import DeepLabV3Plus
from .deeplabv3 import DeepLabV3ResNet50
from .pspnet import PSPNet
from .model_factory import ModelFactory, create_model

__all__ = [
    "BaseSegmentationModel",
    "DeepLabV3Plus",
    "DeepLabV3ResNet50",
    "PSPNet",
    "ModelFactory",
    "create_model",
]

# Model registry for easy access
MODELS = {
    "deeplabv3plus": DeepLabV3Plus,
    "deeplabv3_resnet50": DeepLabV3ResNet50,
    "pspnet": PSPNet,
}


def list_models():
    """List all available models"""
    ModelFactory.print_available_models()


def get_model_comparison():
    """Get comparison of all models"""
    return ModelFactory.compare_models()

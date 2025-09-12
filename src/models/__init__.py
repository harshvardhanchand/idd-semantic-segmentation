"""
Models package for IDD semantic segmentation.
Provides clean API for model creation and management.
"""

from .base_model import BaseSegmentationModel
from .deeplabv3 import DeepLabV3Plus
from .fcn import FCN
from .model_factory import ModelFactory, create_model

# Define what gets imported with "from src.models import *"
__all__ = [
    "BaseSegmentationModel",
    "DeepLabV3Plus",
    "FCN",
    "ModelFactory",
    "create_model",
]

# Version info
__version__ = "1.0.0"

# Model registry for easy access
AVAILABLE_MODELS = {
    "deeplabv3": DeepLabV3Plus,
    "fcn": FCN,
}


def list_models():
    """List all available models"""
    ModelFactory.print_available_models()


def get_model_comparison():
    """Get comparison of all models"""
    return ModelFactory.compare_models()

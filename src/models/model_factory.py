"""Model factory for creating segmentation models."""

from typing import Dict, Any
from .base_model import BaseSegmentationModel
from .deeplabv3plus import DeepLabV3Plus
from .deeplabv3 import DeepLabV3ResNet50
from .pspnet import PSPNet


class ModelFactory:
    """Factory class for creating segmentation models."""

    _models = {
        "deeplabv3plus": DeepLabV3Plus,
        "deeplabv3_resnet50": DeepLabV3ResNet50,
        "pspnet": PSPNet,
    }

    @classmethod
    def create_model(
        cls, model_name: str, num_classes: int = 7, **kwargs
    ) -> BaseSegmentationModel:
        """Create a segmentation model by name."""
        model_name_lower = model_name.lower()

        if model_name_lower not in cls._models:
            available_models = list(cls._models.keys())
            raise ValueError(
                f"Unknown model '{model_name}'. Available models: {available_models}"
            )

        model_class = cls._models[model_name_lower]
        return model_class(num_classes=num_classes, **kwargs)

    @classmethod
    def get_available_models(cls) -> Dict[str, str]:
        """Get dictionary of available models with descriptions."""
        return {
            "deeplabv3plus": "DeepLabV3Plus with MobileNetV2 backbone and encoder-decoder architecture (VainF)",
            "deeplabv3_resnet50": "DeepLabV3 with ResNet50 backbone (PyTorch Built-in, COCO pretrained)",
            "pspnet": "PSPNet with ResNet50 backbone and Pyramid Scene Parsing (segmentation_models_pytorch)",
        }

    @classmethod
    def register_model(cls, name: str, model_class: type):
        """Register a new model class."""
        if not issubclass(model_class, BaseSegmentationModel):
            raise ValueError(
                f"Model class must inherit from BaseSegmentationModel, "
                f"got {model_class.__name__}"
            )

        cls._models[name.lower()] = model_class

    @classmethod
    def create_model_from_config(cls, config) -> BaseSegmentationModel:
        """Create model from configuration object."""
        # Handle both dict and object-like config access
        if hasattr(config, "model"):
            # Object-like access
            model_config = config.model
            data_config = config.data
            backbone_name = getattr(model_config, "backbone", "mobilenet")
            model_name = model_config.name
            num_classes = data_config.num_classes
        else:
            # Dictionary access
            model_config = config["model"]
            data_config = config["data"]
            backbone_name = model_config.get("backbone", "mobilenet")
            model_name = model_config["name"]
            num_classes = data_config["num_classes"]

        return cls.create_model(
            model_name=model_name, num_classes=num_classes, backbone_name=backbone_name
        )

    @classmethod
    def get_model_info(cls, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model."""
        model_name_lower = model_name.lower()

        if model_name_lower not in cls._models:
            raise ValueError(f"Unknown model '{model_name}'")

        model_class = cls._models[model_name_lower]
        temp_model = model_class(num_classes=7)
        total_params, trainable_params = temp_model.count_parameters()

        return {
            "name": temp_model.get_model_name(),
            "class": model_class.__name__,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "supports_aux_loss": hasattr(temp_model, "aux_classifier"),
            "pretrained": True,
        }

    @classmethod
    def compare_models(cls, model_names: list = None) -> Dict[str, Dict[str, Any]]:
        """Compare multiple models side by side."""
        if model_names is None:
            model_names = ["deeplabv3plus", "deeplabv3_resnet50", "pspnet"]

        comparison = {}
        for model_name in model_names:
            try:
                comparison[model_name] = cls.get_model_info(model_name)
            except ValueError as e:
                print(f"Warning: {e}")
                continue

        return comparison

    @classmethod
    def print_available_models(cls):
        """Print all available models with descriptions."""
        print("\n" + "=" * 60)
        print("AVAILABLE SEGMENTATION MODELS")
        print("=" * 60)

        descriptions = cls.get_available_models()

        for name, description in descriptions.items():
            print(f"\n🔹 {name.upper()}")
            print(f"   {description}")

            try:
                info = cls.get_model_info(name)
                print(f"   Parameters: {info['total_parameters']:,}")
                print(f"   Aux Loss: {'Yes' if info['supports_aux_loss'] else 'No'}")
            except:
                pass

        print("\n" + "=" * 60)


def create_model(
    model_name: str, num_classes: int = 7, **kwargs
) -> BaseSegmentationModel:
    """Convenience function to create a model."""
    return ModelFactory.create_model(model_name, num_classes, **kwargs)

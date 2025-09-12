"""
Model factory for creating segmentation models.
Follows Factory pattern and Open/Closed principle - open for extension, closed for modification.
"""

from typing import Dict, Any
from .base_model import BaseSegmentationModel
from .deeplabv3 import DeepLabV3Plus
from .fcn import FCN


class ModelFactory:
    """
    Factory class for creating segmentation models.

    Makes it easy to:
    - Switch between models with configuration
    - Add new models without changing existing code
    - Maintain consistent interface across models
    """

    # Registry of available models
    _models = {
        "deeplabv3": DeepLabV3Plus,
        "deeplabv3_plus": DeepLabV3Plus,  # Alias
        "deeplab": DeepLabV3Plus,  # Alias
        "fcn": FCN,
        "fcn_resnet50": FCN,  # Alias
    }

    @classmethod
    def create_model(
        cls, model_name: str, num_classes: int = 7, **kwargs
    ) -> BaseSegmentationModel:
        """
        Create a segmentation model by name.

        Args:
            model_name: Name of the model ('deeplabv3', 'fcn', etc.)
            num_classes: Number of segmentation classes
            **kwargs: Additional model-specific arguments

        Returns:
            Instantiated model inheriting from BaseSegmentationModel

        Raises:
            ValueError: If model_name is not recognized
        """
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
        """
        Get dictionary of available models with descriptions.

        Returns:
            Dictionary mapping model names to descriptions
        """
        descriptions = {
            "deeplabv3": "DeepLabV3+ with ResNet-50 backbone (torchvision pre-trained)",
            "fcn": "Fully Convolutional Network with ResNet-50 backbone (torchvision pre-trained)",
        }

        return descriptions

    @classmethod
    def register_model(cls, name: str, model_class: type):
        """
        Register a new model class.
        Allows extending the factory without modifying existing code.

        Args:
            name: Name to register the model under
            model_class: Model class inheriting from BaseSegmentationModel
        """
        if not issubclass(model_class, BaseSegmentationModel):
            raise ValueError(
                f"Model class must inherit from BaseSegmentationModel, "
                f"got {model_class.__name__}"
            )

        cls._models[name.lower()] = model_class

    @classmethod
    def create_model_from_config(cls, config) -> BaseSegmentationModel:
        """
        Create model from configuration object.

        Args:
            config: Configuration object with model settings

        Returns:
            Instantiated model
        """
        return cls.create_model(
            model_name=config.model.model_name, num_classes=config.data.num_classes
        )

    @classmethod
    def get_model_info(cls, model_name: str) -> Dict[str, Any]:
        """
        Get information about a specific model.

        Args:
            model_name: Name of the model

        Returns:
            Dictionary with model information
        """
        model_name_lower = model_name.lower()

        if model_name_lower not in cls._models:
            raise ValueError(f"Unknown model '{model_name}'")

        model_class = cls._models[model_name_lower]

        # Create a temporary instance to get info
        temp_model = model_class(num_classes=7)
        total_params, trainable_params = temp_model.count_parameters()

        info = {
            "name": temp_model.get_model_name(),
            "class": model_class.__name__,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "supports_aux_loss": hasattr(temp_model, "aux_classifier"),
            "pretrained": True,  # All our models use pre-trained weights
        }

        return info

    @classmethod
    def compare_models(cls, model_names: list = None) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple models side by side.

        Args:
            model_names: List of model names to compare. If None, compare all.

        Returns:
            Dictionary with comparison information
        """
        if model_names is None:
            model_names = list(set(cls._models.values()))  # Unique model classes
            model_names = ["deeplabv3", "fcn"]  # Use aliases for clean comparison

        comparison = {}

        for model_name in model_names:
            try:
                comparison[model_name] = cls.get_model_info(model_name)
            except ValueError as e:
                print(f"Warning: {e}")

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

            # Print model info
            try:
                info = cls.get_model_info(name)
                print(f"   Parameters: {info['total_parameters']:,}")
                print(f"   Aux Loss: {'Yes' if info['supports_aux_loss'] else 'No'}")
            except:
                pass

        print("\n" + "=" * 60)


# Convenience function for easy model creation
def create_model(
    model_name: str, num_classes: int = 7, **kwargs
) -> BaseSegmentationModel:
    """
    Convenience function to create a model.

    Args:
        model_name: Name of the model
        num_classes: Number of classes
        **kwargs: Additional arguments

    Returns:
        Instantiated model
    """
    return ModelFactory.create_model(model_name, num_classes, **kwargs)

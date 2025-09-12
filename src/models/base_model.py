"""
Abstract base class for semantic segmentation models.
Follows Interface Segregation and Dependency Inversion principles.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
import torch
import torch.nn as nn


class BaseSegmentationModel(ABC, nn.Module):
    """
    Abstract base class for all segmentation models.
    Enforces consistent interface across different architectures.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            Segmentation logits of shape (B, num_classes, H, W)
        """
        pass

    @abstractmethod
    def get_param_groups(self, backbone_lr: float, classifier_lr: float) -> List[Dict]:
        """
        Get parameter groups for different learning rates.
        Backbone typically gets lower LR, classifier gets higher LR.

        Args:
            backbone_lr: Learning rate for backbone/encoder
            classifier_lr: Learning rate for classifier/decoder

        Returns:
            List of parameter groups for optimizer
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Get human-readable model name"""
        pass

    def count_parameters(self) -> Tuple[int, int]:
        """
        Count total and trainable parameters.

        Returns:
            Tuple of (total_params, trainable_params)
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def freeze_backbone(self):
        """Freeze backbone parameters for transfer learning"""
        backbone = self.get_backbone()
        if backbone is not None:
            for param in backbone.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters for fine-tuning"""
        backbone = self.get_backbone()
        if backbone is not None:
            for param in backbone.parameters():
                param.requires_grad = True

    @abstractmethod
    def get_backbone(self) -> nn.Module:
        """Get the backbone/encoder module for freezing/unfreezing"""
        pass

    def print_model_info(self):
        """Print model information"""
        total_params, trainable_params = self.count_parameters()
        print(f"\n{self.get_model_name()} Model Info:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Number of classes: {self.num_classes}")

        # Check if backbone is frozen
        backbone = self.get_backbone()
        if backbone is not None:
            backbone_trainable = sum(
                p.numel() for p in backbone.parameters() if p.requires_grad
            )
            backbone_total = sum(p.numel() for p in backbone.parameters())
            backbone_frozen = backbone_trainable == 0
            print(f"  Backbone status: {'Frozen' if backbone_frozen else 'Trainable'}")
            print(f"  Backbone parameters: {backbone_total:,}")

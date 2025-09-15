"""PyTorch built-in DeepLabV3 ResNet50 model wrapper."""

from typing import Dict, List
import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50
from .base_model import BaseSegmentationModel


class DeepLabV3ResNet50(BaseSegmentationModel):
    """DeepLabV3 with ResNet50 backbone using PyTorch's built-in implementation."""

    def __init__(self, num_classes: int = 7, pretrained: bool = True, **kwargs):
        super().__init__(num_classes)

        self.model = deeplabv3_resnet50(
            pretrained=pretrained,
            num_classes=21 if pretrained else num_classes,
        )

        if pretrained and num_classes != 21:

            in_channels = self.model.classifier[-1].in_channels
            self.model.classifier[-1] = nn.Conv2d(
                in_channels, num_classes, kernel_size=1
            )

            if (
                hasattr(self.model, "aux_classifier")
                and self.model.aux_classifier is not None
            ):
                aux_in_channels = self.model.aux_classifier[-1].in_channels
                self.model.aux_classifier[-1] = nn.Conv2d(
                    aux_in_channels, num_classes, kernel_size=1
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through DeepLabV3."""

        output = self.model(x)
        if isinstance(output, dict):

            return output
        return output

    def get_param_groups(self, backbone_lr: float, classifier_lr: float) -> List[Dict]:
        """Get parameter groups for differential learning rates."""
        backbone_params = []
        classifier_params = []

        for name, param in self.model.named_parameters():
            if "classifier" in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)

        return [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": classifier_params, "lr": classifier_lr},
        ]

    def get_model_name(self) -> str:
        """Get human-readable model name."""
        return "DeepLabV3 ResNet50 (PyTorch Built-in)"

    def get_backbone(self) -> nn.Module:
        """Get the backbone/encoder module."""
        return self.model.backbone

    def freeze_backbone(self):
        """Freeze backbone for transfer learning."""
        for param in self.model.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone for fine-tuning."""
        for param in self.model.backbone.parameters():
            param.requires_grad = True

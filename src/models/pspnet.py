"""PSPNet implementation using segmentation_models_pytorch."""

from typing import Dict, List
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from .base_model import BaseSegmentationModel


class PSPNet(BaseSegmentationModel):
    """PSPNet with ResNet50 backbone using segmentation_models_pytorch."""

    def __init__(
        self,
        num_classes: int = 7,
        encoder_name: str = "resnet50",
        encoder_weights: str = "imagenet",
        **kwargs,
    ):
        super().__init__(num_classes)

        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights

       
        self.model = smp.PSPNet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=num_classes,
            activation=None,  
            in_channels=3,
        )

        
        self.encoder = self.model.encoder
        self.decoder = self.model.decoder
        self.segmentation_head = self.model.segmentation_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through PSPNet."""
        
        output = self.model(x)
        return {"out": output} 

    def get_param_groups(self, backbone_lr: float, classifier_lr: float) -> List[Dict]:
        """Get parameter groups for differential learning rates."""
        backbone_params = []
        classifier_params = []

        
        for param in self.encoder.parameters():
            backbone_params.append(param)

        
        for param in self.decoder.parameters():
            classifier_params.append(param)
        for param in self.segmentation_head.parameters():
            classifier_params.append(param)

        return [
            {"params": backbone_params, "lr": backbone_lr, "name": "encoder"},
            {"params": classifier_params, "lr": classifier_lr, "name": "decoder_head"},
        ]

    def get_backbone(self) -> nn.Module:
        """Get the encoder/backbone module."""
        return self.encoder

    def get_model_name(self) -> str:
        """Get human-readable model name."""
        return f"PSPNet {self.encoder_name.upper()} (segmentation_models_pytorch)"

    def count_parameters(self):
        """Count total and trainable parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params

    def freeze_backbone(self):
        """Freeze encoder parameters for transfer learning."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze encoder parameters for fine-tuning."""
        for param in self.encoder.parameters():
            param.requires_grad = True

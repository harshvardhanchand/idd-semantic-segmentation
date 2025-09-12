"""
DeepLabV3+ implementation using torchvision pre-trained models.
Optimized for IDD semantic segmentation with full model weights.
"""

from typing import Dict, List
import torch
import torch.nn as nn
from torchvision.models import segmentation
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights

from .base_model import BaseSegmentationModel


class DeepLabV3Plus(BaseSegmentationModel):
    """
    DeepLabV3+ with ResNet-50 backbone using torchvision pre-trained weights.

    Key features:
    - Full model pre-trained weights (not just backbone)
    - Custom classifier head for 7 IDD classes
    - Differential learning rates (backbone 1x, classifier 10x)
    - End-to-end fine-tuning capability
    """

    def __init__(self, num_classes: int = 7):
        super().__init__(num_classes)

        # Load pre-trained DeepLabV3 with full model weights
        self.model = segmentation.deeplabv3_resnet50(
            weights=DeepLabV3_ResNet50_Weights.DEFAULT,
            weights_backbone=None,  # Use full model weights, not separate backbone
            num_classes=21,  # Original COCO/VOC classes
            aux_loss=True,  # Auxiliary classifier for better training
        )

        # Replace final classifier for IDD classes (7 classes)
        self._replace_classifier()

        # Store references to backbone and classifier for param grouping
        self.backbone = self.model.backbone
        self.classifier = self.model.classifier
        self.aux_classifier = (
            self.model.aux_classifier if hasattr(self.model, "aux_classifier") else None
        )

    def _replace_classifier(self):
        """Replace the final classifier layer for IDD classes"""
        # Main classifier
        in_channels = self.model.classifier[-1].in_channels
        self.model.classifier[-1] = nn.Conv2d(
            in_channels, self.num_classes, kernel_size=1
        )

        # Auxiliary classifier (if exists)
        if (
            hasattr(self.model, "aux_classifier")
            and self.model.aux_classifier is not None
        ):
            aux_in_channels = self.model.aux_classifier[-1].in_channels
            self.model.aux_classifier[-1] = nn.Conv2d(
                aux_in_channels, self.num_classes, kernel_size=1
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through DeepLabV3+

        Args:
            x: Input tensor (B, 3, H, W)

        Returns:
            Segmentation logits (B, num_classes, H, W)
        """
        # During training, model returns dict with 'out' and 'aux'
        # During eval, we only want the main output
        output = self.model(x)

        if isinstance(output, dict):
            return output["out"]  # Main segmentation output
        else:
            return output

    def forward_with_aux(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning both main and auxiliary outputs.
        Useful for training with auxiliary loss.
        """
        return self.model(x)

    def get_param_groups(self, backbone_lr: float, classifier_lr: float) -> List[Dict]:
        """
        Create parameter groups for differential learning rates.

        Args:
            backbone_lr: Learning rate for backbone (e.g., 0.001)
            classifier_lr: Learning rate for classifier (e.g., 0.01)

        Returns:
            List of parameter groups for optimizer
        """
        param_groups = []

        # Backbone parameters (lower learning rate)
        backbone_params = []
        for name, param in self.named_parameters():
            if "backbone" in name and param.requires_grad:
                backbone_params.append(param)

        if backbone_params:
            param_groups.append(
                {"params": backbone_params, "lr": backbone_lr, "name": "backbone"}
            )

        # Classifier parameters (higher learning rate)
        classifier_params = []
        for name, param in self.named_parameters():
            if "classifier" in name and param.requires_grad:
                classifier_params.append(param)

        if classifier_params:
            param_groups.append(
                {"params": classifier_params, "lr": classifier_lr, "name": "classifier"}
            )

        # Any other parameters (use classifier LR)
        other_params = []
        for name, param in self.named_parameters():
            if (
                "backbone" not in name
                and "classifier" not in name
                and param.requires_grad
            ):
                other_params.append(param)

        if other_params:
            param_groups.append(
                {"params": other_params, "lr": classifier_lr, "name": "other"}
            )

        return param_groups

    def get_backbone(self) -> nn.Module:
        """Get backbone module for freezing/unfreezing"""
        return self.backbone

    def get_model_name(self) -> str:
        """Get human-readable model name"""
        return "DeepLabV3+ ResNet-50"

    def enable_aux_loss(self):
        """Enable auxiliary loss during training"""
        if hasattr(self.model, "aux_classifier"):
            self.model.aux_classifier.training = True

    def disable_aux_loss(self):
        """Disable auxiliary loss during evaluation"""
        if hasattr(self.model, "aux_classifier"):
            self.model.aux_classifier.training = False

    def get_feature_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract intermediate feature maps for visualization/analysis.
        Useful for understanding what the model learned.
        """
        features = {}

        # Hook function to capture features
        def hook_fn(name):
            def hook(module, input, output):
                features[name] = output.detach()

            return hook

        # Register hooks
        hooks = []
        hooks.append(self.backbone.layer1.register_forward_hook(hook_fn("layer1")))
        hooks.append(self.backbone.layer2.register_forward_hook(hook_fn("layer2")))
        hooks.append(self.backbone.layer3.register_forward_hook(hook_fn("layer3")))
        hooks.append(self.backbone.layer4.register_forward_hook(hook_fn("layer4")))

        # Forward pass
        with torch.no_grad():
            _ = self.forward(x)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return features

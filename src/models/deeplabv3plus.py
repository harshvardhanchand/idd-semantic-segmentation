"""DeepLabV3Plus implementation using VainF's proven architecture."""

from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

from .base_model import BaseSegmentationModel


class IntermediateLayerGetter(nn.ModuleDict):
    """Module wrapper that returns intermediate layers from a model."""

    def __init__(self, model, return_layers):
        if not set(return_layers).issubset(
            [name for name, _ in model.named_children()]
        ):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class ASPPConv(nn.Sequential):
    """ASPP Convolution module."""

    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    """ASPP Pooling module."""

    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling module."""

    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        )

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class DeepLabHeadV3Plus(nn.Module):
    """DeepLabV3Plus decoder head with skip connections."""

    def __init__(
        self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]
    ):
        super(DeepLabHeadV3Plus, self).__init__()

        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(in_channels, aspp_dilate)

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1),
        )
        self._init_weight()

    def forward(self, feature):
        low_level_feature = self.project(feature["low_level"])
        output_feature = self.aspp(feature["out"])
        output_feature = F.interpolate(
            output_feature,
            size=low_level_feature.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.classifier(torch.cat([low_level_feature, output_feature], dim=1))

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class VainFDeepLabV3(nn.Module):
    """VainF's DeepLabV3 base model structure."""

    def __init__(self, backbone, classifier):
        super(VainFDeepLabV3, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x = self.classifier(features)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        return x


class DeepLabV3Plus(BaseSegmentationModel):
    """DeepLabV3Plus using VainF's exact implementation with configurable backbone."""

    def __init__(
        self,
        num_classes: int = 7,
        output_stride: int = 16,
        pretrained_backbone: bool = True,
        backbone_name: str = "mobilenet",
    ):
        super().__init__(num_classes)
        self.backbone_name = backbone_name.lower()

        if output_stride == 8:
            aspp_dilate = [12, 24, 36]
        else:
            aspp_dilate = [6, 12, 18]

        if self.backbone_name != "mobilenet":
            raise ValueError(
                f"Only MobileNet backbone is supported, got: {backbone_name}"
            )

        backbone = mobilenet_v2(
            weights=MobileNet_V2_Weights.DEFAULT if pretrained_backbone else None
        ).features

        inplanes = 320
        low_level_planes = 24
        return_layers = {
            "17": "out",
            "3": "low_level",
        }

        backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

        classifier = DeepLabHeadV3Plus(
            inplanes, low_level_planes, num_classes, aspp_dilate
        )

        self.model = VainFDeepLabV3(backbone, classifier)

        self.backbone = self.model.backbone
        self.classifier = self.model.classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        return {"out": output}

    def get_param_groups(self, backbone_lr: float, classifier_lr: float) -> List[Dict]:
        """Get parameter groups for differential learning rates."""
        backbone_params = list(self.backbone.parameters())
        classifier_params = list(self.classifier.parameters())

        return [
            {"params": backbone_params, "lr": backbone_lr, "name": "backbone"},
            {"params": classifier_params, "lr": classifier_lr, "name": "classifier"},
        ]

    def get_backbone(self) -> nn.Module:
        """Get backbone module."""
        return self.backbone

    def get_model_name(self) -> str:
        return "DeepLabV3Plus MobileNetV2"

    def count_parameters(self):
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True

"""Loss functions for semantic segmentation."""

import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
import segmentation_models_pytorch as smp


class SegmentationLoss(nn.Module):
    """Enhanced segmentation loss with CrossEntropy + Lovász-Softmax."""

    def __init__(
        self,
        ignore_index: int = 255,
        aux_weight: float = 0.0,
        class_weights: torch.Tensor = None,
        lovasz_weight: float = 0.0,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.aux_weight = aux_weight
        self.lovasz_weight = lovasz_weight

        self.main_loss = nn.CrossEntropyLoss(
            ignore_index=ignore_index, weight=class_weights
        )

        if lovasz_weight > 0:
            self.lovasz_loss = smp.losses.LovaszLoss(
                mode="multiclass",
                ignore_index=ignore_index,
                per_image=False,
                from_logits=True,
            )

        if aux_weight > 0:
            self.aux_loss = nn.CrossEntropyLoss(
                ignore_index=ignore_index, weight=class_weights
            )

    def forward(self, outputs, targets):
        if isinstance(outputs, dict):
            main_output = outputs["out"]

            ce_loss = self.main_loss(main_output, targets)
            total_loss = ce_loss

            # Add Lovász-Softmax loss
            if self.lovasz_weight > 0:
                lovasz_loss = self.lovasz_loss(main_output, targets)
                total_loss = total_loss + self.lovasz_weight * lovasz_loss

            if "aux" in outputs and self.aux_weight > 0:
                aux_output = outputs["aux"]
                aux_ce_loss = self.aux_loss(aux_output, targets)
                total_loss = total_loss + self.aux_weight * aux_ce_loss

                if self.lovasz_weight > 0:
                    aux_lovasz_loss = self.lovasz_loss(aux_output, targets)
                    total_loss = (
                        total_loss
                        + self.aux_weight * self.lovasz_weight * aux_lovasz_loss
                    )

            return total_loss
        else:

            ce_loss = self.main_loss(outputs, targets)

            if self.lovasz_weight > 0:
                lovasz_loss = self.lovasz_loss(outputs, targets)
                return ce_loss + self.lovasz_weight * lovasz_loss

            return ce_loss


def compute_class_weights(
    loader, num_classes: int, ignore_index: int = 255
) -> torch.Tensor:
    """Compute balanced class weights using sklearn."""
    print("Collecting labels for class weight computation...")

    all_labels = []
    for batch_idx, (_, targets) in enumerate(loader):
        if batch_idx % 50 == 0:
            print(f"Processing batch {batch_idx + 1}/{len(loader)}")

        valid_labels = targets[targets != ignore_index].flatten()
        all_labels.extend(valid_labels.cpu().numpy())

    all_labels = np.array(all_labels)
    classes = np.arange(num_classes)

    weights = compute_class_weight("balanced", classes=classes, y=all_labels)

    return torch.FloatTensor(weights)

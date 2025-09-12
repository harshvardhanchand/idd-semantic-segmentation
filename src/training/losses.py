"""Loss functions for semantic segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationLoss(nn.Module):
    """Combined segmentation loss with auxiliary loss support."""

    def __init__(
        self,
        ignore_index: int = 255,
        aux_weight: float = 0.4,
        class_weights: torch.Tensor = None,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.aux_weight = aux_weight

        self.main_loss = nn.CrossEntropyLoss(
            ignore_index=ignore_index, weight=class_weights
        )

        if aux_weight > 0:
            self.aux_loss = nn.CrossEntropyLoss(
                ignore_index=ignore_index, weight=class_weights
            )

    def forward(self, outputs, targets):
        """Compute segmentation loss."""
        if isinstance(outputs, dict):
            # Model returns dict with 'out' and 'aux'
            main_output = outputs["out"]
            loss = self.main_loss(main_output, targets)

            if "aux" in outputs and self.aux_weight > 0:
                aux_output = outputs["aux"]
                aux_loss_val = self.aux_loss(aux_output, targets)
                loss = loss + self.aux_weight * aux_loss_val

            return loss
        else:
            # Model returns tensor directly
            return self.main_loss(outputs, targets)


def compute_class_weights(
    loader, num_classes: int, ignore_index: int = 255
) -> torch.Tensor:
    """Compute class weights from dataset for balanced training."""
    class_counts = torch.zeros(num_classes, dtype=torch.float)

    for _, targets in loader:
        for cls in range(num_classes):
            mask = targets == cls
            class_counts[cls] += mask.sum().item()

    # Inverse frequency weighting
    total_pixels = class_counts.sum()
    class_weights = total_pixels / (num_classes * class_counts + 1e-8)

    return class_weights

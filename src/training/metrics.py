"""Metrics for semantic segmentation."""

import torch
import numpy as np
from typing import Tuple, Dict, List


class SegmentationMetrics:
    """Compute segmentation metrics including mIoU."""

    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        """Reset metrics state."""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Update metrics with batch predictions."""
        # Get predictions from logits
        if predictions.dim() == 4:  # (B, C, H, W)
            predictions = predictions.argmax(dim=1)  # (B, H, W)

        predictions = predictions.cpu().numpy().flatten()
        targets = targets.cpu().numpy().flatten()

        # Mask out ignore index
        valid_mask = targets != self.ignore_index
        predictions = predictions[valid_mask]
        targets = targets[valid_mask]

        # Update confusion matrix
        for pred, target in zip(predictions, targets):
            if 0 <= pred < self.num_classes and 0 <= target < self.num_classes:
                self.confusion_matrix[target, pred] += 1

    def compute_iou(self) -> Tuple[np.ndarray, float]:
        """Compute per-class IoU and mean IoU."""
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusion_matrix)  # TP
        union = (
            self.confusion_matrix.sum(axis=1)  # TP + FN
            + self.confusion_matrix.sum(axis=0)  # TP + FP
            - intersection  # Remove double-counted TP
        )

        # Avoid division by zero
        iou_per_class = intersection / np.maximum(union, 1e-8)

        # Mean IoU (excluding classes with no samples)
        valid_classes = union > 0
        if valid_classes.sum() > 0:
            mean_iou = iou_per_class[valid_classes].mean()
        else:
            mean_iou = 0.0

        return iou_per_class, mean_iou

    def compute_pixel_accuracy(self) -> float:
        """Compute overall pixel accuracy."""
        correct = np.diag(self.confusion_matrix).sum()
        total = self.confusion_matrix.sum()
        return correct / np.maximum(total, 1e-8)

    def compute_mean_accuracy(self) -> float:
        """Compute mean per-class accuracy."""
        per_class_acc = np.diag(self.confusion_matrix) / np.maximum(
            self.confusion_matrix.sum(axis=1), 1e-8
        )
        return per_class_acc.mean()

    def get_results(self) -> Dict[str, float]:
        """Get all computed metrics."""
        iou_per_class, mean_iou = self.compute_iou()
        pixel_acc = self.compute_pixel_accuracy()
        mean_acc = self.compute_mean_accuracy()

        results = {
            "mIoU": mean_iou,
            "pixel_accuracy": pixel_acc,
            "mean_accuracy": mean_acc,
        }

        # Add per-class IoU
        for i, iou in enumerate(iou_per_class):
            results[f"IoU_class_{i}"] = iou

        return results

    def print_results(self, class_names: List[str] = None):
        """Print detailed results."""
        iou_per_class, mean_iou = self.compute_iou()
        pixel_acc = self.compute_pixel_accuracy()
        mean_acc = self.compute_mean_accuracy()

        print(f"\nSegmentation Metrics:")
        print(f"mIoU: {mean_iou:.4f}")
        print(f"Pixel Accuracy: {pixel_acc:.4f}")
        print(f"Mean Accuracy: {mean_acc:.4f}")

        print(f"\nPer-class IoU:")
        for i, iou in enumerate(iou_per_class):
            class_name = class_names[i] if class_names else f"Class {i}"
            print(f"  {class_name}: {iou:.4f}")


def evaluate_model(
    model, dataloader, device, num_classes: int, ignore_index: int = 255
) -> Dict[str, float]:
    """Evaluate model on dataloader."""
    model.eval()
    metrics = SegmentationMetrics(num_classes, ignore_index)

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            metrics.update(outputs, targets)

    return metrics.get_results()

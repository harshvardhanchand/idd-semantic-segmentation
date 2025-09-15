"""Modern metrics for semantic segmentation using TorchMetrics."""

import torch
from typing import Dict, List, Optional
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy


class SegmentationMetrics:

    def __init__(self, num_classes: int, ignore_index: int = 255, device: str = "cuda"):
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        self.metrics = MetricCollection(
            {
                "mIoU": MulticlassJaccardIndex(
                    num_classes=num_classes, ignore_index=ignore_index, average="macro"
                ),
                "pixel_accuracy": MulticlassAccuracy(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    average="micro",
                ),
                "mean_accuracy": MulticlassAccuracy(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    average="macro",
                ),
            }
        ).to(device)

        self.per_class_iou = MulticlassJaccardIndex(
            num_classes=num_classes,
            ignore_index=ignore_index,
            average=None,
        ).to(device)

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        

        if isinstance(predictions, dict):
            predictions = predictions["out"]

        if predictions.dim() == 4:
            predictions = predictions.argmax(dim=1)

        self.metrics.update(predictions, targets)
        self.per_class_iou.update(predictions, targets)

    def compute(self) -> Dict[str, float]:
        """Compute final metrics."""
        results = self.metrics.compute()
        per_class_ious = self.per_class_iou.compute()

        final_results = {
            "mIoU": float(results["mIoU"]),
            "pixel_accuracy": float(results["pixel_accuracy"]),
            "mean_accuracy": float(results["mean_accuracy"]),
        }

        for i, iou in enumerate(per_class_ious):
            final_results[f"IoU_class_{i}"] = float(iou)

        return final_results

    def reset(self):
        """Reset all metrics."""
        self.metrics.reset()
        self.per_class_iou.reset()

    def print_results(self, class_names: Optional[List[str]] = None):
        """Print comprehensive results."""
        results = self.compute()

        print(f" Segmentation Metrics:")
        print(f"  mIoU: {results['mIoU']:.4f}")
        print(f"  Pixel Accuracy: {results['pixel_accuracy']:.4f}")
        print(f"  Mean Accuracy: {results['mean_accuracy']:.4f}")

        print(f" Per-class IoU:")
        for i in range(self.num_classes):
            class_name = class_names[i] if class_names else f"Class {i}"
            iou_value = results[f"IoU_class_{i}"]
            print(f"  {class_name}: {iou_value:.4f}")


def evaluate_model(
    model, dataloader, device: str, num_classes: int, ignore_index: int = 255
) -> Dict[str, float]:
    
    model.eval()

    metrics = SegmentationMetrics(num_classes, ignore_index, device)

    print(f" Evaluating model on {len(dataloader)} batches...")

    with torch.no_grad():
        for i, (images, targets) in enumerate(dataloader):
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)

            metrics.update(outputs, targets)

            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(dataloader)} batches")

    print(" Evaluation complete!")
    return metrics.compute()

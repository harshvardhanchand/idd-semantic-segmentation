#!/usr/bin/env python3

import numpy as np
import yaml
from sklearn.utils.class_weight import compute_class_weight
from src.data.dataset import create_dataloaders
from src.utils.config import load_config


def apply_minmax_scaling(weights, target_min=0.6, target_max=5.0):
    """MinMax scale weights to [target_min, target_max], then normalize to mean=1."""
    w_min, w_max = weights.min(), weights.max()

    scaled = (weights - w_min) / (w_max - w_min) * (
        target_max - target_min
    ) + target_min

    return scaled / scaled.mean()


def main():

    config = load_config("src/configs/base.yaml")
    config["data"]["num_workers"] = 0
    config["data"]["batch_size"] = 8

    train_loader, _ = create_dataloaders(config)

    print(f"Collecting labels from {len(train_loader)} batches...")
    all_labels = []
    for batch_idx, (_, targets) in enumerate(train_loader):
        if batch_idx % 50 == 0:
            print(f"Batch {batch_idx + 1}/{len(train_loader)}")
        valid_labels = targets[targets != 255].flatten().cpu().numpy()
        all_labels.extend(valid_labels)

    all_labels = np.array(all_labels)

    print("Computing class weights...")
    original_weights = compute_class_weight(
        "balanced", classes=np.arange(7), y=all_labels
    )

    scaled_weights = apply_minmax_scaling(
        original_weights, target_min=0.6, target_max=5.0
    )

    weights_data = {
        "class_weights": scaled_weights.tolist(),
        "method": "minmax_scaled_balanced",
        "target_range": [0.6, 5.0],
        "dataset": "idd20k_lite",
    }

    with open("idd_class_weights.yaml", "w") as f:
        yaml.dump(weights_data, f)

    print("\nClass | Original | Scaled | Change")
    print("-" * 40)
    for i in range(7):
        orig = original_weights[i]
        scaled = scaled_weights[i]
        change = " Stable" if orig > 5.0 else "✓ Balanced"
        print(f"  {i}   |   {orig:5.2f}  | {scaled:5.2f} | {change}")

    print(
        f"\nRange: [{original_weights.min():.2f}, {original_weights.max():.2f}] → [{scaled_weights.min():.2f}, {scaled_weights.max():.2f}]"
    )
    print(" Saved to: idd_class_weights.yaml")


if __name__ == "__main__":
    main()

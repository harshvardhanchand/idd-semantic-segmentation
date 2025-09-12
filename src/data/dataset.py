"""IDD dataset loader with proper transforms."""

import os
from pathlib import Path
from typing import Tuple, Optional, Callable, List
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class IDDDataset(Dataset):
    """IDD semantic segmentation dataset."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        joint_transform: Optional[Callable] = None,
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform

        # Build file lists
        self.images, self.targets = self._make_dataset()

        if len(self.images) == 0:
            raise RuntimeError(f"Found 0 images in {self.root}/{split}/")

    def _make_dataset(self) -> Tuple[List[str], List[str]]:
        """Build list of image-target pairs."""
        images = []
        targets = []

        img_dir = self.root / "leftImg8bit" / self.split
        target_dir = self.root / "gtFine" / self.split

        if not img_dir.exists():
            raise RuntimeError(f"Image directory not found: {img_dir}")
        if not target_dir.exists():
            raise RuntimeError(f"Target directory not found: {target_dir}")

        # Iterate through subdirectories
        for subdir in sorted(img_dir.iterdir()):
            if not subdir.is_dir():
                continue

            target_subdir = target_dir / subdir.name
            if not target_subdir.exists():
                continue

            # Find image files
            for img_path in sorted(subdir.glob("*_image.jpg")):
                # Extract base name to find corresponding label
                base_name = img_path.stem.replace("_image", "")
                label_path = target_subdir / f"{base_name}_label.png"

                if label_path.exists():
                    images.append(str(img_path))
                    targets.append(str(label_path))

        return images, targets

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get image and target pair."""
        img_path = self.images[index]
        target_path = self.targets[index]

        # Load image and target
        image = Image.open(img_path).convert("RGB")
        target = Image.open(target_path)  # Keep as grayscale

        # Apply joint transforms (geometric transforms applied to both)
        if self.joint_transform is not None:
            image, target = self.joint_transform(image, target)

        # Apply individual transforms
        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self) -> int:
        return len(self.images)


def create_datasets(config: dict) -> Tuple[IDDDataset, IDDDataset]:
    """Create train and validation datasets."""
    from .transforms import get_transforms

    data_config = config["data"]
    aug_config = config["augmentation"]

    # Get transforms
    train_transforms = get_transforms(
        image_size=data_config["image_size"], is_training=True, **aug_config
    )

    val_transforms = get_transforms(
        image_size=data_config["image_size"], is_training=False, **aug_config
    )

    # Create datasets
    train_dataset = IDDDataset(
        root=data_config["root"], split="train", **train_transforms
    )

    val_dataset = IDDDataset(root=data_config["root"], split="val", **val_transforms)

    return train_dataset, val_dataset


def create_dataloaders(
    config: dict,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train and validation data loaders."""
    train_dataset, val_dataset = create_datasets(config)

    data_config = config["data"]

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=data_config["batch_size"],
        shuffle=True,
        num_workers=data_config["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=data_config["batch_size"],
        shuffle=False,
        num_workers=data_config["num_workers"],
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader

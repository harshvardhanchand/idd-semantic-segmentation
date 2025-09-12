"""Transform utilities for semantic segmentation."""

import random
from typing import Tuple, Dict, Any
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageEnhance
import numpy as np


class JointResize:
    """Resize image and mask with appropriate interpolation."""

    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def __call__(
        self, image: Image.Image, target: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        # Bilinear for image, nearest for mask
        image = image.resize(self.size, Image.BILINEAR)
        target = target.resize(self.size, Image.NEAREST)
        return image, target


class JointRandomHorizontalFlip:
    """Random horizontal flip applied to both image and mask."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(
        self, image: Image.Image, target: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        if random.random() < self.p:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            target = target.transpose(Image.FLIP_LEFT_RIGHT)
        return image, target


class JointRandomRotation:
    """Random rotation applied to both image and mask."""

    def __init__(self, degrees: float):
        self.degrees = degrees

    def __call__(
        self, image: Image.Image, target: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        if self.degrees > 0:
            angle = random.uniform(-self.degrees, self.degrees)
            image = image.rotate(angle, Image.BILINEAR)
            target = target.rotate(angle, Image.NEAREST)
        return image, target


class JointRandomCrop:
    """Random crop applied to both image and mask."""

    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def __call__(
        self, image: Image.Image, target: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        w, h = image.size
        th, tw = self.size

        if w < tw or h < th:
            # Pad if needed
            pad_w = max(0, tw - w)
            pad_h = max(0, th - h)
            image = transforms.Pad(
                (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
            )(image)
            target = transforms.Pad(
                (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2),
                fill=255,
            )(target)
            w, h = image.size

        # Random crop
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)

        image = image.crop((j, i, j + tw, i + th))
        target = target.crop((j, i, j + tw, i + th))

        return image, target


class JointCompose:
    """Compose transforms for joint image-mask processing."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(
        self, image: Image.Image, target: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class TargetToTensor:
    """Convert target mask to tensor with proper dtype."""

    def __call__(self, target: Image.Image) -> torch.Tensor:
        target = np.array(target, dtype=np.int64)
        return torch.from_numpy(target).long()


def get_transforms(
    image_size: Tuple[int, int],
    is_training: bool,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    horizontal_flip: float = 0.5,
    rotation: float = 10,
    brightness: float = 0.1,
    contrast: float = 0.1,
    saturation: float = 0.1,
    **kwargs
) -> Dict[str, Any]:
    """Get image and target transforms."""

    joint_transforms = []
    image_transforms = []
    target_transforms = []

    if is_training:
        # Joint geometric transforms
        joint_transforms.extend(
            [
                JointRandomCrop(image_size),
                JointRandomHorizontalFlip(horizontal_flip),
            ]
        )

        if rotation > 0:
            joint_transforms.append(JointRandomRotation(rotation))

        # Image-only photometric transforms
        if brightness > 0 or contrast > 0 or saturation > 0:
            image_transforms.append(
                transforms.ColorJitter(
                    brightness=brightness, contrast=contrast, saturation=saturation
                )
            )
    else:
        # Validation: only resize
        joint_transforms.append(JointResize(image_size))

    # Common transforms
    image_transforms.extend(
        [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    )

    target_transforms.append(TargetToTensor())

    return {
        "joint_transform": JointCompose(joint_transforms) if joint_transforms else None,
        "transform": transforms.Compose(image_transforms) if image_transforms else None,
        "target_transform": (
            transforms.Compose(target_transforms) if target_transforms else None
        ),
    }

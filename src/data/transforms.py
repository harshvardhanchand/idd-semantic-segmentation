"""Transform utilities for semantic segmentation."""

import random
from typing import Tuple, Dict, Any
import torch
from torchvision import transforms
from PIL import Image, ImageFilter
import numpy as np


class JointRandomScale:
    """Randomly scale image & mask by factor in [scale_min, scale_max]."""

    def __init__(self, scale_min: float = 0.5, scale_max: float = 2.0):
        assert scale_max >= scale_min > 0
        self.scale_min = scale_min
        self.scale_max = scale_max

    def __call__(self, image: Image.Image, target: Image.Image):
        s = random.uniform(self.scale_min, self.scale_max)
        w, h = image.size
        nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
        image = image.resize((nw, nh), Image.BILINEAR)
        target = target.resize((nw, nh), Image.NEAREST)
        return image, target


class JointRandomCrop:
    """Random crop to size=(H,W). Use after padding to ensure sufficient size."""

    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def __call__(self, image: Image.Image, target: Image.Image):
        w, h = image.size
        th, tw = self.size
        assert h >= th and w >= tw, "Image smaller than crop size"
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        image = image.crop((j, i, j + tw, i + th))
        target = target.crop((j, i, j + tw, i + th))
        return image, target


class JointRandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: Image.Image, target: Image.Image):
        if random.random() < self.p:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            target = target.transpose(Image.FLIP_LEFT_RIGHT)
        return image, target


class JointRandomRotation:
    """Random rotation. Set degrees=0 to disable."""

    def __init__(self, degrees: float, ignore_index: int = 255):
        self.degrees = max(0.0, float(degrees))
        self.ignore_index = int(ignore_index)

    def __call__(self, image: Image.Image, target: Image.Image):
        if self.degrees <= 0:
            return image, target
        angle = random.uniform(-self.degrees, self.degrees)
        image = image.rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0))
        target = target.rotate(
            angle, resample=Image.NEAREST, fillcolor=self.ignore_index
        )
        return image, target


class JointCompose:
    def __init__(self, transforms_list):
        self.transforms = transforms_list or []

    def __call__(self, image: Image.Image, target: Image.Image):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class TargetToTensor:
    """Convert mask to Long tensor of class IDs."""

    def __init__(self, dtype=np.int64):
        self.dtype = dtype

    def __call__(self, target: Image.Image) -> torch.Tensor:
        target = np.array(target, dtype=self.dtype)
        return torch.from_numpy(target).long()


class RandomGaussianBlurPIL:
    """Optional Gaussian blur for images."""

    def __init__(
        self, p: float = 0.1, radius_min: float = 0.1, radius_max: float = 1.5
    ):
        self.p = p
        self.rmin, self.rmax = radius_min, radius_max

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            r = random.uniform(self.rmin, self.rmax)
            return img.filter(ImageFilter.GaussianBlur(radius=r))
        return img


class JointDirectResize:
    def __init__(self, size):
        self.target_h, self.target_w = size

    def __call__(self, image, target):

        image = image.resize((self.target_w, self.target_h), Image.BILINEAR)
        target = target.resize((self.target_w, self.target_h), Image.NEAREST)
        return image, target


def get_transforms(
    image_size: Tuple[int, int] = (224, 320),
    is_training: bool = True,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    horizontal_flip: float = 0.5,
    rotation: float = 0.0,
    scale_min: float = 0.5,
    scale_max: float = 2.0,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.2,
    blur_p: float = 0.10,
    ignore_index: int = 255,
) -> Dict[str, Any]:
    """
    Returns transform dict: joint_transform, transform, target_transform
    """
    joint_transforms = []
    image_transforms = []
    target_transforms = []
    target_h, target_w = image_size

    if is_training:

        safe_scale_min = max(scale_min, 1.0)
        safe_scale_max = max(scale_max, safe_scale_min + 1e-6)

        joint_transforms.extend(
            [
                JointRandomScale(scale_min=safe_scale_min, scale_max=safe_scale_max),
                JointRandomCrop((target_h, target_w)),
                JointRandomHorizontalFlip(horizontal_flip),
            ]
        )

        if rotation > 0:
            joint_transforms.append(
                JointRandomRotation(rotation, ignore_index=ignore_index)
            )

        if any(v > 0 for v in (brightness, contrast, saturation)):
            image_transforms.append(
                transforms.ColorJitter(
                    brightness=brightness, contrast=contrast, saturation=saturation
                )
            )
        if blur_p > 0:
            image_transforms.append(RandomGaussianBlurPIL(p=blur_p))
    else:
        joint_transforms.append(JointDirectResize((target_h, target_w)))

    image_transforms.extend(
        [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    )
    target_transforms.append(TargetToTensor())

    return {
        "joint_transform": JointCompose(joint_transforms),
        "transform": transforms.Compose(image_transforms),
        "target_transform": transforms.Compose(target_transforms),
    }

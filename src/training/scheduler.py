"""Learning rate schedulers for semantic segmentation."""

import torch
from torch.optim.lr_scheduler import _LRScheduler


class PolynomialLR(_LRScheduler):
    """Polynomial learning rate decay scheduler."""

    def __init__(
        self, optimizer, max_iter: int, power: float = 0.9, last_epoch: int = -1
    ):
        self.max_iter = max_iter
        self.power = power
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute current learning rate."""
        if self.last_epoch >= self.max_iter:
            return [0 for _ in self.base_lrs]

        factor = (1 - self.last_epoch / self.max_iter) ** self.power
        return [base_lr * factor for base_lr in self.base_lrs]


def create_scheduler(optimizer, config: dict, total_steps: int):
    """Create scheduler based on config."""
    scheduler_name = config["training"]["scheduler"].lower()

    if scheduler_name == "poly":
        power = config["training"].get("poly_power", 0.9)
        return PolynomialLR(optimizer, total_steps, power)

    elif scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)

    elif scheduler_name == "step":
        step_size = config["training"].get("step_size", total_steps // 3)
        gamma = config["training"].get("step_gamma", 0.1)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma)

    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

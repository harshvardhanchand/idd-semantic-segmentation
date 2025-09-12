"""Main trainer class for semantic segmentation."""

import os
import time
from pathlib import Path
from typing import Dict, Any
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from .losses import SegmentationLoss, compute_class_weights
from .metrics import SegmentationMetrics, evaluate_model
from .scheduler import create_scheduler


class SegmentationTrainer:
    """Main trainer for semantic segmentation models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        config: Dict[str, Any],
        device: torch.device,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Setup training components
        self._setup_optimizer()
        self._setup_loss()
        self._setup_scheduler()
        self._setup_amp()
        self._setup_checkpointing()

        # Training state
        self.current_epoch = 0
        self.best_miou = 0.0
        self.train_history = []
        self.val_history = []

    def _setup_optimizer(self):
        """Setup optimizer with differential learning rates."""
        training_config = self.config["training"]

        # Get parameter groups with differential LRs
        backbone_lr = training_config["lr"] * training_config["backbone_lr_multiplier"]
        classifier_lr = (
            training_config["lr"] * training_config["classifier_lr_multiplier"]
        )

        param_groups = self.model.get_param_groups(backbone_lr, classifier_lr)

        if training_config["optimizer"].lower() == "sgd":
            self.optimizer = torch.optim.SGD(
                param_groups,
                momentum=training_config["momentum"],
                weight_decay=training_config["weight_decay"],
            )
        elif training_config["optimizer"].lower() == "adam":
            self.optimizer = torch.optim.Adam(
                param_groups, weight_decay=training_config["weight_decay"]
            )
        else:
            raise ValueError(f"Unknown optimizer: {training_config['optimizer']}")

    def _setup_loss(self):
        """Setup loss function."""
        data_config = self.config["data"]
        training_config = self.config["training"]

        # Compute class weights if needed
        class_weights = None
        if training_config.get("class_weights", False):
            print("Computing class weights...")
            class_weights = compute_class_weights(
                self.train_loader,
                data_config["num_classes"],
                data_config["ignore_index"],
            ).to(self.device)
            print(f"Class weights: {class_weights}")

        self.criterion = SegmentationLoss(
            ignore_index=data_config["ignore_index"],
            aux_weight=training_config["aux_loss_weight"],
            class_weights=class_weights,
        )

    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        total_steps = len(self.train_loader) * self.config["training"]["epochs"]
        self.scheduler = create_scheduler(self.optimizer, self.config, total_steps)

    def _setup_amp(self):
        """Setup automatic mixed precision."""
        self.use_amp = self.config["training"].get("amp", True)
        if self.use_amp:
            self.scaler = GradScaler()

    def _setup_checkpointing(self):
        """Setup checkpoint directory."""
        exp_config = self.config["experiment"]
        self.save_dir = Path(exp_config["save_dir"]) / exp_config["name"]
        self.checkpoint_dir = self.save_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        running_loss = 0.0
        metrics = SegmentationMetrics(
            self.config["data"]["num_classes"], self.config["data"]["ignore_index"]
        )

        pbar = tqdm(self.train_loader, desc=f"Train Epoch {self.current_epoch}")

        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

            # Update metrics
            running_loss += loss.item()
            if isinstance(outputs, dict):
                metrics.update(outputs["out"], targets)
            else:
                metrics.update(outputs, targets)

            # Update progress bar
            if batch_idx % 50 == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Update scheduler (per-iteration for polynomial)
            if hasattr(self.scheduler, "step"):
                self.scheduler.step()

        # Compute epoch metrics
        results = metrics.get_results()
        results["loss"] = running_loss / len(self.train_loader)

        return results

    def validate(self) -> Dict[str, float]:
        """Validate model."""
        results = evaluate_model(
            self.model,
            self.val_loader,
            self.device,
            self.config["data"]["num_classes"],
            self.config["data"]["ignore_index"],
        )
        return results

    def save_checkpoint(self, is_best: bool = False, epoch: int = None):
        """Save model checkpoint."""
        state = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_miou": self.best_miou,
            "config": self.config,
            "train_history": self.train_history,
            "val_history": self.val_history,
        }

        if self.use_amp:
            state["scaler_state_dict"] = self.scaler.state_dict()

        # Save latest checkpoint
        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(state, latest_path)

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(state, best_path)
            print(f"New best model saved with mIoU: {self.best_miou:.4f}")

        # Save epoch checkpoint
        if epoch is not None:
            epoch_path = self.checkpoint_dir / f"epoch_{epoch:03d}.pt"
            torch.save(state, epoch_path)

    def train(self):
        """Main training loop."""
        print("Starting training...")
        print(f"Model: {self.model.get_model_name()}")
        print(f"Epochs: {self.config['training']['epochs']}")
        print(f"Device: {self.device}")
        print(f"AMP: {self.use_amp}")

        start_time = time.time()

        for epoch in range(self.config["training"]["epochs"]):
            self.current_epoch = epoch

            # Train
            train_results = self.train_epoch()
            self.train_history.append(train_results)

            # Validate
            val_every = self.config["training"]["val_every"]
            if epoch % val_every == 0 or epoch == self.config["training"]["epochs"] - 1:
                val_results = self.validate()
                self.val_history.append(val_results)

                # Check if best model
                current_miou = val_results["mIoU"]
                is_best = current_miou > self.best_miou
                if is_best:
                    self.best_miou = current_miou

                # Print results
                print(f"\nEpoch {epoch}")
                print(
                    f"Train - Loss: {train_results['loss']:.4f}, mIoU: {train_results['mIoU']:.4f}"
                )
                print(
                    f"Val   - mIoU: {val_results['mIoU']:.4f} {'(BEST)' if is_best else ''}"
                )

                # Save checkpoint
                self.save_checkpoint(is_best)

        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f} seconds")
        print(f"Best validation mIoU: {self.best_miou:.4f}")

        return self.best_miou

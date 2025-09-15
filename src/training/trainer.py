"""Main trainer class for semantic segmentation."""

import time
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from PIL import Image
import yaml
from .losses import SegmentationLoss, compute_class_weights
from .metrics import SegmentationMetrics, evaluate_model
from .scheduler import create_scheduler


class EarlyStopping:
    """Early stopping for segmentation training."""

    def __init__(
        self, patience: int = 20, min_delta: float = 0.005, verbose: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_score: float) -> bool:
        """Check if early stopping should trigger.

        Args:
            val_score: Current validation mIoU (higher is better)

        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(" Early stopping triggered!")
        else:
            self.best_score = val_score
            self.counter = 0
            if self.verbose:
                print(" Validation improved - resetting early stopping counter")

        return self.early_stop


class SegmentationTrainer:
    """Main trainer for semantic segmentation models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        config: Dict[str, Any],
        device: torch.device,
        export_predictions: bool = False,
        export_during_training: bool = False,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.export_predictions = export_predictions
        self.export_during_training = export_during_training

        self._setup_optimizer()
        self._setup_loss()
        self._setup_scheduler()
        self._setup_amp()
        self._setup_checkpointing()
        self._setup_early_stopping()
        self._setup_tensorboard()

        self.current_epoch = 0
        self.best_miou = 0.0
        self.train_history = []
        self.val_history = []

    def _setup_optimizer(self):
        """Setup optimizer with differential learning rates."""
        training_config = self.config["training"]

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
        elif training_config["optimizer"].lower() == "adamw":
            self.optimizer = torch.optim.AdamW(
                param_groups, weight_decay=training_config["weight_decay"]
            )
        else:
            raise ValueError(f"Unknown optimizer: {training_config['optimizer']}")

    def _setup_loss(self):
        """Setup loss function."""
        data_config = self.config["data"]
        training_config = self.config["training"]

        class_weights = None
        if training_config.get("class_weights", False):

            weights_file = Path("idd_class_weights.yaml")
            if weights_file.exists():
                print("Loading pre-computed class weights...")

                with open(weights_file) as f:
                    weights_data = yaml.safe_load(f)
                class_weights = torch.FloatTensor(weights_data["class_weights"]).to(
                    self.device
                )
                print(f"Loaded class weights: {class_weights}")
            else:
                print("Computing class weights...")
                class_weights = compute_class_weights(
                    self.train_loader,
                    data_config["num_classes"],
                    data_config["ignore_index"],
                ).to(self.device)
                print(f"Computed class weights: {class_weights}")

        self.criterion = SegmentationLoss(
            ignore_index=data_config["ignore_index"],
            aux_weight=training_config.get("aux_loss_weight", 0.0),
            class_weights=class_weights,
            lovasz_weight=training_config.get("lovasz_weight", 0.0),
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

    def _compute_loss(self, outputs, targets):
        """Compute loss supporting both dict outputs and auxiliary loss."""
        if isinstance(outputs, dict):

            main_loss = self.criterion(outputs["out"], targets)

            if "aux" in outputs:
                aux_weight = self.config["training"].get("aux_loss_weight", 0.0)
                if aux_weight > 0:
                    aux_loss = self.criterion(outputs["aux"], targets)
                    total_loss = main_loss + aux_weight * aux_loss
                    return total_loss

            return main_loss
        else:

            return self.criterion(outputs, targets)

    def _setup_early_stopping(self):
        """Setup early stopping with configurable parameters."""
        training_config = self.config["training"]

        early_stopping_config = training_config.get("early_stopping", {})
        patience = early_stopping_config.get("patience", 15)
        min_delta = early_stopping_config.get("min_delta", 0.001)

        self.early_stopping = EarlyStopping(
            patience=patience, min_delta=min_delta, verbose=True
        )
        self.best_model_path = self.checkpoint_dir / "best_model_early_stop.pt"

    def _setup_tensorboard(self):
        """Setup TensorBoard logging."""
        self.tensorboard_dir = self.save_dir / "tensorboard"
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(
            log_dir=str(self.tensorboard_dir),
            comment=f"_{self.model.get_model_name().replace(' ', '_')}",
        )

        print(f" TensorBoard logging: {self.tensorboard_dir}")
        print(f"   Launch with: tensorboard --logdir={self.tensorboard_dir.parent}")

    def _log_metrics_to_tensorboard(
        self, metrics: Dict[str, float], phase: str, epoch: int
    ):
        """Log metrics to TensorBoard."""
        for metric_name, value in metrics.items():
            self.writer.add_scalar(f"{metric_name}/{phase}", value, epoch)

    def _log_learning_rate(self, epoch: int):
        """Log current learning rate to TensorBoard."""

        for i, param_group in enumerate(self.optimizer.param_groups):
            lr = param_group["lr"]
            group_name = param_group.get("name", f"group_{i}")
            self.writer.add_scalar(f"learning_rate/{group_name}", lr, epoch)

    def _log_gradients_and_params(self, epoch: int):
        """Log gradient norms and parameter histograms to TensorBoard."""
        total_grad_norm = 0.0
        param_count = 0

        for name, param in self.model.named_parameters():
            if param.grad is not None and param.requires_grad:

                if epoch % 10 == 0:
                    self.writer.add_histogram(f"parameters/{name}", param, epoch)
                    self.writer.add_histogram(f"gradients/{name}", param.grad, epoch)

                # Calculate gradient norm
                grad_norm = param.grad.data.norm().item()
                total_grad_norm += grad_norm**2
                param_count += 1

        if param_count > 0:
            total_grad_norm = total_grad_norm**0.5
            self.writer.add_scalar("gradients/total_norm", total_grad_norm, epoch)

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with enhanced logging."""
        self.model.train()

        epoch_start_time = time.time()
        running_loss = 0.0
        metrics = SegmentationMetrics(
            self.config["data"]["num_classes"],
            self.config["data"]["ignore_index"],
            self.device,
        )

        pbar = tqdm(self.train_loader, desc=f"Train Epoch {self.current_epoch}")

        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets = targets.to(self.device)

            if self.use_amp:
                with autocast(device_type=self.device.type):
                    outputs = self.model(images)
                    loss = self._compute_loss(outputs, targets)

                self.scaler.scale(loss).backward()

                grad_clip_norm = self.config["training"].get("grad_clip_norm", 0.0)
                if grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), grad_clip_norm
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
            else:
                outputs = self.model(images)
                loss = self._compute_loss(outputs, targets)
                loss.backward()

                grad_clip_norm = self.config["training"].get("grad_clip_norm", 0.0)
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), grad_clip_norm
                    )

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()
            if isinstance(outputs, dict):
                metrics.update(outputs["out"], targets)
            else:
                metrics.update(outputs, targets)

            if batch_idx % 50 == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        if self.current_epoch % 5 == 0:
            self._log_gradients_and_params(self.current_epoch)

        results = metrics.compute()
        results["loss"] = running_loss / len(self.train_loader)

        epoch_time = time.time() - epoch_start_time
        self.writer.add_scalar("timing/epoch_duration", epoch_time, self.current_epoch)

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

    def export_val_predictions(self, save_dir: Optional[Path] = None) -> Path:
        """Export validation predictions in AutoNUE format."""
        if save_dir is None:
            save_dir = self.save_dir / "predictions" / "val"

        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"Exporting validation predictions to {save_dir}")
        self.model.eval()

        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(
                tqdm(self.val_loader, desc="Exporting predictions")
            ):
                images = images.to(self.device)

                outputs = self.model(images)
                if isinstance(outputs, dict):
                    preds = outputs["out"]
                else:
                    preds = outputs

                preds = torch.argmax(preds, dim=1)

                for i, pred in enumerate(preds):

                    global_idx = batch_idx * self.val_loader.batch_size + i
                    if global_idx >= len(self.val_loader.dataset):
                        break

                    img_path = self.val_loader.dataset.images[global_idx]

                    path_parts = Path(img_path).parts
                    city = path_parts[-2]
                    img_name = Path(img_path).stem.replace("_image", "")

                    city_dir = save_dir / city
                    city_dir.mkdir(exist_ok=True)

                    pred_path = city_dir / f"{img_name}_label.png"

                    pred_np = pred.cpu().numpy().astype(np.uint8)
                    pred_img = Image.fromarray(pred_np)
                    pred_img.save(pred_path)

        print(f"Predictions exported to: {save_dir}")

        data_root = self.config["data"]["root"]
        print(f"\nReady for AutoNUE evaluation:")
        print(f"python third_party/autonue/evaluate/idd_lite_evaluate_mIoU.py \\")
        print(f"  --gts {data_root}/gtFine/val \\")
        print(f"  --preds {save_dir} \\")
        print(f"  --num-workers 4")

        return save_dir

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

        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(state, latest_path)

        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(state, best_path)
            print(f"New best model saved with mIoU: {self.best_miou:.4f}")

            if self.export_predictions and self.export_during_training:
                self.export_val_predictions()

        if epoch is not None:
            epoch_path = self.checkpoint_dir / f"epoch_{epoch:03d}.pt"
            torch.save(state, epoch_path)

    def train(self):
        """Main training loop with early stopping and TensorBoard logging."""
        print("Starting training...")
        print(f"Model: {self.model.get_model_name()}")
        print(f"Epochs: {self.config['training']['epochs']}")
        print(f"Device: {self.device}")
        print(f"AMP: {self.use_amp}")
        print(f"Export predictions: {self.export_predictions}")
        print(
            f"Early stopping: Patience={self.early_stopping.patience}, Min Δ={self.early_stopping.min_delta*100:.1f}%"
        )

        start_time = time.time()

        for epoch in range(self.config["training"]["epochs"]):
            self.current_epoch = epoch

            train_results = self.train_epoch()
            self.train_history.append(train_results)

            self._log_metrics_to_tensorboard(train_results, "train", epoch)
            self._log_learning_rate(epoch)

            val_every = self.config["training"]["val_every"]
            if epoch % val_every == 0 or epoch == self.config["training"]["epochs"] - 1:
                val_results = self.validate()
                self.val_history.append(val_results)

                self._log_metrics_to_tensorboard(val_results, "val", epoch)

                current_miou = val_results["mIoU"]
                is_best = current_miou > self.best_miou
                if is_best:
                    self.best_miou = current_miou

                    self.writer.add_scalar(
                        "milestones/best_mIoU", self.best_miou, epoch
                    )

                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "best_miou": self.best_miou,
                        },
                        self.best_model_path,
                    )

                print(f"\nEpoch {epoch}")
                print(
                    f"Train - Loss: {train_results['loss']:.4f}, mIoU: {train_results['mIoU']:.4f}"
                )
                print(
                    f"Val   - mIoU: {val_results['mIoU']:.4f} {'(BEST)' if is_best else ''}"
                )

                if self.early_stopping(current_miou):
                    print(
                        f"\n Early stopping triggered after {self.early_stopping.counter} epochs without improvement"
                    )
                    print(
                        f" Best mIoU: {self.best_miou:.4f} (epoch {epoch - self.early_stopping.counter})"
                    )

                    self.writer.add_text(
                        "training/early_stopping",
                        f"Stopped at epoch {epoch}, best mIoU: {self.best_miou:.4f}",
                        epoch,
                    )
                    break
                elif self.early_stopping.counter > 0:
                    print(
                        f"  No improvement for {self.early_stopping.counter}/{self.early_stopping.patience} epochs"
                    )

                self.save_checkpoint(is_best)

        if self.best_model_path.exists():
            print(f"\n✨ Restoring best model weights (mIoU: {self.best_miou:.4f})")
            checkpoint = torch.load(
                self.best_model_path, map_location=self.device, weights_only=False
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])

        total_time = time.time() - start_time
        print(f"\n Training completed in {total_time:.2f} seconds")
        print(f"   Final model mIoU: {self.best_miou:.4f}")

        self.writer.add_text(
            "training/summary",
            f"Training completed in {total_time:.2f}s, Best mIoU: {self.best_miou:.4f}",
            self.current_epoch,
        )

        if self.export_predictions:
            pred_dir = self.export_val_predictions()
            print(f"\n  Final predictions available at: {pred_dir}")

        self.writer.close()
        print(f" TensorBoard logs saved to: {self.tensorboard_dir}")

        return self.best_miou

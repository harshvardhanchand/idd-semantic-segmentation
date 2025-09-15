"""Main training script for IDD semantic segmentation."""

import argparse
import sys
from pathlib import Path
import torch
import json


sys.path.append(str(Path(__file__).parent / "src"))

from utils.config import load_config
from utils.reproducibility import set_seed, print_env_info, get_env_info
from models import create_model
from data.dataset import create_dataloaders
from training.trainer import SegmentationTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train IDD semantic segmentation models"
    )
    parser.add_argument("config", type=str, help="Path to config file")
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint"
    )
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument(
        "--export-predictions",
        action="store_true",
        help="Export predictions for AutoNUE evaluation",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading config from {args.config}")
    config = load_config(args.config)

    set_seed(config["experiment"]["seed"])
    print_env_info()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    print(f"Creating model: {config['model']['name']}")
    model = create_model(
        model_name=config["model"]["name"], num_classes=config["data"]["num_classes"]
    )
    model.print_model_info(config)

    print("Creating data loaders...")
    train_loader, val_loader = create_dataloaders(config)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    trainer = SegmentationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        export_predictions=args.export_predictions,
        export_during_training=False,
    )

    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        trainer.current_epoch = checkpoint["epoch"]
        trainer.best_miou = checkpoint["best_miou"]

    exp_info = {
        "config": config,
        "env_info": get_env_info(),
        "model_info": {
            "name": model.get_model_name(),
            "parameters": model.count_parameters(),
        },
    }

    metadata_path = trainer.save_dir / "experiment_info.json"
    with open(metadata_path, "w") as f:
        json.dump(exp_info, f, indent=2, default=str)

    try:
        best_miou = trainer.train()
        print(f"Training completed! Best mIoU: {best_miou:.4f}")
    except KeyboardInterrupt:
        print("Training interrupted by user")
        trainer.save_checkpoint(epoch=trainer.current_epoch)
    except Exception as e:
        print(f"Training failed: {e}")
        trainer.save_checkpoint(epoch=trainer.current_epoch)
        raise


if __name__ == "__main__":
    main()

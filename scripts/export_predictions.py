"""Export predictions in AutoNUE evaluator format."""

import sys
from pathlib import Path
import argparse
from tqdm import tqdm
import torch
import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).parent.parent / "src"))

from models import create_model
from data.dataset import create_dataloaders
from utils.config import load_config


def export_predictions(
    model, dataloader, save_dir: Path, device: torch.device, dataset_root: Path
):
    """Export predictions in AutoNUE format."""
    save_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    print(f"Exporting predictions to {save_dir}")

    with torch.no_grad():
        for batch_num, (images, targets) in enumerate(
            tqdm(dataloader, desc="Exporting")
        ):
            images = images.to(device)

            outputs = model(images)
            if isinstance(outputs, dict):
                preds = outputs["out"]
            else:
                preds = outputs

            preds = torch.argmax(preds, dim=1)

            for i, pred in enumerate(preds):
                batch_idx = batch_num * dataloader.batch_size + i
                if batch_idx >= len(dataloader.dataset):
                    break

                img_path = dataloader.dataset.images[batch_idx]

                path_parts = Path(img_path).parts
                city = path_parts[-2]
                img_name = Path(img_path).stem.replace("_image", "")

                city_dir = save_dir / city
                city_dir.mkdir(exist_ok=True)

                pred_path = city_dir / f"{img_name}_label.png"

                pred_np = pred.cpu().numpy().astype(np.uint8)
                pred_img = Image.fromarray(pred_np)
                pred_img.save(pred_path)


def main():
    parser = argparse.ArgumentParser(
        description="Export predictions for AutoNUE evaluation"
    )
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("config", type=str, help="Path to config file")
    parser.add_argument("--output", type=str, default="preds", help="Output directory")
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val"],
        help="Dataset split",
    )
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    model = create_model(
        model_name=config["model"]["name"], num_classes=config["data"]["num_classes"]
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    print(f"Loaded checkpoint from {args.checkpoint}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Checkpoint mIoU: {checkpoint.get('best_miou', 'unknown'):.4f}")

    train_loader, val_loader = create_dataloaders(config)
    dataloader = val_loader if args.split == "val" else train_loader

    print(f"Dataset split: {args.split}")
    print(f"Number of samples: {len(dataloader.dataset)}")

    output_dir = Path(args.output)
    model_name = config["experiment"]["name"]
    pred_dir = output_dir / f"{model_name}_{args.split}_pngs"

    export_predictions(
        model=model,
        dataloader=dataloader,
        save_dir=pred_dir,
        device=device,
        dataset_root=Path(config["data"]["root"]),
    )

    print(f"\nPredictions exported to: {pred_dir}")
    print(f"Ready for AutoNUE evaluation:")
    print(f"python third_party/autonue/evaluate/idd_lite_evaluate_mIoU.py \\")
    print(f"  --gts {config['data']['root']}/gtFine/{args.split} \\")
    print(f"  --preds {pred_dir} \\")
    print(f"  --num-workers 4")


if __name__ == "__main__":
    main()

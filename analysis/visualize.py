"""Visualization utilities for IDD semantic segmentation analysis"""

import sys
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PIL import Image

plt.style.use("seaborn-v0_8")
sns.set_palette("husl")

sys.path.append(str(Path(__file__).parent.parent / "src"))

IDD_CLASSES = [
    "drivable",
    "non-drivable",
    "living-things",
    "vehicles",
    "road-side-objs",
    "far-objects",
    "sky",
]

COLORS = [
    [128, 64, 128],
    [244, 35, 232],
    [220, 20, 60],
    [0, 0, 142],
    [153, 153, 153],
    [70, 70, 70],
    [70, 130, 180],
]


def load_all_model_results(results_dir: Path, domain_gap_dir: Path = None):
    """Load evaluation results from all models."""
    all_results = {}

    for eval_file in results_dir.glob("*eval*.json"):
        try:
            with open(eval_file, "r") as f:
                data = json.load(f)

            if "parsed_metrics" in data and data["parsed_metrics"]:
                model_name = eval_file.stem.replace("_eval", "")

                if "deeplabv3plus" in model_name.lower():
                    model_name = "DeepLabV3Plus"
                elif "deeplabv3" in model_name.lower():
                    model_name = "DeepLabV3"
                elif "pspnet" in model_name.lower():
                    model_name = "PSPNet"

                all_results[model_name] = data["parsed_metrics"]
        except Exception:
            continue

    if domain_gap_dir and (domain_gap_dir / "results").exists():
        cs_eval = domain_gap_dir / "results" / "cityscapes_eval.json"
        if cs_eval.exists():
            try:
                with open(cs_eval, "r") as f:
                    data = json.load(f)
                if "parsed_metrics" in data:
                    all_results["VainF Zero-Shot"] = data["parsed_metrics"]
            except Exception:
                pass

    return all_results


def create_miou_comparison(all_results: dict, save_path: Path):
    """Create mIoU comparison chart."""
    if not all_results:
        return

    models, mious, colors = [], [], []

    for model_name, metrics in all_results.items():
        miou = metrics.get("mIoU", 0)

        if miou > 100:
            miou = miou / 100
        elif miou > 1:
            miou = miou / 100

        models.append(model_name)
        mious.append(miou)
        colors.append("steelblue" if "Zero-Shot" not in model_name else "indianred")

    sorted_data = sorted(zip(models, mious, colors), key=lambda x: x[1], reverse=True)
    models, mious, colors = zip(*sorted_data)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(models)), mious, color=colors, alpha=0.8, edgecolor="black")

    for bar, miou in zip(bars, mious):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.01,
            f"{miou:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Models", fontweight="bold")
    ax.set_ylabel("mIoU", fontweight="bold")
    ax.set_title("Domain Gap Analysis: mIoU Comparison", fontweight="bold", pad=20)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, ha="center")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_class_iou_comparison(all_results: dict, save_path: Path):
    """Create per-class IoU comparison chart."""
    if not all_results:
        return

    df_data = []
    for model_name, metrics in all_results.items():
        for class_name in IDD_CLASSES:
            class_iou = metrics.get(f"IoU_{class_name}", 0)

            if class_iou > 100:
                class_iou = class_iou / 10000
            elif class_iou > 1:
                class_iou = class_iou / 100

            df_data.append({"Class": class_name, "Model": model_name, "IoU": class_iou})

    df = pd.DataFrame(df_data)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df, x="Class", y="IoU", hue="Model", ax=ax)

    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Classes", fontweight="bold")
    ax.set_ylabel("IoU", fontweight="bold")
    ax.set_title("Per-Class IoU Comparison", fontweight="bold", pad=20)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(title="Models", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_segmentation_visualization(save_dir: Path, num_samples: int = 2):
    """Create side-by-side segmentation comparison."""
    save_dir.mkdir(parents=True, exist_ok=True)

    val_images_dir = Path("data/idd20k_lite/leftImg8bit/val")
    if not val_images_dir.exists():
        return

    image_files = []
    for subdir in val_images_dir.iterdir():
        if subdir.is_dir():
            image_files.extend(subdir.glob("*_image.jpg"))

    if not image_files:
        return

    for i, img_path in enumerate(image_files[:num_samples]):
        try:
            image = Image.open(img_path).convert("RGB")
            city_name = img_path.parent.name
            base_name = img_path.stem.replace("_image", "")

            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f"Segmentation Comparison - {base_name}", fontweight="bold")

            axes[0, 0].imshow(image)
            axes[0, 0].set_title("Original", fontweight="bold")
            axes[0, 0].axis("off")

            gt_path = Path(
                f"data/idd20k_lite/gtFine/val/{city_name}/{base_name}_label.png"
            )
            if gt_path.exists():
                axes[0, 1].imshow(Image.open(gt_path), cmap="tab10", vmin=0, vmax=6)
                axes[0, 1].set_title("Ground Truth", fontweight="bold")
            axes[0, 1].axis("off")

            model_preds = [
                (
                    "PSPNet",
                    f"results/preds/pspnet_val_pngs/{city_name}/{base_name}_label.png",
                ),
                (
                    "DeepLabV3",
                    f"results/preds/deeplabv3_val_pngs/{city_name}/{base_name}_label.png",
                ),
                (
                    "DeepLabV3Plus",
                    f"results/preds/deeplabv3plus_val_pngs/{city_name}/{base_name}_label.png",
                ),
                (
                    "VainF Zero-Shot",
                    f"domain_gap/preds/vainf_zero_shot/{city_name}/{base_name}_label.png",
                ),
            ]

            positions = [(0, 2), (1, 0), (1, 1), (1, 2)]

            for (model_name, pred_path), (row, col) in zip(model_preds, positions):
                if Path(pred_path).exists():
                    axes[row, col].imshow(
                        Image.open(pred_path), cmap="tab10", vmin=0, vmax=6
                    )
                    axes[row, col].set_title(model_name, fontweight="bold")
                axes[row, col].axis("off")

            plt.tight_layout()
            plt.savefig(
                save_dir / f"comparison_{i+1}.png", dpi=300, bbox_inches="tight"
            )
            plt.close()

        except Exception:
            continue


def create_class_legend(save_path: Path):
    """Create class color legend."""
    fig, ax = plt.subplots(figsize=(8, 4))

    patches = [
        plt.Rectangle(
            (0, 0), 1, 1, facecolor=np.array(color) / 255.0, edgecolor="black"
        )
        for color in COLORS
    ]

    ax.legend(patches, IDD_CLASSES, loc="center", ncol=3, frameon=True)
    ax.axis("off")
    plt.title("IDD-Lite Classes", fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Domain gap visualization")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    parser.add_argument(
        "--domain-gap-dir", default="domain_gap", help="Domain gap directory"
    )
    parser.add_argument("--output", default="visualizations", help="Output directory")
    parser.add_argument("--num-samples", type=int, default=2, help="Number of samples")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = load_all_model_results(
        Path(args.results_dir), Path(args.domain_gap_dir)
    )

    create_miou_comparison(all_results, output_dir / "miou_comparison.png")
    create_class_iou_comparison(all_results, output_dir / "class_iou_comparison.png")
    create_segmentation_visualization(
        output_dir / "segmentation_samples", args.num_samples
    )
    create_class_legend(output_dir / "class_legend.png")

    print(f"Visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()

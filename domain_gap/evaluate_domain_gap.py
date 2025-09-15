"""Domain gap evaluation and comparison."""

import argparse
import json
from pathlib import Path
import pandas as pd
import sys

sys.path.append(str(Path(__file__).parent.parent))

from scripts.run_evaluation import run_autonue_evaluation


def run_cityscapes_evaluation(pred_dir: Path, gt_dir: Path, output_file: Path):
    """Evaluate Cityscapes zero-shot performance."""
    results = run_autonue_evaluation(gt_dir=gt_dir, pred_dir=pred_dir, num_workers=4)

    if not results["success"]:
        print(f" Evaluation failed: {results.get('error', 'Unknown error')}")
        return None

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    metrics = results.get("parsed_metrics", {})
    miou = metrics.get("mIoU", 0.0)
    print(f"Cityscapes zero-shot mIoU: {miou:.3f}")

    return results


def load_idd_trained_results(results_dir: Path):
    """Load IDD-trained model results."""
    idd_results = {}

    for result_file in results_dir.glob("*eval*.json"):
        try:
            with open(result_file, "r") as f:
                data = json.load(f)

            if "parsed_metrics" in data:
                name = result_file.stem.replace("_eval", "").replace("official_", "")

                # Clean naming
                if "deeplabv3plus" in name.lower():
                    name = "DeepLabV3Plus MobileNet (IDD)"
                elif "fcn" in name.lower():
                    name = "FCN (IDD)"
                else:
                    name = f"{name} (IDD)"

                idd_results[name] = data["parsed_metrics"]

        except Exception as e:
            print(f"Error loading {result_file.name}: {e}")

    return idd_results


def create_comparison_table(cs_results: dict, idd_results: dict, output_file: Path):
    """Create model comparison table."""
    data = []

    if cs_results and "parsed_metrics" in cs_results:
        cs_metrics = cs_results["parsed_metrics"]
        data.append(
            {
                "Model": "DeepLabV3Plus MobileNet (Cityscapes)",
                "Training": "Cityscapes",
                "Test": "IDD-Lite",
                "mIoU": cs_metrics.get("mIoU", 0.0),
                "Type": "Zero-shot",
            }
        )

    for model, metrics in idd_results.items():
        data.append(
            {
                "Model": model,
                "Training": "IDD-Lite",
                "Test": "IDD-Lite",
                "mIoU": metrics.get("mIoU", 0.0),
                "Type": "Same-domain",
            }
        )

    df = pd.DataFrame(data).sort_values("mIoU", ascending=False)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    print(f"\n Domain Gap Comparison:")
    print(df.to_string(index=False, float_format="%.3f"))

    # Calculate domain gap
    if len(data) >= 2:
        cs_miou = next(
            (item["mIoU"] for item in data if item["Type"] == "Zero-shot"), 0
        )
        idd_mious = [item["mIoU"] for item in data if item["Type"] == "Same-domain"]

        if cs_miou > 0 and idd_mious:
            best_idd = max(idd_mious)
            gap = best_idd - cs_miou
            gap_pct = (gap / best_idd) * 100

            print(f"\n Domain Gap Analysis:")
            print(f"   Best IDD-trained: {best_idd:.3f}")
            print(f"   Cityscapes zero-shot: {cs_miou:.3f}")
            print(f"   Gap: {gap:.3f} ({gap_pct:.1f}%)")

    return df


def create_per_class_comparison(cs_results: dict, idd_results: dict, output_file: Path):
    """Create per-class IoU comparison."""
    try:
        from .label_mapping import IDD_LITE_L1_CLASSES
    except ImportError:
        from label_mapping import IDD_LITE_L1_CLASSES

    cs_class_ious = {}
    if cs_results and "parsed_metrics" in cs_results:
        for key, value in cs_results["parsed_metrics"].items():
            if key.startswith("IoU_") and key != "IoU_mean":
                class_name = key.replace("IoU_", "")
                cs_class_ious[class_name] = value

    class_data = []
    for i, class_name in enumerate(IDD_LITE_L1_CLASSES):
        row = {
            "Class": class_name,
            "ID": i,
            "Cityscapes_IoU": cs_class_ious.get(class_name, 0.0),
        }

        # Add IDD results
        for model, metrics in idd_results.items():
            iou = next(
                (
                    v
                    for k, v in metrics.items()
                    if k.startswith("IoU_") and class_name in k
                ),
                0.0,
            )
            row[f"{model}_IoU"] = iou

        class_data.append(row)

    df = pd.DataFrame(class_data)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    return df


def main():
    parser = argparse.ArgumentParser(description="Domain gap evaluation")
    parser.add_argument(
        "--cs-pred-dir", type=str, required=True, help="Cityscapes predictions"
    )
    parser.add_argument(
        "--idd-gt-dir", type=str, required=True, help="IDD ground truth"
    )
    parser.add_argument(
        "--idd-results-dir", type=str, default="results", help="IDD results directory"
    )
    parser.add_argument(
        "--output-dir", type=str, default="domain_gap/results", help="Output directory"
    )

    args = parser.parse_args()

    cs_pred_dir = Path(args.cs_pred_dir)
    idd_gt_dir = Path(args.idd_gt_dir)
    idd_results_dir = Path(args.idd_results_dir)
    output_dir = Path(args.output_dir)

    # Validate paths
    for path, name in [(cs_pred_dir, "cs-pred-dir"), (idd_gt_dir, "idd-gt-dir")]:
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")

    print(f"  Domain Gap Analysis")
    print(f"   Cityscapes: {cs_pred_dir}")
    print(f"   IDD GT: {idd_gt_dir}")
    print(f"   Output: {output_dir}")

    cs_eval_file = output_dir / "cityscapes_eval.json"
    cs_results = run_cityscapes_evaluation(cs_pred_dir, idd_gt_dir, cs_eval_file)

    if not cs_results:
        print(" Cityscapes evaluation failed")
        return

    idd_results = load_idd_trained_results(idd_results_dir)
    if not idd_results:
        print(" No IDD results found")
        return

    print(f"Found {len(idd_results)} IDD-trained models")

    comparison_file = output_dir / "comparison.csv"
    comparison_df = create_comparison_table(cs_results, idd_results, comparison_file)

    per_class_file = output_dir / "per_class.csv"
    per_class_df = create_per_class_comparison(cs_results, idd_results, per_class_file)

    print(f"\n Results → {output_dir}/")


if __name__ == "__main__":
    main()

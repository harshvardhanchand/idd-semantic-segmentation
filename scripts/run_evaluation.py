"""Wrapper for running AutoNUE official evaluation."""

import argparse
import subprocess
import json
from pathlib import Path
import re
from typing import Dict, Any


def parse_evaluation_output(output: str) -> Dict[str, Any]:
    """Parse AutoNUE evaluator output to extract metrics."""
    lines = output.strip().split("\n")
    results = {}

    for line in lines:
        if "Mean IoU" in line or "mIoU" in line:
            match = re.search(r"[\d\.]+", line)
            if match:
                results["mIoU"] = float(match.group())

        elif "IoU for class" in line or line.strip().startswith("IoU"):
            parts = line.split()
            if len(parts) >= 3:
                try:
                    class_name = parts[1] if "class" in parts else parts[0]
                    iou_val = float(parts[-1])
                    results[f"IoU_{class_name}"] = iou_val
                except:
                    continue

    return results


def run_autonue_evaluation(
    gt_dir: Path, pred_dir: Path, num_workers: int = 4, evaluator_path: Path = None
) -> Dict[str, Any]:
    """Run AutoNUE official evaluation."""
    if evaluator_path is None:
        evaluator_path = (
            Path(__file__).parent.parent
            / "third_party"
            / "autonue"
            / "evaluation"
            / "idd_lite_evaluate_mIoU.py"
        )

    if not evaluator_path.exists():
        raise FileNotFoundError(f"AutoNUE evaluator not found at {evaluator_path}")

    if not gt_dir.exists():
        raise FileNotFoundError(f"Ground truth directory not found: {gt_dir}")

    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")

    print(f"Running official AutoNUE evaluation:")
    print(f"  GT dir: {gt_dir}")
    print(f"  Pred dir: {pred_dir}")
    print(f"  Evaluator: {evaluator_path}")

    cmd = [
        "python",
        str(evaluator_path),
        "--gts",
        str(gt_dir),
        "--preds",
        str(pred_dir),
        "--num-workers",
        str(num_workers),
    ]

    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=1500,
        )

        print("Evaluation completed successfully!")
        print("\nEvaluator output:")
        print("-" * 50)
        print(result.stdout)
        if result.stderr:
            print("Errors/Warnings:")
            print(result.stderr)
        print("-" * 50)

        parsed_results = parse_evaluation_output(result.stdout)

        return {
            "success": True,
            "raw_output": result.stdout,
            "parsed_metrics": parsed_results,
            "command": " ".join(cmd),
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "Evaluation timed out after 25 minutes",
            "command": " ".join(cmd),
        }

    except subprocess.CalledProcessError as e:
        return {
            "success": False,
            "error": f"Evaluation failed with return code {e.returncode}",
            "stdout": e.stdout,
            "stderr": e.stderr,
            "command": " ".join(cmd),
        }


def main():
    parser = argparse.ArgumentParser(description="Run AutoNUE official evaluation")
    parser.add_argument("--gts", type=str, required=True, help="Ground truth directory")
    parser.add_argument(
        "--preds", type=str, required=True, help="Predictions directory"
    )
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--output", type=str, help="Output JSON file for results")
    parser.add_argument("--evaluator", type=str, help="Path to evaluator script")
    args = parser.parse_args()

    evaluator_path = Path(args.evaluator) if args.evaluator else None

    results = run_autonue_evaluation(
        gt_dir=Path(args.gts),
        pred_dir=Path(args.preds),
        num_workers=args.num_workers,
        evaluator_path=evaluator_path,
    )

    if results["success"]:
        print(f"\n Evaluation successful!")
        if "parsed_metrics" in results and results["parsed_metrics"]:
            print("Parsed metrics:")
            for metric, value in results["parsed_metrics"].items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")
    else:
        print(f"\n Evaluation failed: {results['error']}")
        if "stderr" in results:
            print(f"Error output: {results['stderr']}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()

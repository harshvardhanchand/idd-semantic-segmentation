"""Cityscapes → IDD-Lite zero-shot inference using VainF DeepLabV3Plus MobileNet."""

import argparse
from pathlib import Path
import sys
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


vainf_path = Path(__file__).parent.parent / "vainf-deeplabv3"
sys.path.insert(0, str(vainf_path))

from network.modeling import deeplabv3plus_mobilenet


from label_mapping import CITYSCAPES_TO_IDD_MAPPING


def setup_model(checkpoint_path: Path, device: str = "cuda:0"):
    """Setup VainF DeepLabV3Plus MobileNet model."""

    model = deeplabv3plus_mobilenet(num_classes=19, output_stride=16)

    print(f"Loading VainF checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    print(" VainF model loaded successfully")

    return model.to(device).eval()


def get_transform():
    """Image preprocessing transform (VainF standard: 513x513)."""
    return transforms.Compose(
        [
            transforms.Resize(513),
            transforms.CenterCrop(513),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def find_images(root: Path):
    """Find IDD validation images."""
    extensions = ["*.jpg", "*.png"]
    images = []
    for ext in extensions:
        images.extend(root.rglob(ext))
    return sorted(images)


def run_inference(
    model,
    idd_val_root: Path,
    output_root: Path,
    mapping: dict,
    device: str,
    max_images: int = None,
):
    """Run zero-shot inference."""
    images = (
        find_images(idd_val_root)[:max_images]
        if max_images
        else find_images(idd_val_root)
    )

    if not images:
        raise ValueError(f"No images found in {idd_val_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    transform = get_transform()
    processed = 0

    with torch.no_grad():
        for img_path in tqdm(images, desc="Processing"):
            try:
                city = img_path.parent.name
                stem = img_path.stem.replace("_image", "").replace("_leftImg8bit", "")

                output_city_dir = output_root / city
                output_city_dir.mkdir(exist_ok=True)
                output_path = output_city_dir / f"{stem}_label.png"

                if output_path.exists():
                    continue

                image = Image.open(img_path).convert("RGB")
                input_tensor = transform(image).unsqueeze(0).to(device)

                outputs = model(input_tensor)

                predictions = torch.argmax(outputs, dim=1).squeeze(0)

                predictions = (
                    F.interpolate(
                        predictions.unsqueeze(0).unsqueeze(0).float(),
                        size=image.size[::-1],
                        mode="nearest",
                    )
                    .squeeze()
                    .long()
                )

                seg_mask = predictions.cpu().numpy().astype(np.uint8)
                idd_mask = np.full_like(seg_mask, 255, dtype=np.uint8)
                for cs_id, idd_id in mapping.items():
                    idd_mask[seg_mask == cs_id] = idd_id

                Image.fromarray(idd_mask).save(output_path)
                processed += 1

            except Exception as e:
                print(f"Error: {img_path.name}: {e}")
                continue

    print(f"Processed {processed}/{len(images)} images → {output_root}")
    return processed


def main():
    parser = argparse.ArgumentParser(description="Cityscapes → IDD zero-shot inference")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Cityscapes checkpoint path"
    )
    parser.add_argument(
        "--idd-val-root", type=str, required=True, help="IDD validation images root"
    )
    parser.add_argument("--out-root", type=str, required=True, help="Output directory")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--max-images", type=int, help="Limit images for testing")
    parser.add_argument(
        "--show-mapping", action="store_true", help="Show mapping and exit"
    )

    args = parser.parse_args()

    if args.show_mapping:
        try:
            from .label_mapping import print_mapping_summary

            print_mapping_summary()
        except ImportError:
            from label_mapping import print_mapping_summary

            print_mapping_summary()
        return

    checkpoint_path = Path(args.checkpoint)
    idd_val_root = Path(args.idd_val_root)
    output_root = Path(args.out_root)

    for path, name in [(checkpoint_path, "checkpoint"), (idd_val_root, "idd-val-root")]:
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")

    print(f" Cityscapes → IDD Zero-Shot Inference (VainF)")
    print(f"   Model: {checkpoint_path.name}")
    print(f"   Images: {idd_val_root}")
    print(f"   Output: {output_root}")

    try:
        model = setup_model(checkpoint_path, args.device)
        mapping = CITYSCAPES_TO_IDD_MAPPING

        processed = run_inference(
            model, idd_val_root, output_root, mapping, args.device, args.max_images
        )

        if processed > 0:
            print("\n Next: Run evaluation:")
            print(f"python third_party/autonue/evaluate/idd_lite_evaluate_mIoU.py \\")
            print(f"  --gts data/idd20k_lite/gtFine/val --preds {output_root}")

    except Exception as e:
        print(f" Failed: {e}")


if __name__ == "__main__":
    main()

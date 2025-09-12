"""
Test script to verify models are working correctly.
Quick smoke test for DeepLabV3+ and FCN models.
"""

import torch
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))


def test_models():
    """Test model creation and basic functionality"""

    print("🧪 TESTING IDD SEGMENTATION MODELS")
    print("=" * 50)

    # Import our models
    from models import create_model, list_models, get_model_comparison

    # List available models
    list_models()

    # Test both models
    model_names = ["deeplabv3", "fcn"]

    for model_name in model_names:
        print(f"\n🔹 TESTING {model_name.upper()}")
        print("-" * 30)

        try:
            # Create model
            model = create_model(model_name, num_classes=7)
            print(f"✅ Model created: {model.get_model_name()}")

            # Print model info
            model.print_model_info()

            # Test forward pass with dummy data
            batch_size = 2
            height, width = 512, 512
            dummy_input = torch.randn(batch_size, 3, height, width)

            model.eval()
            with torch.no_grad():
                output = model(dummy_input)
                print(f"✅ Forward pass successful")
                print(f"   Input shape: {dummy_input.shape}")
                print(f"   Output shape: {output.shape}")
                print(f"   Expected shape: ({batch_size}, 7, {height}, {width})")

                # Verify output shape
                expected_shape = (batch_size, 7, height, width)
                if output.shape == expected_shape:
                    print(f"✅ Output shape correct")
                else:
                    print(f"❌ Output shape mismatch")

            # Test parameter groups for differential learning rates
            backbone_lr = 0.001
            classifier_lr = 0.01
            param_groups = model.get_param_groups(backbone_lr, classifier_lr)
            print(f"✅ Parameter groups created: {len(param_groups)} groups")

            for i, group in enumerate(param_groups):
                group_name = group.get("name", f"group_{i}")
                group_lr = group["lr"]
                num_params = len(group["params"])
                print(f"   {group_name}: {num_params} params, lr={group_lr}")

        except Exception as e:
            print(f"❌ Error testing {model_name}: {str(e)}")
            import traceback

            traceback.print_exc()

    # Model comparison
    print(f"\n📊 MODEL COMPARISON")
    print("-" * 30)
    comparison = get_model_comparison()

    for model_name, info in comparison.items():
        print(f"\n{model_name.upper()}:")
        for key, value in info.items():
            if isinstance(value, int) and value > 1000:
                print(f"  {key}: {value:,}")
            else:
                print(f"  {key}: {value}")

    print(f"\n✅ ALL TESTS COMPLETED")
    print("=" * 50)


if __name__ == "__main__":
    test_models()

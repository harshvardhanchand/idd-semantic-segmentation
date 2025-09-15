# Makefile for IDD Semantic Segmentation

# Default paths
DATA_ROOT ?= data/idd20k_lite
GT_VAL ?= $(DATA_ROOT)/gtFine/val
PRED_DIR ?= preds
WORKERS ?= 2

# Configs
PSPNET_CONFIG = src/configs/pspnet.yaml
DEEPLABV3_CONFIG = src/configs/deeplabv3.yaml
DEEPLABV3PLUS_CONFIG = src/configs/deeplabv3plus.yaml
TEST_CONFIG = src/configs/test_cpu.yaml

# Training targets
train-pspnet: ## Train PSPNet ResNet50
	python train.py $(PSPNET_CONFIG)

train-pspnet-colab: ## Train PSPNet in Colab environment
	@echo "Training PSPNet ResNet50"
	@echo "GPU Memory - Allocated: $$(python -c 'import torch; print(f"{torch.cuda.memory_allocated()/1e9:.1f}GB")') / $$(python -c 'import torch; print(f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")') ($$(python -c 'import torch; print(f"{100*torch.cuda.memory_allocated()/torch.cuda.get_device_properties(0).total_memory:.1f}%")'))"
	@echo "GPU Cache: $$(python -c 'import torch; print(f"{torch.cuda.memory_reserved()/1e9:.1f}GB")')"
	python train.py src/configs/pspnet_colab.yaml --export-predictions
	@echo "✅ PSPNet ResNet50 training completed successfully!"
	@python -c 'import torch; torch.cuda.empty_cache(); print("Memory cleared")'

train-deeplabv3-colab: ## Train DeepLabV3 ResNet50 in Colab 
	@echo "Training DeepLabV3 ResNet50"
	@echo "GPU Memory - Allocated: $$(python -c 'import torch; print(f"{torch.cuda.memory_allocated()/1e9:.1f}GB")') / $$(python -c 'import torch; print(f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")') ($$(python -c 'import torch; print(f"{100*torch.cuda.memory_allocated()/torch.cuda.get_device_properties(0).total_memory:.1f}%")'))"
	@echo "GPU Cache: $$(python -c 'import torch; print(f"{torch.cuda.memory_reserved()/1e9:.1f}GB")')"
	python train.py src/configs/deeplabv3_colab.yaml --export-predictions
	@echo "✅ DeepLabV3 ResNet50 training completed successfully!"
	@python -c 'import torch; torch.cuda.empty_cache(); print("Memory cleared")'

train-deeplabv3plus-colab: ## Train DeepLabV3Plus MobileNet in Colab
	@echo "Training DeepLabV3Plus MobileNet"
	@echo "GPU Memory - Allocated: $$(python -c 'import torch; print(f"{torch.cuda.memory_allocated()/1e9:.1f}GB")') / $$(python -c 'import torch; print(f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")') ($$(python -c 'import torch; print(f"{100*torch.cuda.memory_allocated()/torch.cuda.get_device_properties(0).total_memory:.1f}%")'))"
	@echo "GPU Cache: $$(python -c 'import torch; print(f"{torch.cuda.memory_reserved()/1e9:.1f}GB")')"
	python train.py src/configs/deeplabv3plus_colab.yaml --export-predictions
	@echo "✅ DeepLabV3Plus MobileNet training completed successfully!"
	@python -c 'import torch; torch.cuda.empty_cache(); print("Memory cleared")'

train-deeplabv3: ## Train DeepLabV3 ResNet50
	python train.py $(DEEPLABV3_CONFIG)

train-deeplabv3plus: ## Train DeepLabV3Plus MobileNet
	python train.py $(DEEPLABV3PLUS_CONFIG)

train-all-colab: ## Train all models in Colab with memory management
	@echo "[1/3] Training PSPNet ResNet50"
	@$(MAKE) train-pspnet-colab
	@echo ""
	@echo "[2/3] Training DeepLabV3 ResNet50"  
	@$(MAKE) train-deeplabv3-colab
	@echo ""
	@echo "[3/3] Training DeepLabV3Plus MobileNet"
	@$(MAKE) train-deeplabv3plus-colab
	@echo ""
	@echo "🎉 All models trained successfully!"

# CPU Testing
test-cpu: ## Quick CPU test
	python train.py $(TEST_CONFIG)

# Evaluation targets - FIXED to properly export predictions and create JSON files
eval-pspnet: ## Evaluate PSPNet
	@echo "Exporting PSPNet predictions..."
	python scripts/export_predictions.py \
		runs/idd_pspnet_resnet50_conservative/checkpoints/best.pt \
		$(PSPNET_CONFIG) \
		--output results/preds/pspnet_val_pngs \
		--split val
	@echo "Running official evaluation..."
	python scripts/run_evaluation.py \
		--gts $(GT_VAL) \
		--preds results/preds/pspnet_val_pngs \
		--output results/pspnet_eval.json \
		--num-workers $(WORKERS)

eval-deeplabv3: ## Evaluate DeepLabV3
	@echo "Exporting DeepLabV3 predictions..."
	python scripts/export_predictions.py \
		runs/idd_deeplabv3_resnet50_optimized/checkpoints/best.pt \
		$(DEEPLABV3_CONFIG) \
		--output results/preds/deeplabv3_val_pngs \
		--split val
	@echo "Running official evaluation..."
	python scripts/run_evaluation.py \
		--gts $(GT_VAL) \
		--preds results/preds/deeplabv3_val_pngs \
		--output results/deeplabv3_eval.json \
		--num-workers $(WORKERS)

eval-deeplabv3plus: ## Evaluate DeepLabV3Plus
	@echo "Exporting DeepLabV3Plus predictions..."
	python scripts/export_predictions.py \
		runs/idd_deeplabv3plus_mobilenet_optimized/checkpoints/best.pt \
		$(DEEPLABV3PLUS_CONFIG) \
		--output results/preds/deeplabv3plus_val_pngs \
		--split val
	@echo "Running official evaluation..."
	python scripts/run_evaluation.py \
		--gts $(GT_VAL) \
		--preds results/preds/deeplabv3plus_val_pngs \
		--output results/deeplabv3plus_eval.json \
		--num-workers $(WORKERS)

# Visualization - Fixed for cleaned up visualize.py script
visualize-deeplabv3plus: ## Visualize DeepLabV3Plus training results
	python analysis/visualize.py \
		--results-dir results \
		--output visualizations/deeplabv3plus \
		--num-samples 3

visualize-deeplabv3: ## Visualize DeepLabV3 training results
	python analysis/visualize.py \
		--results-dir results \
		--output visualizations/deeplabv3 \
		--num-samples 3

visualize-pspnet: ## Visualize PSPNet training results
	python analysis/visualize.py \
		--results-dir results \
		--output visualizations/pspnet \
		--num-samples 3

visualize-domain-gap: ## Create comprehensive domain gap visualizations
	python analysis/visualize.py \
		--results-dir results \
		--domain-gap-dir domain_gap \
		--output visualizations/domain_gap \
		--num-samples 2

visualize-all:
	@echo "=== Generating Segmentation Visualizations for All Models ==="
	@echo "Creating PSPNet visualizations..."
	@$(MAKE) visualize-pspnet
	@echo ""
	@echo "Creating DeepLabV3 visualizations..."
	@$(MAKE) visualize-deeplabv3
	@echo ""
	@echo "Creating DeepLabV3Plus visualizations..."
	@$(MAKE) visualize-deeplabv3plus
	@echo ""
	@echo "✅ All model visualizations completed!"

# Domain gap analysis
cityscapes-inference: ## Run Cityscapes zero-shot inference (VainF DeepLabV3Plus)
	python domain_gap/cs_to_idd_lite_infer.py \
		--checkpoint pretrained_weights/deeplabv3plus_mobilenet_cityscapes.pth \
		--idd-val-root data/idd20k_lite/leftImg8bit/val \
		--out-root domain_gap/preds/vainf_zero_shot
	@echo "Running zero-shot evaluation..."
	python scripts/run_evaluation.py \
		--gts $(GT_VAL) \
		--preds domain_gap/preds/vainf_zero_shot \
		--output domain_gap/results/vainf_zero_shot_eval.json \
		--num-workers $(WORKERS)

eval-domain-gap: ## Evaluate domain gap between models  
	python domain_gap/evaluate_domain_gap.py

domain-gap-full: ## Complete domain gap analysis with ALL models
	@echo "=== IDD-Trained Models Evaluation ==="
	@echo "Evaluating PSPNet..."
	@$(MAKE) eval-pspnet
	@echo ""
	@echo "Evaluating DeepLabV3..."
	@$(MAKE) eval-deeplabv3
	@echo ""
	@echo "Evaluating DeepLabV3Plus..."
	@$(MAKE) eval-deeplabv3plus
	@echo ""
	@echo "=== Cityscapes Zero-Shot Inference ==="
	@$(MAKE) cityscapes-inference
	@echo ""
	@echo "=== Domain Gap Comparison ==="
	@$(MAKE) eval-domain-gap
	@echo ""
	@echo "=== Generate Complete Visualizations ==="
	@$(MAKE) visualize-domain-gap

# Export predictions for submission
export-predictions: ## Export test predictions for all models
	python scripts/export_predictions.py --model-type pspnet
	python scripts/export_predictions.py --model-type deeplabv3  
	python scripts/export_predictions.py --model-type deeplabv3plus

# Cleanup
clean: ## Clean up temporary files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: train-pspnet train-deeplabv3 train-deeplabv3plus train-all-colab test-cpu eval-pspnet eval-deeplabv3 eval-deeplabv3plus cityscapes-inference eval-domain-gap domain-gap-full export-predictions clean help visualize-deeplabv3plus visualize-deeplabv3 visualize-pspnet visualize-domain-gap visualize-all 
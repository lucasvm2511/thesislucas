#!/bin/bash

# Ablation Study: Compare Hard-Only vs Alternating Soft/Hard Training
# This script runs both training strategies on multiple subnet configurations

echo "=========================================="
echo "Training Strategy Comparison"
echo "=========================================="
echo ""
echo "This ablation compares:"
echo "  1. Hard-only training (no soft gates)"
echo "  2. Alternating soft/hard training (standard)"
echo ""
echo "Testing on 3 different subnet configurations:"
echo "  - Small/Efficient subnet"
echo "  - Average subnet"
echo "  - Large/Accurate subnet"
echo ""
echo "This will run 30 epochs total (5 epochs × 2 strategies × 3 subnets)"
echo ""

# Configuration
DATASET="cifar100"
DATA_PATH="datasets/cifar100"
N_CLASSES=100
EPOCHS=5
BATCH_SIZE=128
LR=0.1
GPU=0

# Run comparison (both strategies on all subnets)
echo "Running comparison experiment..."
echo "Estimated time: ~30-45 minutes"
echo ""
python3 ablations/hard_only_training.py \
    --dataset $DATASET \
    --data $DATA_PATH \
    --n_classes $N_CLASSES \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LR \
    --device $GPU \
    --output_path ablations/results/hard_only_training

echo ""
echo "=========================================="
echo "Comparison completed!"
echo "=========================================="
echo ""
echo "Results saved to: ablations/results/hard_only_training/"
echo "  - comparison_results.json: Aggregated results across all subnets"
echo "  - comparison.log: Detailed training logs"
echo ""

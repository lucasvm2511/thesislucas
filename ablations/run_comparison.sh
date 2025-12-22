#!/bin/bash

# Compare Training Strategies: Hard-Only vs Alternating Soft/Hard
# This script runs both strategies and compares their results

echo "=========================================="
echo "Training Strategy Comparison"
echo "=========================================="
echo ""
echo "This experiment compares:"
echo "  1. Hard-only training (no soft gates)"
echo "  2. Alternating soft/hard training (standard)"
echo ""
echo "Both use the same subnet configuration for fair comparison."
echo ""

# Configuration
DATASET="cifar100"
N_CLASSES=10
EPOCHS=5
BATCH_SIZE=128
LR=0.1
GPU=0

# Run comparison
echo "Running comparison experiment (5 epochs each)..."
python3 ablations/compare_training_strategies.py \
    --dataset $DATASET \
    --n_classes $N_CLASSES \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LR \
    --device $GPU \
    --output_path ablations/results/comparison

echo ""
echo "=========================================="
echo "Comparison completed!"
echo "=========================================="
echo ""
echo "Results saved to: ablations/results/comparison/"
echo ""

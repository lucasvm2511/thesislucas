#!/bin/bash
# Script to compare different gate types for Layer Skipping models
# Trains 4 models: conv gate, attention gate, stable gate, and no gate
# Each model has 1 gate defined in the subnet config (or no gate for baseline)

# Configuration
DATASET="cifar100"
N_CLASSES=100
DEVICE="0"
SUPERNET_PATH="./supernets/ofa_mbv3_d234_e346_k357_w1.0"
BATCH_SIZE=128
N_WORKERS=4
RESOLUTION=84
BACKBONE_EPOCHS=0
TRAINING_EPOCHS=10
VAL_SPLIT=0.0

# Base output directory
BASE_OUTPUT_DIR="results/gate_comparison_cifar100"
mkdir -p ${BASE_OUTPUT_DIR}

# Subnet path - assumes you have a subnet configuration with gate parameters
# Modified to use 1 gate at position 10 (target_sparsity = 0.5)
SUBNET_PATH="results/layerskipping_cifar100_10e_attention_seed1/final/net-trade-off_0/net.subnet"

if [ ! -f "$SUBNET_PATH" ]; then
    echo "Error: Subnet config not found at $SUBNET_PATH"
    echo "Please provide a valid subnet configuration path"
    exit 1
fi

echo "=================================================="
echo "Gate Type Comparison Experiment"
echo "Dataset: $DATASET"
echo "Subnet config: $SUBNET_PATH"
echo "Training epochs: $TRAINING_EPOCHS"
echo "=================================================="

# Function to train a model with specific gate type
train_with_gate_type() {
    local gate_type=$1
    local output_dir="${BASE_OUTPUT_DIR}/${gate_type}_gate"
    
    echo ""
    echo "=================================================="
    echo "Training model with ${gate_type} gate"
    echo "Output directory: $output_dir"
    echo "=================================================="
    
    python3 ls_train.py \
        --confusion_matrix True \
        --model skippingmobilenetv3 \
        --dataset ${DATASET} \
        --n_classes ${N_CLASSES} \
        --device ${DEVICE} \
        --supernet_path ${SUPERNET_PATH} \
        --model_path ${SUBNET_PATH} \
        --output_path ${output_dir} \
        --batch_size ${BATCH_SIZE} \
        --n_workers ${N_WORKERS} \
        --resolution ${RESOLUTION} \
        --backbone_epochs ${BACKBONE_EPOCHS} \
        --training_epochs ${TRAINING_EPOCHS} \
        --val_split ${VAL_SPLIT} \
        --gate_type ${gate_type} \
        --save \
        --eval_test 
    
    echo "Completed training with ${gate_type} gate"
    echo "Results saved to: ${output_dir}"
}

# Train model with no gate (baseline)
train_baseline() {
    local output_dir="${BASE_OUTPUT_DIR}/no_gate"
    
    echo ""
    echo "=================================================="
    echo "Training baseline model (no gates)"
    echo "Output directory: $output_dir"
    echo "=================================================="
    
    # Create output directory
    mkdir -p "${output_dir}"
    
    # For baseline, we use the regular train.py script without layer skipping
    python3 train.py \
        --model mobilenetv3 \
        --dataset ${DATASET} \
        --n_classes ${N_CLASSES} \
        --device ${DEVICE} \
        --supernet_path ${SUPERNET_PATH} \
        --model_path ${SUBNET_PATH} \
        --output_path ${output_dir} \
        --batch_size ${BATCH_SIZE} \
        --n_workers ${N_WORKERS} \
        --res ${RESOLUTION} \
        --epochs ${TRAINING_EPOCHS} \
        --val_split ${VAL_SPLIT} \
        --save \
        --eval_test
    
    echo "Completed training baseline model"
    echo "Results saved to: ${output_dir}"
}

# Train all models sequentially
echo "Starting gate type comparison experiment..."

# # 1. Stable gate (simple gate)
train_with_gate_type "stable"

# # 2. Conv gate
train_with_gate_type "conv"

# # 3. Attention gate
train_with_gate_type "attention"

# 4. No gate baseline
train_baseline

echo ""
echo "=================================================="
echo "All training completed!"
echo "=================================================="
echo "Results summary:"
echo "  Stable gate:    ${BASE_OUTPUT_DIR}/stable_gate"
echo "  Conv gate:      ${BASE_OUTPUT_DIR}/conv_gate"
echo "  Attention gate: ${BASE_OUTPUT_DIR}/attention_gate"
echo "  No gate:        ${BASE_OUTPUT_DIR}/no_gate"
echo ""
echo "To analyze results, check the net_*.stats files in each directory"
echo "=================================================="

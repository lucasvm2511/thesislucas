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

# Optional: Generate comparison summary
echo ""
echo "Generating comparison summary..."
python3 - <<EOF
import json
import os

base_dir = "${BASE_OUTPUT_DIR}"
gate_types = ["stable_gate", "conv_gate", "attention_gate", "no_gate"]

print("\n" + "="*80)
print("GATE TYPE COMPARISON RESULTS")
print("="*80)

results = {}
for gate_type in gate_types:
    stats_files = [f for f in os.listdir(os.path.join(base_dir, gate_type)) if f.endswith('.stats')]
    if stats_files:
        stats_path = os.path.join(base_dir, gate_type, stats_files[0])
        with open(stats_path, 'r') as f:
            results[gate_type] = json.load(f)

# Print comparison table
print("\n{:<20} {:<15} {:<15} {:<15} {:<15}".format(
    "Gate Type", "Top-1 Error (%)", "Avg MACs (M)", "Params (M)", "MAC Savings (%)"))
print("-" * 80)

for gate_type in gate_types:
    if gate_type in results:
        r = results[gate_type]
        top1 = r.get('top1', 'N/A')
        avg_macs = r.get('avg_macs', r.get('macs', 'N/A'))
        params = r.get('params', 'N/A')
        
        # Calculate MAC savings if available
        if gate_type != 'no_gate' and 'macs' in r and 'avg_macs' in r:
            baseline_macs = r['macs']
            actual_macs = r['avg_macs']
            savings = ((baseline_macs - actual_macs) / baseline_macs) * 100
        else:
            savings = 0.0
        
        print("{:<20} {:<15} {:<15} {:<15} {:<15.2f}".format(
            gate_type.replace('_', ' ').title(),
            f"{top1:.2f}" if isinstance(top1, (int, float)) else top1,
            f"{avg_macs:.2f}" if isinstance(avg_macs, (int, float)) else avg_macs,
            f"{params/1e6:.2f}" if isinstance(params, (int, float)) else params,
            savings
        ))

print("="*80)
print("\nComparison complete! See individual directories for detailed logs.")
EOF

echo ""
echo "Script completed successfully!"

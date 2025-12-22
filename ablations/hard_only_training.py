"""
Ablation Study: Compare Hard-Only vs Alternating Soft/Hard Training
=====================================================================
This script demonstrates that training with only hard gates (no soft gates) 
leads to poor performance compared to the alternating soft/hard training strategy.

The script:
1. Loads an average subnet configuration
2. Trains with ONLY hard gates for all epochs
3. Trains with ALTERNATING soft/hard gates for all epochs
4. Compares the results of both approaches
"""

import json
import logging
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import gc
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from train_utils import get_data_loaders, initialize_seed
from utils import get_network_search
from LayerSkipping.utils_ls import get_skipping_mobilenetv3

def train_hard_only(train_loader, val_loader, backbone, epochs, res, device, optimizer, criterion, logging):
    """Train with ONLY hard gates (no soft gates)"""
    
    best_accuracy = 0.0
    best_epoch = -1
    best_mac_savings = 0.0
    
    logging.info(f"Starting HARD-ONLY training for {epochs} epochs...")
    logging.info("Gate strategy: HARD gates for ALL epochs (no soft gates)")
    logging.info(f"Number of gates: {len(backbone.gates)}")
    if len(backbone.gates) > 0:
        logging.info(f"Per-gate target sparsities: {backbone.target_sparsities.tolist()}")
    
    epoch_results = []
    
    for epoch in range(epochs):
        # Training phase - ALWAYS use hard gates
        use_soft_gates = False
        backbone.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        sparsity_losses = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [HARD]")
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            # Forward pass - ALWAYS HARD
            output, aux = backbone(data, hard=True)
            
            # Task loss
            task_loss = criterion(output, target)
            
            # Sparsity loss (per-gate with individual targets)
            sparsity_loss = 0.0
            if aux and "gate_values" in aux and aux["gate_values"] is not None:
                # Calculate sparsity weight
                ramp_progress = min(1.0, epoch / 10.0)
                current_sparsity_weight = 0.1 + 0.9 * ramp_progress
                
                gate_values = aux["gate_values"]  # Shape: (batch, num_gates)
                
                # Ensure target_sparsities is on the same device
                if not hasattr(backbone, '_target_sparsities_device_moved'):
                    backbone.target_sparsities = backbone.target_sparsities.to(device)
                    backbone._target_sparsities_device_moved = True
                
                # Per-gate sparsity loss using individual targets
                for gate_idx in range(gate_values.shape[1]):
                    gate_open_prob = torch.mean(gate_values[:, gate_idx])  # Fraction gate is OPEN
                    target_sparsity = backbone.target_sparsities[gate_idx]  # Target fraction CLOSED
                    target_open_prob = 1.0 - target_sparsity  # Convert to target OPEN probability
                    sparsity_diff = gate_open_prob - target_open_prob
                    sparsity_loss += current_sparsity_weight * (sparsity_diff ** 2)
                
                # Add diversity loss to prevent gate collapse
                gate_variance = gate_values.var(dim=1).mean()  # Variance across gates per sample
                diversity_loss = max(0, 0.01 - gate_variance)
                sparsity_loss += current_sparsity_weight * diversity_loss
            
            # Combine losses
            total_loss = task_loss + sparsity_loss
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(backbone.parameters(), 1.0)
            optimizer.step()
            
            # Track statistics
            train_loss += total_loss.item()
            sparsity_losses += (sparsity_loss if isinstance(sparsity_loss, (int, float)) else sparsity_loss.item())
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{task_loss.item():.4f}',
                'acc': f'{100. * train_correct / train_total:.2f}%'
            })
        
        # Calculate training statistics
        epoch_loss = train_loss / len(train_loader)
        epoch_acc = 100. * train_correct / train_total
        epoch_sparsity_loss = sparsity_losses / len(train_loader)
        
        # Validation phase
        backbone.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output, aux = backbone(data, hard=True)
                
                if target.dim() == 0:
                    target = target.unsqueeze(0)
                batch_size = target.size(0)
                val_loss += criterion(output, target).item() * batch_size
                _, predicted = output.max(1)
                val_total += batch_size
                val_correct += predicted.eq(target).sum().item()
        
        # Calculate validation statistics
        val_loss /= val_total
        val_acc = 100. * val_correct / val_total
        
        # Calculate efficiency metrics
        all_gate_decisions = []
        backbone.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                if batch_idx >= 3:
                    break
                data = data.to(device, non_blocking=True)
                _, aux_val = backbone(data, hard=True)
                if aux_val and "gate_decisions" in aux_val and aux_val["gate_decisions"] is not None:
                    all_gate_decisions.append(aux_val["gate_decisions"])
        
        if all_gate_decisions:
            all_decisions = torch.cat(all_gate_decisions, dim=0)
            mac_ratio, sparsity_rate, baseline_macs, real_gated_macs = backbone.calculate_macs_accurate(
                all_decisions, input_size=(3, res, res), device=device
            )
            efficiency_percent = (1 - mac_ratio) * 100
            avg_skip_prob = sparsity_rate
        else:
            sparsity_rate = 0.0
            efficiency_percent = 0.0
            avg_skip_prob = 0.0
        
        # Log epoch results
        logging.info(f'Epoch {epoch+1}/{epochs}: Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Sparsity Loss: {epoch_sparsity_loss:.4f}, Gate Mode: HARD')
        if hasattr(backbone, 'gates') and len(backbone.gates) > 0:
            logging.info(f'  Gates: Sparsity={sparsity_rate:.2%}, Est. MAC Savings={efficiency_percent:.1f}%')
            
            if sparsity_rate > 0.8:
                logging.info(f"  ⚠️  HIGH SPARSITY WARNING: {sparsity_rate:.1%} - Gates may be collapsing!")
            if val_acc < epoch_acc - 20:
                logging.info(f"  ⚠️  OVERFITTING WARNING: Train-Val gap = {epoch_acc - val_acc:.1f}%")
        
        # Track best accuracy
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_epoch = epoch + 1
            best_mac_savings = efficiency_percent
        
        # Store epoch results
        epoch_results.append({
            'epoch': epoch + 1,
            'train_loss': epoch_loss,
            'train_acc': epoch_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'sparsity': sparsity_rate,
            'mac_savings': efficiency_percent,
            'gate_mode': 'HARD'
        })
    
    logging.info(f"\nHARD-ONLY training completed!")
    logging.info(f"Best validation accuracy: {best_accuracy:.2f}% (Epoch {best_epoch})")
    logging.info(f"MAC savings at best epoch: {best_mac_savings:.1f}%")
    
    return best_accuracy, best_mac_savings, epoch_results


def train_alternating(train_loader, val_loader, backbone, epochs, res, device, optimizer, criterion, logging):
    """Train with ALTERNATING soft/hard gates (standard approach)"""
    
    best_accuracy = 0.0
    best_epoch = -1
    best_mac_savings = 0.0
    
    logging.info(f"Starting ALTERNATING soft/hard training for {epochs} epochs...")
    logging.info("Gate strategy: Soft gates 50% of time (alternating with hard gates)")
    logging.info(f"Number of gates: {len(backbone.gates)}")
    if len(backbone.gates) > 0:
        logging.info(f"Per-gate target sparsities: {backbone.target_sparsities.tolist()}")
    
    epoch_results = []
    
    for epoch in range(epochs):
        # Determine gate mode - alternating strategy
        if epochs % 2 == 1:  # Odd number of epochs
            use_soft_gates = epoch % 2 == 0 and epoch < epochs - 1
        else:  # Even number of epochs
            use_soft_gates = epoch % 2 == 0
        
        # Training phase
        backbone.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        sparsity_losses = 0.0
        
        gate_mode_label = 'Soft' if use_soft_gates else 'Hard'
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [{gate_mode_label}]")
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            output, aux = backbone(data, hard=not use_soft_gates)
            
            # Task loss
            task_loss = criterion(output, target)
            
            # Sparsity loss (per-gate with individual targets)
            sparsity_loss = 0.0
            if aux and "gate_values" in aux and aux["gate_values"] is not None:
                # Calculate sparsity weight
                ramp_progress = min(1.0, epoch / 10.0)
                current_sparsity_weight = 0.1 + 0.9 * ramp_progress
                
                gate_values = aux["gate_values"]  # Shape: (batch, num_gates)
                
                # Ensure target_sparsities is on the same device
                if not hasattr(backbone, '_target_sparsities_device_moved'):
                    backbone.target_sparsities = backbone.target_sparsities.to(device)
                    backbone._target_sparsities_device_moved = True
                
                # Per-gate sparsity loss using individual targets
                for gate_idx in range(gate_values.shape[1]):
                    gate_open_prob = torch.mean(gate_values[:, gate_idx])  # Fraction gate is OPEN
                    target_sparsity = backbone.target_sparsities[gate_idx]  # Target fraction CLOSED
                    target_open_prob = 1.0 - target_sparsity  # Convert to target OPEN probability
                    sparsity_diff = gate_open_prob - target_open_prob
                    sparsity_loss += current_sparsity_weight * (sparsity_diff ** 2)
                
                # Add diversity loss to prevent gate collapse
                gate_variance = gate_values.var(dim=1).mean()  # Variance across gates per sample
                diversity_loss = max(0, 0.01 - gate_variance)
                sparsity_loss += current_sparsity_weight * diversity_loss
            
            # Combine losses
            total_loss = task_loss + sparsity_loss
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(backbone.parameters(), 1.0)
            optimizer.step()
            
            # Track statistics
            train_loss += total_loss.item()
            sparsity_losses += (sparsity_loss if isinstance(sparsity_loss, (int, float)) else sparsity_loss.item())
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{task_loss.item():.4f}',
                'acc': f'{100. * train_correct / train_total:.2f}%'
            })
        
        # Calculate training statistics
        epoch_loss = train_loss / len(train_loader)
        epoch_acc = 100. * train_correct / train_total
        epoch_sparsity_loss = sparsity_losses / len(train_loader)
        
        # Validation phase
        backbone.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output, aux = backbone(data, hard=True)
                
                if target.dim() == 0:
                    target = target.unsqueeze(0)
                batch_size = target.size(0)
                val_loss += criterion(output, target).item() * batch_size
                _, predicted = output.max(1)
                val_total += batch_size
                val_correct += predicted.eq(target).sum().item()
        
        # Calculate validation statistics
        val_loss /= val_total
        val_acc = 100. * val_correct / val_total
        
        # Calculate efficiency metrics
        all_gate_decisions = []
        backbone.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                if batch_idx >= 3:
                    break
                data = data.to(device, non_blocking=True)
                _, aux_val = backbone(data, hard=True)
                if aux_val and "gate_decisions" in aux_val and aux_val["gate_decisions"] is not None:
                    all_gate_decisions.append(aux_val["gate_decisions"])
        
        if all_gate_decisions:
            all_decisions = torch.cat(all_gate_decisions, dim=0)
            mac_ratio, sparsity_rate, baseline_macs, real_gated_macs = backbone.calculate_macs_accurate(
                all_decisions, input_size=(3, res, res), device=device
            )
            efficiency_percent = (1 - mac_ratio) * 100
        else:
            sparsity_rate = 0.0
            efficiency_percent = 0.0
        
        # Log epoch results
        gate_mode = 'Soft' if use_soft_gates else 'Hard'
        logging.info(f'Epoch {epoch+1}/{epochs}: Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Sparsity Loss: {epoch_sparsity_loss:.4f}, Gate Mode: {gate_mode}')
        if hasattr(backbone, 'gates') and len(backbone.gates) > 0:
            logging.info(f'  Gates: Sparsity={sparsity_rate:.2%}, Est. MAC Savings={efficiency_percent:.1f}%')
            
            if sparsity_rate > 0.8:
                logging.info(f"  ⚠️  HIGH SPARSITY WARNING: {sparsity_rate:.1%} - Gates may be collapsing!")
            if val_acc < epoch_acc - 20:
                logging.info(f"  ⚠️  OVERFITTING WARNING: Train-Val gap = {epoch_acc - val_acc:.1f}%")
        
        # Track best accuracy
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_epoch = epoch + 1
            best_mac_savings = efficiency_percent
        
        # Store epoch results
        epoch_results.append({
            'epoch': epoch + 1,
            'train_loss': epoch_loss,
            'train_acc': epoch_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'sparsity': sparsity_rate,
            'mac_savings': efficiency_percent,
            'gate_mode': gate_mode
        })
    
    logging.info(f"\nALTERNATING training completed!")
    logging.info(f"Best validation accuracy: {best_accuracy:.2f}% (Epoch {best_epoch})")
    logging.info(f"MAC savings at best epoch: {best_mac_savings:.1f}%")
    
    return best_accuracy, best_mac_savings, epoch_results


def main():
    parser = argparse.ArgumentParser(description='Ablation: Compare Hard-Only vs Alternating Training')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--data', type=str, default='datasets/cifar10', help='path to dataset')
    parser.add_argument('--n_classes', type=int, default=10, help='number of classes')
    parser.add_argument('--epochs', type=int, default=5, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='learning rate')
    parser.add_argument('--device', type=str, default='0', help='GPU device ID')
    parser.add_argument('--output_path', type=str, default='ablations/results/hard_only_training', 
                        help='path to save results')
    parser.add_argument('--supernet_path', type=str, 
                        default='NasSearchSpace/ofa/supernets/ofa_mbv3_d234_e346_k357_w1.0',
                        help='path to supernet weights')
    
    args = parser.parse_args()
    
    # Setup logging
    os.makedirs(args.output_path, exist_ok=True)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(os.path.join(args.output_path, 'comparison.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.info("=" * 80)
    logging.info("ABLATION STUDY: Hard-Only Gate Training")
    logging.info("=" * 80)
    logging.info(f"Dataset: {args.dataset}")
    logging.info(f"Epochs: {args.epochs}")
    logging.info(f"Batch size: {args.batch_size}")
    logging.info(f"Learning rate: {args.learning_rate}")
    
    # Setup device
    device = args.device
    use_cuda = False
    if torch.cuda.is_available() and device != 'cpu':
        device = f'cuda:{device}'
        logging.info(f"Running on GPU: {device}")
        use_cuda = True
    else:
        device = 'cpu'
        logging.info("Running on CPU")
    
    device = torch.device(device)
    
    if use_cuda:
        torch.cuda.empty_cache()
        gc.collect()
        gpu_id = int(str(device).split(':')[1]) if ':' in str(device) else 0
        free_mem = torch.cuda.mem_get_info(gpu_id)[0] / 1024**3
        total_mem = torch.cuda.mem_get_info(gpu_id)[1] / 1024**3
        logging.info(f"GPU {gpu_id} Memory: {free_mem:.2f}GB free out of {total_mem:.2f}GB total")
    
    initialize_seed(42, use_cuda)
    
    # Define multiple subnet configurations for robust comparison
    subnet_configs = {
        'small_efficient': {
            "ks": [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            "e": [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            "d": [2, 2, 2, 2, 2],
            "r": 88,
            "gate_hidden_sizes": [16, 16, 32, 32, 32, 16, 16, 16, 16, 16, 16, 16, 16, 32, 16, 16, 32, 32, 16, 16],
            "target_sparsities": [0.6, 0.7, 0.8, 0.0, 0.6, 0.7, 0.8, 0.7, 0.8, 0.0, 0.7, 0.6, 0.7, 0.0, 0.8, 0.6, 0.7, 0.7, 0.6, 0.0]
        },
        'average': {
            "ks": [5, 3, 5, 3, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            "e": [4, 3, 4, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            "d": [3, 2, 3, 3, 3],
            "r": 88,
            "gate_hidden_sizes": [32, 32, 32, 32, 64, 32, 32, 16, 16, 32, 16, 16, 32, 64, 32, 16, 64, 64, 32, 32],
            "target_sparsities": [0.5, 0.7, 0.8, 0.0, 0.5, 0.7, 0.8, 0.6, 0.8, 0.0, 0.7, 0.5, 0.6, 0.0, 0.8, 0.5, 0.6, 0.6, 0.5, 0.0]
        },
        'large_accurate': {
            "ks": [7, 5, 7, 5, 7, 7, 7, 7, 7, 7, 7, 7, 7],
            "e": [6, 4, 6, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6],
            "d": [4, 3, 4, 4, 4],
            "r": 88,
            "gate_hidden_sizes": [64, 64, 64, 64, 64, 64, 64, 32, 32, 64, 32, 32, 64, 64, 64, 32, 64, 64, 64, 64],
            "target_sparsities": [0.4, 0.6, 0.7, 0.0, 0.4, 0.6, 0.7, 0.5, 0.7, 0.0, 0.6, 0.4, 0.5, 0.0, 0.7, 0.4, 0.5, 0.5, 0.4, 0.0]
        }
    }
    
    # Load data once (shared for all experiments)
    logging.info("Loading dataset...")
    train_loader, val_loader, test_loader = get_data_loaders(
        dataset=args.dataset,
        batch_size=args.batch_size,
        threads=4,
        img_size=88,
        augmentation=True,
        val_split=0.1,
        eval_test=False
    )
    
    if val_loader is None:
        val_loader = test_loader
    
    logging.info(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Store results for all subnets
    all_subnet_results = {}
    
    # Run experiments on each subnet configuration
    for subnet_name, subnet_config in subnet_configs.items():
        logging.info("\n" + "=" * 80)
        logging.info(f"SUBNET: {subnet_name.upper()}")
        logging.info("=" * 80)
        
        # Save subnet configuration
        subnet_path = os.path.join(args.output_path, f'{subnet_name}_subnet.json')
        with open(subnet_path, 'w') as f:
            json.dump(subnet_config, f, indent=2)
        logging.info(f"Subnet configuration saved to: {subnet_path}")
        
        gate_hidden_sizes = subnet_config['gate_hidden_sizes']
        target_sparsities = subnet_config['target_sparsities']
        num_gates = sum(1 for ts in target_sparsities if ts != 0)
        
        logging.info(f"Architecture: ks={subnet_config['ks'][:5]}..., e={subnet_config['e'][:5]}..., d={subnet_config['d']}")
        logging.info(f"Number of gates to create: {num_gates}")
        
        results_comparison = {}
        
        # Run both training strategies
        for training_mode in ['hard_only', 'alternating']:
            logging.info("\n" + "=" * 80)
            logging.info(f"EXPERIMENT: {training_mode.upper()} TRAINING")
            logging.info("=" * 80)
            
            # Build fresh backbone for each experiment
            logging.info("Building backbone network...")
            backbone_fresh, res_fresh = get_network_search(
                model='mobilenetv3',
                subnet_path=subnet_path,
                supernet=args.supernet_path,
                n_classes=args.n_classes,
                pretrained=True,
                func_constr=False
            )
            
            res = res_fresh if res_fresh is not None else 88
            
            # Create skipping model
            backbone_fresh = get_skipping_mobilenetv3(
                subnet=backbone_fresh,
                subnet_path=subnet_path,
                res=res,
                n_classes=args.n_classes,
                gate_type='stable',
                enable_gates=True,
                gate_hidden_sizes=gate_hidden_sizes,
                target_sparsities=target_sparsities
            )
            
            backbone_fresh.to(device)
            
            # Calculate baseline MACs
            backbone_fresh.eval()
            if hasattr(backbone_fresh, 'gates') and len(backbone_fresh.gates) > 0:
                dummy_decisions = torch.ones(10, len(backbone_fresh.gates), device=device)
                _, _, baseline_macs, _ = backbone_fresh.calculate_macs_accurate(
                    dummy_decisions, input_size=(3, res, res), device=device
                )
                logging.info(f"Baseline MACs: {baseline_macs/1e6:.2f}M")
            
            # Setup optimizer
            optimizer = torch.optim.SGD(
                backbone_fresh.parameters(),
                lr=args.learning_rate,
                momentum=0.9,
                weight_decay=5e-5
            )
            
            # Train with selected strategy
            if training_mode == 'hard_only':
                best_accuracy, best_mac_savings, epoch_results = train_hard_only(
                    train_loader, val_loader, backbone_fresh, args.epochs, res, device, optimizer, criterion, logging
                )
            else:  # alternating
                best_accuracy, best_mac_savings, epoch_results = train_alternating(
                    train_loader, val_loader, backbone_fresh, args.epochs, res, device, optimizer, criterion, logging
                )
            
            # Store results
            results_comparison[training_mode] = {
                'best_val_accuracy': best_accuracy,
                'best_mac_savings': best_mac_savings,
                'epoch_results': epoch_results
            }
            
            # Clean up
            del backbone_fresh
            del optimizer
            if use_cuda:
                torch.cuda.empty_cache()
            gc.collect()
        
        # Store subnet results
        all_subnet_results[subnet_name] = {
            'hard_only': results_comparison['hard_only'],
            'alternating': results_comparison['alternating'],
            'accuracy_improvement': results_comparison['alternating']['best_val_accuracy'] - results_comparison['hard_only']['best_val_accuracy'],
            'mac_savings_diff': results_comparison['alternating']['best_mac_savings'] - results_comparison['hard_only']['best_mac_savings']
        }
        
        # Print subnet summary
        logging.info("\n" + "-" * 80)
        logging.info(f"SUBNET {subnet_name.upper()} SUMMARY")
        logging.info("-" * 80)
        logging.info(f"Hard-Only: Acc={results_comparison['hard_only']['best_val_accuracy']:.2f}%, MAC Savings={results_comparison['hard_only']['best_mac_savings']:.1f}%")
        logging.info(f"Alternating: Acc={results_comparison['alternating']['best_val_accuracy']:.2f}%, MAC Savings={results_comparison['alternating']['best_mac_savings']:.1f}%")
        logging.info(f"Improvement: Acc={all_subnet_results[subnet_name]['accuracy_improvement']:+.2f}%, MAC Savings={all_subnet_results[subnet_name]['mac_savings_diff']:+.1f}%")
        logging.info("-" * 80)
    
    # Save all comparison results
    final_results = {
        'dataset': args.dataset,
        'epochs': args.epochs,
        'subnets': all_subnet_results,
        'average_accuracy_improvement': np.mean([r['accuracy_improvement'] for r in all_subnet_results.values()]),
        'average_mac_savings_diff': np.mean([r['mac_savings_diff'] for r in all_subnet_results.values()])
    }
    
    results_path = os.path.join(args.output_path, 'comparison_results.json')
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Print final comparison summary
    logging.info("\n" + "=" * 80)
    logging.info("FINAL COMPARISON SUMMARY (ALL SUBNETS)")
    logging.info("=" * 80)
    
    for subnet_name, results in all_subnet_results.items():
        logging.info(f"\n{subnet_name.upper()}:")
        logging.info(f"  Hard-Only: Acc={results['hard_only']['best_val_accuracy']:.2f}%, MAC Savings={results['hard_only']['best_mac_savings']:.1f}%")
        logging.info(f"  Alternating: Acc={results['alternating']['best_val_accuracy']:.2f}%, MAC Savings={results['alternating']['best_mac_savings']:.1f}%")
        logging.info(f"  Δ Accuracy: {results['accuracy_improvement']:+.2f}%, Δ MAC Savings: {results['mac_savings_diff']:+.1f}%")
    
    logging.info(f"\nAverage Accuracy Improvement: {final_results['average_accuracy_improvement']:+.2f}%")
    logging.info(f"Average MAC Savings Difference: {final_results['average_mac_savings_diff']:+.1f}%")
    
    if final_results['average_accuracy_improvement'] > 0:
        logging.info(f"\n✓ Alternating soft/hard training is consistently BETTER in accuracy")
        if final_results['average_mac_savings_diff'] > 0:
            logging.info(f"✓ Alternating also achieves BETTER MAC savings")
        else:
            logging.info(f"⚠ However, hard-only achieves slightly better MAC savings")
    else:
        logging.info(f"\n✗ Results are unexpected - hard-only performed better in accuracy")
    
    logging.info("=" * 80)
    logging.info(f"Detailed results saved to: {results_path}")


if __name__ == "__main__":
    main()

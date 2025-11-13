"""
Analyze skipping gate usage for a trained skipping-MobileNetV3 model.

Produces per-gate open-probability, per-gate hard-open fraction, per-class gate averages,
and optionally computes MAC usage using calculate_macs_accurate (if available).

Usage (example):
    python3 tools/analyze_gates.py \
        --model_path /home/lvmidden/CNAS/results/cifar100-baseline/final/net-trade-off_0/ckpt.pth \
        --subnet /home/lvmidden/CNAS/results/cifar100-baseline/final/net-trade-off_0/net.subnet \
        --supernet_path ./NasSearchSpace/ofa/supernets/ofa_mbv3_d234_e346_k357_w1.0 \
        --dataset cifar100 \
        --n_classes 100 \
        --data /home/lvmidden/CNAS/data \
        --device cuda:0 \
        --output_dir /home/lvmidden/CNAS/results/cifar100-baseline/final/net-trade-off_0/

The script is intentionally conservative when loading checkpoints (handles several common formats).
"""

import argparse
import json
import os
from pathlib import Path
import copy
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision

import sys

# allow importing project modules
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / 'EarlyExits'))

from utils import get_network_search, get_net_info
from train_utils import get_data_loaders
from EarlyExits.utils_ee import get_skipping_mobilenetv3


def get_class_names(dataset):
    """Return class names for the given dataset."""
    if dataset == 'cifar100':
        # Official CIFAR-100 fine label list in correct index order
        return [
            'apple', 'aquarium_fish', 'baby', 'bear', 'beaver',
            'bed', 'bee', 'beetle', 'bicycle', 'bottle',
            'bowl', 'boy', 'bridge', 'bus', 'butterfly',
            'camel', 'can', 'castle', 'caterpillar', 'cattle',
            'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach',
            'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
            'dolphin', 'elephant', 'flatfish', 'forest', 'fox',
            'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
            'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard',
            'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
            'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid',
            'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
            'plain', 'plate', 'poppy', 'porcupine', 'possum',
            'rabbit', 'raccoon', 'ray', 'road', 'rocket',
            'rose', 'sea', 'seal', 'shark', 'shrew',
            'skunk', 'skyscraper', 'snail', 'snake', 'spider',
            'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
            'tank', 'telephone', 'television', 'tiger', 'tractor',
            'train', 'trout', 'tulip', 'turtle', 'wardrobe',
            'whale', 'willow_tree', 'wolf', 'woman', 'worm'
        ]
    elif dataset == 'cifar10':
        return ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']
    else:
        return None



def robust_load_state(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)

    # common keys: 'model_state_dict', 'state_dict', 'model' or a plain state_dict
    state = None
    if isinstance(ckpt, dict):
        if 'model_state_dict' in ckpt:
            state = ckpt['model_state_dict']
        elif 'state_dict' in ckpt:
            state = ckpt['state_dict']
        elif 'state' in ckpt and isinstance(ckpt['state'], dict):
            state = ckpt['state']
        else:
            # maybe whole checkpoint is a state dict
            state = ckpt
    else:
        state = ckpt

    # Some saved models have 'module.' prefix (DataParallel)
    try:
        model.load_state_dict(state)
        return True
    except Exception:
        # try stripping/adding 'module.' prefix
        new_state = {}
        for k, v in state.items():
            if k.startswith('module.'):
                new_state[k[len('module.'):]] = v
            else:
                new_state['module.' + k] = v

        try:
            model.load_state_dict(new_state)
            return True
        except Exception as e:
            print('Failed to load checkpoint into model:', e)
            return False


def accumulate_gate_stats(gate_stats, activation_rates, targets):
    """activation_rates: list of tensors (B,1) for each gate; targets: (B,)"""
    B = targets.size(0)
    targets_np = targets.cpu().numpy()

    # convert each activation to numpy (B,)
    acts = [act.squeeze().detach().cpu().numpy() for act in activation_rates]

    num_gates = len(acts)
    for g in range(num_gates):
        key = f'gate_{g}'
        if key not in gate_stats:
            gate_stats[key] = {'probs': [], 'hard': [], 'per_class': {}}

        probs = acts[g]
        # hard decisions (threshold 0.5)
        hard = (probs > 0.5).astype(np.float32)

        gate_stats[key]['probs'].extend(probs.tolist())
        gate_stats[key]['hard'].extend(hard.tolist())

        # per-class accumulation
        for i in range(B):
            c = int(targets_np[i])
            gate_stats[key]['per_class'].setdefault(c, [])
            gate_stats[key]['per_class'][c].append(float(probs[i]) )


def summarize_gate_stats(gate_stats):
    summary = {'gates': {}, 'classes': {}}

    # per-gate summary
    for key, v in gate_stats.items():
        probs = np.array(v['probs']) if v['probs'] else np.array([])
        hard = np.array(v['hard']) if v['hard'] else np.array([])

        mean_prob = float(np.mean(probs)) if probs.size else None
        hard_frac = float(np.mean(hard)) if hard.size else None

        # per-class averages
        per_class = {int(c): float(np.mean(vals)) for c, vals in v['per_class'].items() if len(vals) > 0}

        summary['gates'][key] = {
            'mean_prob': mean_prob,
            'hard_fraction_open': hard_frac,
            'num_samples': int(len(probs)),
            'per_class_mean': per_class
        }

        # aggregate per-class into global class summary
        for c, val in per_class.items():
            summary['classes'].setdefault(c, {})
            summary['classes'][c][key] = val

    return summary


def save_sample_images(samples, images, class_names, output_dir, prefix, num_to_save=10):
    """Save sample images with their gate statistics."""
    output_dir = Path(output_dir)
    
    for rank, sample in enumerate(samples[:num_to_save], 1):
        idx = sample['sample_idx']
        if idx >= len(images):
            continue
            
        img_tensor = images[idx]
        class_id = sample['class_id']
        class_name = class_names[class_id] if class_names else f"class_{class_id}"
        gate_open_rate = sample['gate_open_rate']
        avg_prob = sample['avg_gate_prob']
        num_open = sample['num_gates_open']
        total_gates = len(sample['gate_probs'])
        
        # Create figure with image and gate statistics
        fig, (ax_img, ax_gates) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Denormalize image (assuming ImageNet normalization)
        img = img_tensor.permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = np.clip(img, 0, 1)
        
        # Display image
        ax_img.imshow(img)
        ax_img.set_title(f'{class_name} (Class {class_id})\nGate Open Rate: {gate_open_rate:.1%}\n({num_open}/{total_gates} gates)', fontsize=12)
        ax_img.axis('off')
        
        # Display gate probabilities
        gate_probs = sample['gate_probs']
        gate_indices = range(len(gate_probs))
        colors = ['green' if p > 0.5 else 'red' for p in gate_probs]
        ax_gates.bar(gate_indices, gate_probs, color=colors, alpha=0.7)
        ax_gates.axhline(y=0.5, color='black', linestyle='--', linewidth=1)
        ax_gates.set_xlabel('Gate Index', fontsize=10)
        ax_gates.set_ylabel('Gate Probability', fontsize=10)
        ax_gates.set_title(f'Gate Probabilities\nOpen Rate: {gate_open_rate:.1%} | Avg Prob: {avg_prob:.3f}', fontsize=12)
        ax_gates.set_ylim(0, 1)
        ax_gates.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure - use gate_open_rate in filename
        filename = f'{prefix}_rank{rank:02d}_sample{idx:05d}_class{class_id:03d}_{class_name}_openrate{gate_open_rate:.3f}.png'
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  Saved: {filename}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Checkpoint file (ckpt.pth)')
    parser.add_argument('--subnet', type=str, required=True, help='Path to net.subnet (json)')
    parser.add_argument('--supernet_path', type=str, default='./NasSearchSpace/ofa/supernets/ofa_mbv3_d234_e346_k357_w1.0')
    parser.add_argument('--dataset', type=str, default='cifar100')
    parser.add_argument('--n_classes', type=int, default=100)
    parser.add_argument('--data', type=str, default='~/datasets')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_batches', type=int, default=-1, help='Limit number of batches for a quick run')
    parser.add_argument('--output_dir', type=str, default='.')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu')

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print('Loading subnet from', args.subnet)
    subnet, res = get_network_search('mobilenetv3', args.subnet, n_classes=args.n_classes, supernet=args.supernet_path, pretrained=False)

    print('Creating skipping model...')
    model = get_skipping_mobilenetv3(subnet, args.subnet, res or 32, args.n_classes, enable_gates=True)

    # move to device
    model = model.to(device)

    # load checkpoint
    if os.path.exists(args.model_path):
        print('Loading checkpoint:', args.model_path)
        ok = robust_load_state(model, args.model_path, device)
        if not ok:
            print('Warning: checkpoint could not be fully loaded. Proceeding with available weights.')
    else:
        print('Checkpoint not found, proceeding with model weights as initialized')

    model.eval()

    # dataloaders
    print('Preparing data loaders for', args.dataset)
    _, _, test_loader = get_data_loaders(dataset=args.dataset, batch_size=args.batch_size, threads=4, img_size=res or 32, augmentation=False, val_split=0, eval_test=True)
    if test_loader is None:
        raise RuntimeError('Test loader could not be created')

    gate_stats = {}
    all_gate_decisions = []
    sample_gate_stats = []  # Track per-sample statistics
    sample_images = []  # Store sample images

    max_batches = args.max_batches
    sample_idx = 0
    with torch.no_grad():
        for i, (images, targets) in enumerate(test_loader):
            if max_batches > 0 and i >= max_batches:
                break

            images = images.to(device)
            targets = targets.to(device)

            # SkippingMobileNetV3 returns (logits, aux) tuple
            out = model(images, hard=True)

            # model may return (output, aux_dict) or a single tensor
            if isinstance(out, tuple) or isinstance(out, list):
                logits, aux = out[0], out[1]
                # Extract gate_probs from aux dict if available
                if isinstance(aux, dict) and 'gate_probs' in aux and aux['gate_probs'] is not None:
                    activation_rates = [aux['gate_probs'][:, i:i+1] for i in range(aux['gate_probs'].shape[1])]
                else:
                    activation_rates = []
            else:
                # no gates
                activation_rates = []

            if activation_rates:
                # activation_rates is list of (B,1) tensors
                accumulate_gate_stats(gate_stats, activation_rates, targets)

                # store hard decisions for MAC calculation
                batch_gate_decisions = torch.cat([act for act in activation_rates], dim=1)
                binary_decisions = (batch_gate_decisions > 0.5).float().cpu()
                all_gate_decisions.append(binary_decisions)
                
                # Track per-sample gate statistics and store images
                for b in range(images.size(0)):
                    sample_probs = batch_gate_decisions[b].detach().cpu().numpy()
                    num_gates_open = int(np.sum(sample_probs > 0.5))
                    total_gates = len(sample_probs)
                    sample_gate_stats.append({
                        'sample_idx': sample_idx,
                        'class_id': int(targets[b].item()),
                        'gate_probs': sample_probs,
                        'avg_gate_prob': float(np.mean(sample_probs)),
                        'num_gates_open': num_gates_open,
                        'gate_open_rate': float(num_gates_open / total_gates)  # Key metric!
                    })
                    # Store image (keep on CPU)
                    sample_images.append(images[b].cpu())
                    sample_idx += 1

    # summarize
    summary = summarize_gate_stats(gate_stats)

    # Compute MAC usage using model's built-in method if available and we collected gate decisions
    if len(all_gate_decisions) > 0 and hasattr(model, 'calculate_macs_accurate'):
        combined = torch.cat(all_gate_decisions, dim=0)
        try:
            print('\nCalculating MACs...')
            mac_usage_ratio, sparsity_rate, baseline_macs, gated_macs = model.calculate_macs_accurate(
                combined, input_size=(3, res or 32, res or 32), device=device
            )
            summary['macs'] = {
                'mac_usage_ratio': float(mac_usage_ratio),
                'sparsity_rate': float(sparsity_rate),
                'baseline_macs': int(baseline_macs),
                'gated_macs': int(gated_macs),
                'mac_savings_percent': float((1 - mac_usage_ratio) * 100)
            }
            print(f'\nMAC Analysis:')
            print(f'  Baseline MACs: {baseline_macs:,}')
            print(f'  Gated MACs: {gated_macs:,}')
            print(f'  MAC Usage Ratio: {mac_usage_ratio:.3f}')
            print(f'  MAC Savings: {(1 - mac_usage_ratio) * 100:.1f}%')
            print(f'  Sparsity Rate: {sparsity_rate:.1%}')
        except Exception as e:
            print(f'Failed to compute MAC usage: {e}')
            import traceback
            traceback.print_exc()

    # save summary
    out_json = outdir / 'gate_usage_summary.json'
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)

    print('\nSaved summary to', out_json)
    print('Per-gate summary (top-level):')
    for k, v in summary['gates'].items():
        print(f" {k}: mean_prob={v['mean_prob']:.3f} hard_open_frac={v['hard_fraction_open']:.3f} samples={v['num_samples']}")
    
    # Per-class analysis
    print('\n' + '='*80)
    print('PER-CLASS GATE ANALYSIS')
    print('='*80)
    
    if 'classes' in summary and len(summary['classes']) > 0:
        # Build per-class gate matrix
        classes_data = summary['classes']
        gates_data = summary['gates']
        num_gates = len(gates_data)
        class_ids = sorted([int(c) for c in classes_data.keys()])
        
        gate_matrix = np.zeros((len(class_ids), num_gates))
        
        for cls_idx, cls_id in enumerate(class_ids):
            for gate_idx in range(num_gates):
                gate_key = f'gate_{gate_idx}'
                if cls_id in classes_data and gate_key in classes_data[cls_id]:
                    gate_matrix[cls_idx, gate_idx] = classes_data[cls_id][gate_key]
        
        # Compute per-class average gate probability
        class_avg_probs = gate_matrix.mean(axis=1)
        
        # Find classes with highest/lowest gate probabilities
        top_5_idx = np.argsort(class_avg_probs)[-5:][::-1]
        bottom_5_idx = np.argsort(class_avg_probs)[:5]
        
        print('\nClasses with HIGHEST avg gate open probs (most computation):')
        for rank, idx in enumerate(top_5_idx, 1):
            cls_id = class_ids[idx]
            avg_prob = class_avg_probs[idx]
            print(f"  {rank}. Class {cls_id:3d}: {avg_prob:.3f}")
        
        print('\nClasses with LOWEST avg gate open probs (most skipping):')
        for rank, idx in enumerate(bottom_5_idx, 1):
            cls_id = class_ids[idx]
            avg_prob = class_avg_probs[idx]
            print(f"  {rank}. Class {cls_id:3d}: {avg_prob:.3f}")
        
        # Analyze gate variance across classes (discriminative power)
        gate_variances = gate_matrix.var(axis=0)
        most_discriminative = np.argsort(gate_variances)[-5:][::-1]
        
        print('\nMost discriminative gates (highest variance across classes):')
        for rank, gate_idx in enumerate(most_discriminative, 1):
            variance = gate_variances[gate_idx]
            mean_prob = gates_data[f'gate_{gate_idx}']['mean_prob']
            print(f"  {rank}. Gate {gate_idx}: variance={variance:.4f}, mean={mean_prob:.3f}")
        
        print('\nPer-gate statistics across all classes:')
        for gate_idx in range(num_gates):
            gate_probs = gate_matrix[:, gate_idx]
            min_class = class_ids[gate_probs.argmin()]
            max_class = class_ids[gate_probs.argmax()]
            print(f"  Gate {gate_idx}: mean={gate_probs.mean():.3f}, std={gate_probs.std():.3f}, " +
                  f"min={gate_probs.min():.3f} (class {min_class}), max={gate_probs.max():.3f} (class {max_class})")
    
    # Sample-level analysis: show examples that skip most/least gates
    print('\n' + '='*80)
    print('SAMPLE-LEVEL GATE ANALYSIS')
    print('='*80)
    
    if len(sample_gate_stats) > 0:
        class_names = get_class_names(args.dataset)
        
        # Sort by gate open rate (not avg gate prob!)
        sorted_samples = sorted(sample_gate_stats, key=lambda x: x['gate_open_rate'])
        
        print('\nSamples that SKIP MOST gates (lowest gate open rate):')
        for rank, sample in enumerate(sorted_samples[:10], 1):
            class_id = sample['class_id']
            class_name = class_names[class_id] if class_names else f"Class {class_id}"
            print(f"  {rank:2d}. Sample {sample['sample_idx']:5d} | Class {class_id:3d} ({class_name:20s}) | " +
                  f"Gate open rate: {sample['gate_open_rate']:.3f} | Gates open: {sample['num_gates_open']}/{len(sample['gate_probs'])} | Avg prob: {sample['avg_gate_prob']:.3f}")
        
        print('\nSamples that SKIP LEAST gates (highest gate open rate):')
        for rank, sample in enumerate(sorted_samples[-10:][::-1], 1):
            class_id = sample['class_id']
            class_name = class_names[class_id] if class_names else f"Class {class_id}"
            print(f"  {rank:2d}. Sample {sample['sample_idx']:5d} | Class {class_id:3d} ({class_name:20s}) | " +
                  f"Gate open rate: {sample['gate_open_rate']:.3f} | Gates open: {sample['num_gates_open']}/{len(sample['gate_probs'])} | Avg prob: {sample['avg_gate_prob']:.3f}")
        
        # Statistics
        gate_open_rates = [s['gate_open_rate'] for s in sample_gate_stats]
        avg_probs = [s['avg_gate_prob'] for s in sample_gate_stats]
        print(f"\nOverall sample statistics:")
        print(f"  Total samples: {len(sample_gate_stats)}")
        print(f"  Mean gate open rate: {np.mean(gate_open_rates):.3f} (key metric)")
        print(f"  Std gate open rate: {np.std(gate_open_rates):.3f}")
        print(f"  Min gate open rate: {np.min(gate_open_rates):.3f}")
        print(f"  Max gate open rate: {np.max(gate_open_rates):.3f}")
        print(f"  Mean avg gate prob: {np.mean(avg_probs):.3f} (for reference)")
        print(f"  Std avg gate prob: {np.std(avg_probs):.3f}")
        
        # Save sample images
        print('\n' + '='*80)
        print('SAVING SAMPLE IMAGES')
        print('='*80)
        
        print(f"\nSaving images for samples that SKIP MOST gates to {outdir}...")
        save_sample_images(sorted_samples[:10], sample_images, class_names, outdir, 'high_skip', num_to_save=10)
        
        print(f"\nSaving images for samples that SKIP LEAST gates to {outdir}...")
        save_sample_images(sorted_samples[-10:][::-1], sample_images, class_names, outdir, 'low_skip', num_to_save=10)
        
        print(f"\nAll sample images saved to {outdir}")


if __name__ == '__main__':
    main()

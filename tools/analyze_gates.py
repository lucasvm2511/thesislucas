
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
sys.path.append(str(ROOT / 'LayerSkipping'))

from utils import get_network_search, get_net_info
from train_utils import get_data_loaders
from LayerSkipping.utils_ls import get_skipping_mobilenetv3


def get_class_names(dataset):
    if dataset == 'cifar100':
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

    state = None
    if isinstance(ckpt, dict):
        if 'model_state_dict' in ckpt:
            state = ckpt['model_state_dict']
        elif 'state_dict' in ckpt:
            state = ckpt['state_dict']
        elif 'state' in ckpt and isinstance(ckpt['state'], dict):
            state = ckpt['state']
        else:
            state = ckpt
    else:
        state = ckpt

    model.load_state_dict(state)
    return True


def accumulate_gate_stats(gate_stats, activation_rates, targets):
    B = targets.size(0)
    targets_np = targets.cpu().numpy()

    acts = [act.squeeze().detach().cpu().numpy() for act in activation_rates]

    num_gates = len(acts)
    for g in range(num_gates):
        key = f'gate_{g}'
        if key not in gate_stats:
            gate_stats[key] = {'probs': [], 'hard': [], 'per_class_probs': {}, 'per_class_hard': {}}

        probs = acts[g]
        hard = (probs > 0.5).astype(np.float32)

        gate_stats[key]['probs'].extend(probs.tolist())
        gate_stats[key]['hard'].extend(hard.tolist())

        for i in range(B):
            c = int(targets_np[i])
            gate_stats[key]['per_class_probs'].setdefault(c, [])
            gate_stats[key]['per_class_probs'][c].append(float(probs[i]))
            gate_stats[key]['per_class_hard'].setdefault(c, [])
            gate_stats[key]['per_class_hard'][c].append(float(hard[i]))


def summarize_gate_stats(gate_stats):
    summary = {'gates': {}, 'classes': {}}

    for key, v in gate_stats.items():
        probs = np.array(v['probs']) if v['probs'] else np.array([])
        hard = np.array(v['hard']) if v['hard'] else np.array([])

        mean_prob = float(np.mean(probs)) if probs.size else None
        hard_frac = float(np.mean(hard)) if hard.size else None

        per_class_hard = {int(c): float(np.mean(vals)) for c, vals in v['per_class_hard'].items() if len(vals) > 0}
        per_class_probs = {int(c): float(np.mean(vals)) for c, vals in v['per_class_probs'].items() if len(vals) > 0}

        summary['gates'][key] = {
            'mean_prob': mean_prob,
            'hard_fraction_open': hard_frac,
            'num_samples': int(len(probs)),
            'per_class_mean': per_class_probs,
            'per_class_hard': per_class_hard
        }

        for c, val in per_class_hard.items():
            summary['classes'].setdefault(c, {})
            summary['classes'][c][key] = val

    return summary


def generate_gate_usage_plots(summary, dataset, output_dir, subnet_path=None):
    output_dir = Path(output_dir)
    class_names = get_class_names(dataset)
    
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
    
    class_avg_probs = gate_matrix.mean(axis=1)
    num_extreme = min(10, len(class_ids) // 2)
    sorted_indices = np.argsort(class_avg_probs)
    extreme_indices = np.concatenate([sorted_indices[:num_extreme], sorted_indices[-num_extreme:]])
    extreme_classes = [class_ids[i] for i in extreme_indices]
    extreme_probs = [class_avg_probs[i] for i in extreme_indices]
    
    if class_names:
        extreme_labels = [f"{class_names[c]}" for c in extreme_classes]
    else:
        extreme_labels = [f"Class {c}" for c in extreme_classes]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    colors = ['#d62728'] * num_extreme + ['#2ca02c'] * num_extreme
    
    ax.barh(range(len(extreme_classes)), extreme_probs, color=colors, alpha=0.7)
    ax.set_yticks(range(len(extreme_classes)))
    ax.set_yticklabels(extreme_labels, fontsize=9)
    ax.set_xlabel('% Gates Open')
    ax.set_ylabel('Class')
    ax.set_title(f'Gate Usage Per Class: Top & Bottom {num_extreme}')
    ax.axhline(y=num_extreme - 0.5, color='black', linestyle='-', linewidth=2, alpha=0.5)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'avg_gate_usage_per_class.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('  Saved: avg_gate_usage_per_class.png')
    
    all_block_means = []
    all_block_variances = []
    block_labels = []
    block_colors_mean = []
    block_colors_var = []
    
    if subnet_path and os.path.exists(subnet_path):
        try:
            with open(subnet_path, 'r') as f:
                subnet_config = json.load(f)
            
            target_sparsities = subnet_config.get('target_sparsities', [])
            gate_idx = 0
            
            all_block_means.append(1.0)
            all_block_variances.append(0.0)
            block_labels.append('B0\n(no gate)')
            block_colors_mean.append('#90EE90')
            
            for block_idx, sparsity in enumerate(target_sparsities, start=1):
                if sparsity > 0:
                    if gate_idx < num_gates:
                        all_block_means.append(gate_matrix[:, gate_idx].mean())
                        all_block_variances.append(gate_matrix[:, gate_idx].var())
                        block_labels.append(f'B{block_idx}\n(G{gate_idx})')
                        block_colors_mean.append('#ff7f0e')
                        gate_idx += 1
                else:
                    all_block_means.append(1.0)
                    all_block_variances.append(0.0)
                    block_labels.append(f'B{block_idx}\n(no gate)')
                    block_colors_mean.append('#90EE90')
                    
        except Exception as e:
            print(f'Warning: Could not read subnet config: {e}')
            all_block_means = gate_matrix.mean(axis=0).tolist()
            all_block_variances = gate_matrix.var(axis=0).tolist()
            block_labels = [f'G{i}' for i in range(num_gates)]
            block_colors_mean = ['#ff7f0e'] * num_gates
    else:
        all_block_means = gate_matrix.mean(axis=0).tolist()
        all_block_variances = gate_matrix.var(axis=0).tolist()
        block_labels = [f'G{i}' for i in range(num_gates)]
        block_colors_mean = ['#ff7f0e'] * num_gates
    
    plt.figure(figsize=(12, 5))
    plt.bar(range(len(all_block_means)), all_block_means, color=block_colors_mean)
    plt.xticks(range(len(block_labels)), block_labels, fontsize=9)
    plt.xlabel('Block (Gate)')
    plt.ylabel('Mean Gate Open Probability')
    plt.title('Mean Gate Open Probability per Block')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'gate_statistics_per_gate.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('  Saved: gate_statistics_per_gate.png')
    
    plt.figure(figsize=(10, 8))
    gate_corr = np.corrcoef(gate_matrix.T)
    im = plt.imshow(gate_corr, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im, label='Correlation Coefficient')
    plt.xlabel('Gate Index', fontsize=12)
    plt.ylabel('Gate Index', fontsize=12)
    plt.title('Gate Correlation Matrix\n(How similarly do gates behave across classes?)', fontsize=14, fontweight='bold')
    
    for i in range(num_gates):
        for j in range(num_gates):
            text = plt.text(j, i, f'{gate_corr[i, j]:.2f}',
                          ha="center", va="center", color="black" if abs(gate_corr[i, j]) < 0.5 else "white",
                          fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'gate_correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('  Saved: gate_correlation_matrix.png')
    
    if 'per_class_accuracy' in summary:
        acc_data = summary['per_class_accuracy']
        gate_usage_vals = []
        accuracy_vals = []
        class_labels = []
        
        for cls_id in class_ids:
            if cls_id in acc_data:
                cls_idx = class_ids.index(cls_id)
                gate_usage_vals.append(class_avg_probs[cls_idx])
                accuracy_vals.append(acc_data[cls_id]['accuracy'])
                class_labels.append(class_names[cls_id] if class_names else f'{cls_id}')
        
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(gate_usage_vals, accuracy_vals, c=gate_usage_vals, 
                            cmap='RdYlGn', s=80, alpha=0.6)
        z = np.polyfit(gate_usage_vals, accuracy_vals, 1)
        p = np.poly1d(z)
        x_trend = np.array([min(gate_usage_vals), max(gate_usage_vals)])
        plt.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
        plt.colorbar(scatter, label='% Gates Open')
        plt.xlabel('% Gates Open')
        plt.ylabel('Accuracy')
        plt.title('Gate Usage vs Accuracy')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'gate_usage_vs_accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()
        print('  Saved: gate_usage_vs_accuracy.png')
    
    print(f'\nAll visualization plots saved to {output_dir}/')


def plot_skip_rate_vs_confidence(sample_gate_stats, output_dir):
    skip_rates = []
    confidences = []
    
    for sample in sample_gate_stats:
        if 'confidence' in sample:
            skip_rate = 1.0 - sample['gate_open_rate']
            skip_rates.append(skip_rate)
            confidences.append(sample['confidence'])
    
    if len(skip_rates) == 0:
        print('Warning: No confidence data available for skip rate vs confidence plot')
        return
    
    skip_rates = np.array(skip_rates)
    confidences = np.array(confidences)
    
    plt.figure(figsize=(10, 6))
    plt.hist2d(skip_rates, confidences, bins=40, cmap='YlOrRd')
    plt.colorbar(label='Density')
    correlation = np.corrcoef(skip_rates, confidences)[0, 1]
    plt.xlabel('Skip Rate')
    plt.ylabel('Confidence')
    plt.title(f'Skip Rate vs Confidence (corr={correlation:.3f})')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_dir / 'skip_rate_vs_confidence.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('  Saved: skip_rate_vs_confidence.png')
    print(f'  Correlation: {correlation:.3f}, Mean skip: {skip_rates.mean():.3f}')


def save_sample_images(samples, images, class_names, output_dir, prefix, num_to_save=10):
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
        
        fig, (ax_img, ax_gates) = plt.subplots(1, 2, figsize=(12, 4))
        
        img = img_tensor.permute(1, 2, 0).numpy()
        img = np.clip(img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)
        
        ax_img.imshow(img)
        ax_img.set_title(f'{class_name}\nGate Open Rate: {gate_open_rate:.1%} ({num_open}/{total_gates})')
        ax_img.axis('off')
        
        gate_probs = sample['gate_probs']
        gate_indices = range(len(gate_probs))
        colors = ['green' if p > 0.5 else 'red' for p in gate_probs]
        ax_gates.bar(gate_indices, gate_probs, color=colors, alpha=0.7)
        ax_gates.axhline(y=0.5, color='black', linestyle='--', linewidth=1)
        ax_gates.set_xlabel('Gate Index')
        ax_gates.set_ylabel('Probability')
        ax_gates.set_title(f'Gate Probabilities (avg={avg_prob:.3f})')
        ax_gates.set_ylim(0, 1)
        ax_gates.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        filename = f'{prefix}_rank{rank:02d}_{class_name}_rate{gate_open_rate:.3f}.png'
        plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()


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
    parser.add_argument('--output_dir', type=str, default='./results/analysis/gates', help='Directory to save gate analysis results')
    parser.add_argument('--gate_type', type=str, default='stable', choices=['stable', 'conv', 'attention'], help='Type of gate to use')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu')

    # Auto-generate unique output directory based on subnet path
    if args.output_dir == './results/analysis/gates':
        # Extract experiment name from subnet path (e.g., results/experiment_name/...)
        subnet_path = Path(args.subnet)
        exp_parts = [p for p in subnet_path.parts if p not in ['results', 'final', 'net.subnet']]
        if len(exp_parts) > 0:
            exp_name = exp_parts[0] if 'results' in subnet_path.parts else subnet_path.stem
            outdir = Path(args.output_dir) / exp_name
        else:
            outdir = Path(args.output_dir)
    else:
        outdir = Path(args.output_dir)
    
    outdir.mkdir(parents=True, exist_ok=True)
    print(f'Saving results to: {outdir}')

    print('Loading subnet from', args.subnet)
    subnet, res = get_network_search('mobilenetv3', args.subnet, n_classes=args.n_classes, supernet=args.supernet_path, pretrained=False)

    print('Creating skipping model...')
    model = get_skipping_mobilenetv3(subnet, args.subnet, res or 32, args.n_classes, enable_gates=True, gate_type=args.gate_type)

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
    sample_gate_stats = []
    sample_images = []
    per_class_correct = {}
    per_class_total = {}

    max_batches = args.max_batches
    sample_idx = 0
    with torch.no_grad():
        for i, (images, targets) in enumerate(test_loader):
            if max_batches > 0 and i >= max_batches:
                break

            images = images.to(device)
            targets = targets.to(device)

            out = model(images, hard=True)

            if isinstance(out, tuple) or isinstance(out, list):
                logits, aux = out[0], out[1]
                
                # Track per-class accuracy
                _, predicted = logits.max(1)
                for j in range(targets.size(0)):
                    cls = int(targets[j].item())
                    per_class_total[cls] = per_class_total.get(cls, 0) + 1
                    if predicted[j] == targets[j]:
                        per_class_correct[cls] = per_class_correct.get(cls, 0) + 1
                
                if isinstance(aux, dict) and 'gate_probs' in aux and aux['gate_probs'] is not None:
                    activation_rates = [aux['gate_probs'][:, i:i+1] for i in range(aux['gate_probs'].shape[1])]
                else:
                    activation_rates = []
            else:
                activation_rates = []

            if activation_rates:
                accumulate_gate_stats(gate_stats, activation_rates, targets)

                batch_gate_decisions = torch.cat([act for act in activation_rates], dim=1)
                binary_decisions = (batch_gate_decisions > 0.5).float().cpu()
                all_gate_decisions.append(binary_decisions)
                
                softmax_probs = torch.softmax(logits, dim=1)
                confidences = softmax_probs.max(dim=1)[0].detach().cpu().numpy()
                
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
                        'gate_open_rate': float(num_gates_open / total_gates),
                        'confidence': float(confidences[b])
                    })
                    sample_images.append(images[b].cpu())
                    sample_idx += 1

    summary = summarize_gate_stats(gate_stats)
    
    summary['per_class_accuracy'] = {
        int(cls): {
            'accuracy': per_class_correct.get(cls, 0) / per_class_total.get(cls, 1),
            'correct': per_class_correct.get(cls, 0),
            'total': per_class_total.get(cls, 0)
        }
        for cls in per_class_total.keys()
    }

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

    if 'classes' in summary and len(summary['classes']) > 0:
        print('\nGenerating visualizations...')
        generate_gate_usage_plots(summary, args.dataset, outdir, subnet_path=args.subnet)
    
    out_json = outdir / 'gate_usage_summary.json'
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\nSaved summary to {out_json}')
    
    if 'classes' in summary and len(summary['classes']) > 0:
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
    
    class_names = get_class_names(args.dataset)
    if len(sample_gate_stats) > 0:
        sorted_samples = sorted(sample_gate_stats, key=lambda x: x['gate_open_rate'])
        save_sample_images(sorted_samples[:10], sample_images, class_names, outdir, 'high_skip', num_to_save=10)
        save_sample_images(sorted_samples[-10:][::-1], sample_images, class_names, outdir, 'low_skip', num_to_save=10)


if __name__ == '__main__':
    main()

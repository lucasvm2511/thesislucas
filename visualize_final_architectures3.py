#!/usr/bin/env python3
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict
import argparse


def load_stats_file(filepath):
    """Load a net.stats JSON file and extract relevant metrics."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def normalize_macs_units(raw_macs):
    """Ensure MAC counts are expressed in millions for plotting."""
    if raw_macs is None:
        return None
    # Stats sometimes store raw MAC counts; convert those to millions.
    if raw_macs > 1e5:
        return raw_macs / 1e6
    return raw_macs


def extract_metrics(stats_data):
    """
    Extract MACs and accuracy from stats data.
    Handles different formats (baseline vs layer skipping).
    Note: top1 in stats is ERROR RATE, so we convert to accuracy.
    """
    if stats_data is None:
        return None, None
    
    # Check for different formats
    # Prefer avg_macs when available (captures expected MACs with early exits)
    if 'avg_macs' in stats_data and 'top1' in stats_data:
        macs = normalize_macs_units(stats_data['avg_macs'])
        # Convert error rate to accuracy
        accuracy = 100.0 - stats_data['top1']
    elif 'macs' in stats_data and 'top1' in stats_data:
        # Fallback to macs (e.g., baseline or backbone macs)
        macs = normalize_macs_units(stats_data['macs'])
        # Convert error rate to accuracy
        accuracy = 100.0 - stats_data['top1']
    else:
        return None, None
    
    return macs, accuracy


def scan_results_directory(results_dir, num_archs=30):
    """
    Scan the results directory for the last N architectures from the archive.
    Returns a dictionary organized by run type.
    """
    results_dir = Path(results_dir)
    all_results = defaultdict(list)
    
    # Find all iter_*.stats files (archive files)
    for stats_file in results_dir.glob('*/iter_*.stats'):
        # Extract run information from path
        parts = stats_file.parts
        
        # Find the run type (e.g., 'search_layerskippingextended_datasetcifar100_seed1')
        results_idx = parts.index('results')
        run_type = parts[results_idx + 1]
        
        if run_type in ['tempweg', 'old']:
            continue
        
        # Load stats file
        try:
            with open(stats_file, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading {stats_file}: {e}")
            continue
        
        # Extract archive
        if 'archive' not in data or not data['archive']:
            continue
        
        archive = data['archive']
        # Get last num_archs architectures
        last_archs = archive[-num_archs:] if len(archive) > num_archs else archive
        
        # Extract metrics from each architecture
        # Archive format: [[arch_dict, error_rate, macs], ...]
        for idx, entry in enumerate(last_archs):
            if len(entry) != 3:
                continue
            
            arch_dict, error_rate, macs = entry
            
            # Convert error rate to accuracy
            accuracy = 100.0 - error_rate
            
            # Normalize MACs
            macs = normalize_macs_units(macs)
            
            if macs is not None and accuracy is not None:
                all_results[run_type].append({
                    'macs': macs,
                    'accuracy': accuracy,
                    'arch_idx': len(archive) - len(last_archs) + idx,
                    'path': str(stats_file),
                    'raw_data': {'arch': arch_dict, 'top1_acc': accuracy, 'macs': macs}
                })
    
    return all_results


def format_dataset_label(dataset):
    """Convert dataset flag to a readable label for plot titles."""
    if dataset is None:
        return None
    dataset = dataset.strip()
    if not dataset:
        return None
    if dataset.lower() == 'all':
        return 'All Datasets'
    return dataset.upper()


def plot_comparison(all_results, output_dir=None, title="Archive Architectures: MACs vs Accuracy", dataset_label=None):
    """Create comparison plots for all runs."""
    
    if not all_results:
        print("No results found to plot!")
        return
    
    # Create figure with single plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color map for different runs
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
    
    # Plot all runs on same plot
    for idx, (run_name, results) in enumerate(sorted(all_results.items())):
        macs_vals = [r['macs'] for r in results]
        acc_vals = [r['accuracy'] for r in results]
        
        # Sort by MACs
        sorted_indices = np.argsort(macs_vals)
        macs_sorted = [macs_vals[i] for i in sorted_indices]
        acc_sorted = [acc_vals[i] for i in sorted_indices]
        
        # Clean run name for legend
        legend_name = run_name.replace('search_', '').replace('dataset', '').replace('_', ' ')
        
        ax.scatter(macs_sorted, acc_sorted, label=legend_name,
               color=colors[idx], s=60, alpha=0.7, edgecolors='none')
    
    ax.set_xlabel('MACs (Millions)', fontsize=12)
    ax.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
    dataset_suffix = format_dataset_label(dataset_label)
    plot_title = title if dataset_suffix is None else f"{title} – {dataset_suffix}"
    ax.set_title(plot_title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    if output_dir:
        output_path = Path(output_dir) / 'architecture_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")
        
        output_path_pdf = Path(output_dir) / 'architecture_comparison.pdf'
        plt.savefig(output_path_pdf, bbox_inches='tight')
        print(f"Saved plot to: {output_path_pdf}")
    
    plt.show()


def plot_individual_runs(all_results, output_dir=None, dataset_label=None):
    """Create individual plots for each run showing archive architectures."""
    
    for run_name, results in sorted(all_results.items()):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        macs_vals = [r['macs'] for r in results]
        acc_vals = [r['accuracy'] for r in results]
        
        # Sort by MACs
        sorted_indices = np.argsort(macs_vals)
        macs_sorted = [macs_vals[i] for i in sorted_indices]
        acc_sorted = [acc_vals[i] for i in sorted_indices]
        
        # Plot
        ax.scatter(macs_sorted, acc_sorted, s=80, alpha=0.6)
        
        ax.set_xlabel('MACs (Millions)', fontsize=12)
        ax.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
        
        clean_name = run_name.replace('search_', '').replace('dataset', '').replace('_', ' ')
        dataset_suffix = format_dataset_label(dataset_label)
        title_prefix = clean_name if dataset_suffix is None else f"{dataset_suffix} · {clean_name}"
        ax.set_title(f'{title_prefix}: Archive Architectures', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        if output_dir:
            safe_name = run_name.replace('/', '_')
            output_path = Path(output_dir) / f'{safe_name}_tradeoff.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to: {output_path}")
        
        plt.close()


def print_summary_table(all_results):
    """Print a summary table of all results."""
    print("\n" + "="*100)
    print("SUMMARY OF ARCHIVE ARCHITECTURES")
    print("="*100)
    
    for run_name, results in sorted(all_results.items()):
        print(f"\n{run_name}:")
        print(f"Total architectures: {len(results)}")
        
        # Show statistics
        macs_vals = [r['macs'] for r in results]
        acc_vals = [r['accuracy'] for r in results]
        
        print(f"MACs range: {min(macs_vals):.2f} - {max(macs_vals):.2f} M")
        print(f"Accuracy range: {min(acc_vals):.2f}% - {max(acc_vals):.2f}%")
        print(f"Mean accuracy: {np.mean(acc_vals):.2f}%")
    
    print("\n" + "="*100)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize archive architectures comparing MACs vs Accuracy'
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default='./results',
        help='Path to results directory (default: ./results)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./results/visualizations',
        help='Directory to save plots (default: ./results/visualizations)'
    )
    parser.add_argument(
        '--no_individual',
        action='store_true',
        help='Skip creating individual run plots'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['cifar10', 'cifar100', 'all'],
        default='all',
        help='Filter results by dataset: cifar10, cifar100, or all (default: all)'
    )
    parser.add_argument(
        '--num_archs',
        type=int,
        default=30,
        help='Number of architectures to load from the end of the archive (default: 30)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Scanning results directory: {args.results_dir}")
    print(f"Loading last {args.num_archs} architectures from each run's archive")
    all_results = scan_results_directory(args.results_dir, num_archs=args.num_archs)
    
    # Filter out old results if not requested
    all_results = {k: v for k, v in all_results.items() if not k.startswith('old/')}
    
    # Filter by dataset if specified
    if args.dataset == 'cifar10':
        all_results = {
            k: v for k, v in all_results.items() 
            if 'cifar10' in k.lower() and 'cifar100' not in k.lower()
        }
    elif args.dataset == 'cifar100':
        all_results = {
            k: v for k, v in all_results.items() 
            if 'cifar100' in k.lower()
        }
    
    if not all_results:
        print(f"No results found for dataset filter: {args.dataset}")
        return
    
    print(f"Found {len(all_results)} different runs:")
    for run_name, results in sorted(all_results.items()):
        print(f"  - {run_name}: {len(results)} architectures")
    
    # Print summary table
    print_summary_table(all_results)
    
    # Create comparison plots
    print("\nCreating comparison plots...")
    plot_comparison(all_results, output_dir, dataset_label=args.dataset)
    
    # Create individual plots
    if not args.no_individual:
        print("\nCreating individual run plots...")
        plot_individual_runs(all_results, output_dir, dataset_label=args.dataset)
    
    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == '__main__':
    main()

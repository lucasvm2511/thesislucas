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
        macs = stats_data['avg_macs']
        # Convert error rate to accuracy
        accuracy = 100.0 - stats_data['top1']
    elif 'macs' in stats_data and 'top1' in stats_data:
        # Fallback to macs (e.g., baseline or backbone macs)
        macs = stats_data['macs']
        # Convert error rate to accuracy
        accuracy = 100.0 - stats_data['top1']
    else:
        return None, None
    
    return macs, accuracy


def scan_results_directory(results_dir):
    """
    Scan the results directory for all final architectures.
    Returns a dictionary organized by run type.
    """
    results_dir = Path(results_dir)
    all_results = defaultdict(list)
    
    # Find all net.stats files in final directories
    for stats_file in results_dir.rglob('final/*/net.stats'):
        # Extract run information from path
        parts = stats_file.parts
        
        # Find the run type (e.g., 'search_layerskippingextended_datasetcifar100_seed1')
        results_idx = parts.index('results')
        run_type = parts[results_idx + 1]
        

        if run_type == 'tempweg':
            continue
        if run_type == 'old':
            continue
        
        # Get trade-off index
        tradeoff_dir = parts[-2]  # e.g., 'net-trade-off_0'
        
        # Load stats
        stats = load_stats_file(stats_file)
        macs, accuracy = extract_metrics(stats)
        
        if macs is not None and accuracy is not None:
            all_results[run_type].append({
                'macs': macs,
                'accuracy': accuracy,
                'tradeoff': tradeoff_dir,
                'path': str(stats_file),
                'raw_data': stats
            })
    
    return all_results


def plot_comparison(all_results, output_dir=None, title="Architecture Comparison: MACs vs Accuracy"):
    """Create comparison plots for all runs."""
    
    if not all_results:
        print("No results found to plot!")
        return
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Color map for different runs
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
    
    # Plot 1: All runs on same plot
    for idx, (run_name, results) in enumerate(sorted(all_results.items())):
        macs_vals = [r['macs'] for r in results]
        acc_vals = [r['accuracy'] for r in results]
        
        # Sort by MACs for connected line
        sorted_indices = np.argsort(macs_vals)
        macs_sorted = [macs_vals[i] for i in sorted_indices]
        acc_sorted = [acc_vals[i] for i in sorted_indices]
        
        # Clean run name for legend
        legend_name = run_name.replace('search_', '').replace('dataset', '').replace('_', ' ')
        
        ax1.plot(macs_sorted, acc_sorted, 'o-', label=legend_name, 
                color=colors[idx], markersize=8, linewidth=2, alpha=0.7)
    
    ax1.set_xlabel('MACs (Millions)', fontsize=12)
    ax1.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Pareto frontier comparison
    for idx, (run_name, results) in enumerate(sorted(all_results.items())):
        macs_vals = np.array([r['macs'] for r in results])
        acc_vals = np.array([r['accuracy'] for r in results])
        
        # Find Pareto frontier (maximize accuracy, minimize MACs)
        pareto_indices = []
        for i in range(len(results)):
            is_pareto = True
            for j in range(len(results)):
                if i != j:
                    # Check if j dominates i
                    if acc_vals[j] >= acc_vals[i] and macs_vals[j] <= macs_vals[i]:
                        if acc_vals[j] > acc_vals[i] or macs_vals[j] < macs_vals[i]:
                            is_pareto = False
                            break
            if is_pareto:
                pareto_indices.append(i)
        
        if pareto_indices:
            pareto_macs = macs_vals[pareto_indices]
            pareto_acc = acc_vals[pareto_indices]
            
            # Sort for line plot
            sorted_indices = np.argsort(pareto_macs)
            pareto_macs = pareto_macs[sorted_indices]
            pareto_acc = pareto_acc[sorted_indices]
            
            legend_name = run_name.replace('search_', '').replace('dataset', '').replace('_', ' ')
            ax2.plot(pareto_macs, pareto_acc, 'o-', label=legend_name,
                    color=colors[idx], markersize=10, linewidth=2.5, alpha=0.8)
    
    ax2.set_xlabel('MACs (Millions)', fontsize=12)
    ax2.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
    ax2.set_title('Pareto Frontier Comparison', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
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


def plot_individual_runs(all_results, output_dir=None):
    """Create individual plots for each run showing trade-off curve."""
    
    for run_name, results in sorted(all_results.items()):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        macs_vals = [r['macs'] for r in results]
        acc_vals = [r['accuracy'] for r in results]
        tradeoff_labels = [r['tradeoff'] for r in results]
        
        # Sort by MACs
        sorted_indices = np.argsort(macs_vals)
        macs_sorted = [macs_vals[i] for i in sorted_indices]
        acc_sorted = [acc_vals[i] for i in sorted_indices]
        labels_sorted = [tradeoff_labels[i] for i in sorted_indices]
        
        # Plot with labels
        ax.plot(macs_sorted, acc_sorted, 'o-', markersize=10, linewidth=2)
        
        # Add labels for each point
        for i, label in enumerate(labels_sorted):
            offset = 1 if i % 2 == 0 else -1
            ax.annotate(label.replace('net-trade-off_', ''), 
                       (macs_sorted[i], acc_sorted[i]),
                       textcoords="offset points", 
                       xytext=(0, offset*15),
                       ha='center',
                       fontsize=8,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
        
        ax.set_xlabel('MACs (Millions)', fontsize=12)
        ax.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
        
        clean_name = run_name.replace('search_', '').replace('dataset', '').replace('_', ' ')
        ax.set_title(f'{clean_name}: MACs vs Accuracy Trade-off', fontsize=14, fontweight='bold')
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
    print("SUMMARY OF FINAL ARCHITECTURES")
    print("="*100)
    
    for run_name, results in sorted(all_results.items()):
        print(f"\n{run_name}:")
        print(f"{'Trade-off':<15} {'MACs (M)':<15} {'Accuracy (%)':<15} {'Params':<15}")
        print("-" * 60)
        
        # Sort by MACs
        sorted_results = sorted(results, key=lambda x: x['macs'])
        
        for r in sorted_results:
            tradeoff = r['tradeoff'].replace('net-trade-off_', '')
            macs = r['macs']
            acc = r['accuracy']
            
            # Try to get params
            params = r['raw_data'].get('params', 'N/A')
            if isinstance(params, (int, float)):
                params_str = f"{params/1e6:.2f}M"
            else:
                params_str = str(params)
            
            print(f"{tradeoff:<15} {macs:<15.2f} {acc:<15.2f} {params_str:<15}")
    
    print("\n" + "="*100)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize final architectures comparing MACs vs Accuracy'
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
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Scanning results directory: {args.results_dir}")
    all_results = scan_results_directory(args.results_dir)
    
    # Filter out old results if not requested
    all_results = {k: v for k, v in all_results.items() if not k.startswith('old/')}
    
    if not all_results:
        print("No results found!")
        return
    
    print(f"Found {len(all_results)} different runs:")
    for run_name, results in sorted(all_results.items()):
        print(f"  - {run_name}: {len(results)} architectures")
    
    # Print summary table
    print_summary_table(all_results)
    
    # Create comparison plots
    print("\nCreating comparison plots...")
    plot_comparison(all_results, output_dir)
    
    # Create individual plots
    if not args.no_individual:
        print("\nCreating individual run plots...")
        plot_individual_runs(all_results, output_dir)
    
    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict
import argparse


def normalize_macs_units(raw_macs):
    if raw_macs is None:
        return None
    if raw_macs > 1e5:
        return raw_macs / 1e6
    return raw_macs


def scan_results_directory(results_dir, num_archs=30):
    results_dir = Path(results_dir)
    all_results = defaultdict(list)

    for stats_file in results_dir.glob('*/iter_*.stats'):
        parts = stats_file.parts
        try:
            results_idx = parts.index('results')
            run_type = parts[results_idx + 1]
        except ValueError:
            # unexpected layout
            continue

        if run_type in ['tempweg', 'old']:
            continue

        try:
            with open(stats_file, 'r') as f:
                data = json.load(f)
        except Exception:
            continue

        if 'archive' not in data or not data['archive']:
            continue

        archive = data['archive']
        last_archs = archive[-num_archs:] if len(archive) > num_archs else archive

        for idx, entry in enumerate(last_archs):
            if len(entry) != 3:
                continue
            arch_dict, error_rate, macs = entry
            accuracy = 100.0 - error_rate
            macs = normalize_macs_units(macs)
            if macs is not None and accuracy is not None:
                all_results[run_type].append({
                    'macs': macs,
                    'accuracy': accuracy,
                    'arch_idx': len(archive) - len(last_archs) + idx,
                    'path': str(stats_file),
                })

    return all_results


def filter_layerskipping_three(all_results):
    # Keep only layerskipping cifar100 runs and only convn1, simple, attention variants
    # Detection uses substrings present in run names; displayed method keys are renamed
    keep = {}
    targets = {
        'conv': 'convn1',    # detect convn1 but display as 'conv'
        'stable': 'simple',  # detect 'simple' but display as 'stable'
        'attention': 'attention'
    }

    for run_name, results in all_results.items():
        low = run_name.lower()
        if 'layerskipping' not in low or 'cifar100' not in low:
            continue
        for method_key, substr in targets.items():
            if substr in low:
                keep[run_name] = (method_key, results)
                break

    out = {}
    for run_name, (method_key, results) in keep.items():
        out[run_name] = {'method': method_key, 'results': results}

    return out


def plot_fixed_colors(filtered_results, output_dir=None, title="Accuracy-Efficiency Trade-off (layerskipping cifar100)"):
    if not filtered_results:
        print('No matching runs to plot')
        return

    # fixed colors and markers per method (use renamed keys)
    style_map = {
        'conv': {'color': '#1f77b4', 'marker': 'o', 'label': 'conv'},
        'stable': {'color': '#ff7f0e', 'marker': 's', 'label': 'stable'},
        'attention': {'color': '#2ca02c', 'marker': '^', 'label': 'attention'},
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    # plot each run with fixed color for its method
    for run_name in sorted(filtered_results.keys()):
        entry = filtered_results[run_name]
        method = entry['method']
        results = entry['results']
        style = style_map.get(method, {'color': '#7f7f7f', 'marker': 'o', 'label': method})

        macs_vals = [r['macs'] for r in results]
        acc_vals = [r['accuracy'] for r in results]

        ax.scatter(macs_vals, acc_vals,
                   c=style['color'], marker=style['marker'], s=90,
                   alpha=0.85, edgecolors='black', linewidths=0.6,
                   label=style['label'])

    ax.set_xlabel('Computation (MMACs)')
    ax.set_ylabel('Top-1 Accuracy (%)')
    ax.grid(True, alpha=0.25)

    # create unique legend entries (one per method)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc='best', framealpha=0.95)

    plt.tight_layout()

    if output_dir:
        output_path = Path(output_dir) / 'layerskipping_cifar100_fixedcolor.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f'Saved: {output_path}')
        output_path_pdf = Path(output_dir) / 'layerskipping_cifar100_fixedcolor.pdf'
        plt.savefig(output_path_pdf, bbox_inches='tight')
        print(f'Saved: {output_path_pdf}')

    plt.show()


def print_summary(filtered_results):
    print('\nSUMMARY')
    for run_name in sorted(filtered_results.keys()):
        entry = filtered_results[run_name]
        method = entry['method']
        results = entry['results']
        macs = [r['macs'] for r in results]
        acc = [r['accuracy'] for r in results]
        print(f"{run_name}: method={method}, count={len(results)}, macs={min(macs):.2f}-{max(macs):.2f} M, acc={min(acc):.2f}-{max(acc):.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--output_dir', type=str, default='./results/analysis/architectures')
    parser.add_argument('--num_archs', type=int, default=30)
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_results = scan_results_directory(args.results_dir, num_archs=args.num_archs)
    filtered = filter_layerskipping_three(all_results)
    if not filtered:
        print('No layerskipping cifar100 convn1/simple/attention runs found under', args.results_dir)
        return

    print_summary(filtered)
    plot_fixed_colors(filtered, output_dir=outdir)


if __name__ == '__main__':
    main()


import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from acc_predictor.factory import get_acc_predictor


def get_feature_names(search_space='mobilenetv3', max_layers=20):
    """
    Get feature names based on search space encoding.
    Must match the INTERLEAVED encoding order in encode_architecture function.
    
    Encoding format (per block, 5 blocks total):
    [depth_idx, ks[0], ks[1], ks[2], ks[3], e[0], e[1], e[2], e[3]]
    """
    feature_names = []
    
    if search_space == 'mobilenetv3':
        # For each of 5 blocks: depth, then 4 kernels, then 4 expansions
        for block_i in range(5):
            feature_names.append(f'Block{block_i}_depth')
            for layer_i in range(4):
                feature_names.append(f'Block{block_i}_L{layer_i}_kernel')
            for layer_i in range(4):
                feature_names.append(f'Block{block_i}_L{layer_i}_expansion')
        
        # Width multiplier - position 45
        feature_names.append('width_multiplier')
        
        # Resolution - position 46
        feature_names.append('resolution')
        
        return feature_names
        
    elif search_space == 'layerskipping':
        # Same base architecture encoding
        for block_i in range(5):
            feature_names.append(f'Block{block_i}_depth')
            for layer_i in range(4):
                feature_names.append(f'Block{block_i}_L{layer_i}_kernel')
            for layer_i in range(4):
                feature_names.append(f'Block{block_i}_L{layer_i}_expansion')
        
        # Gate hidden sizes - positions 45-64 (20 elements)
        max_gatable_blocks = 20
        for i in range(max_gatable_blocks):
            feature_names.append(f'gate_hidden_size[{i}]')
        
        # Target sparsities - positions 65-84 (20 elements)
        for i in range(max_gatable_blocks):
            feature_names.append(f'target_sparsity[{i}]')
        
        # Resolution - position 85
        feature_names.append('resolution')
        
        return feature_names
    else:
        return None


def load_archive_data(expr_path):
    """
    Load training data from search archive.
    Returns architectures and their corresponding accuracy values.
    """
    # Try to load from JSON file
    if os.path.isfile(expr_path):
        with open(expr_path, 'r') as f:
            data = json.load(f)
            archive = data.get('archive', [])
    else:
        # Load from directory structure
        archive = []
        iter_folders = sorted([d for d in os.listdir(expr_path) if d.startswith('iter_')])
        
        for iter_folder in iter_folders:
            iter_path = os.path.join(expr_path, iter_folder)
            if not os.path.isdir(iter_path):
                continue
            
            # List network directories
            net_dirs = [d for d in os.listdir(iter_path) if d.startswith('net_')]
            
            for net_dir in net_dirs:
                net_path = os.path.join(iter_path, net_dir)
                if not os.path.isdir(net_path):
                    continue
                
                # Find .subnet and .stats files
                subnet_file = None
                stats_file = None
                
                for f in os.listdir(net_path):
                    if f.endswith('.subnet'):
                        subnet_file = os.path.join(net_path, f)
                    elif f.endswith('.stats'):
                        stats_file = os.path.join(net_path, f)
                
                if subnet_file and stats_file:
                    try:
                        with open(subnet_file, 'r') as f:
                            config = json.load(f)
                        with open(stats_file, 'r') as f:
                            stats = json.load(f)
                        
                        # Extract top1 error
                        if 'top1' in stats:
                            top1_err = stats['top1']
                            archive.append([config, top1_err/100.0])  # Convert to error rate
                    except Exception as e:
                        print(f"Warning: Could not load {net_path}: {e}")
                        continue
    
    if len(archive) == 0:
        raise ValueError(f"No archive data found in {expr_path}")
    
    print(f"Loaded {len(archive)} architectures from archive")
    return archive


def encode_architecture(arch, search_space='mobilenetv3', max_layers=20, fix_res=None):
    """
    Encode architecture configuration into feature vector with fixed length.
    This MUST match the encoding used in search_space.py encode() function.
    
    Encoding format (INTERLEAVED per block):
    For each of 5 blocks: [depth_idx, ks[0], ks[1], ks[2], ks[3], e[0], e[1], e[2], e[3]]
    = 5 blocks * 9 elements = 45 base elements
    
    Note: Values are encoded as INDICES into the option arrays, not raw values!
    
    Args:
        arch: Architecture configuration dict
        search_space: Type of search space
        max_layers: Maximum number of layers
        fix_res: Whether resolution is fixed (if None, auto-detect from arch)
    """
    features = []
    
    # Auto-detect fix_res if not provided
    # If all archs have the same resolution, it's likely fix_res mode
    if fix_res is None:
        # For now, assume fix_res based on presence of 'r' in arch
        # In reality, this should be passed from the search configuration
        fix_res = False  # Default to variable resolution
    
    # Get depth, kernel, expansion values
    depths = arch.get('d', [0] * 5)
    kernels = arch.get('ks', arch.get('k', []))
    expansions = arch.get('e', [])
    
    # Options (must match search_space.py)
    depth_options = [2, 3, 4]
    kernel_options = [3, 5, 7]
    exp_options = [3, 4, 6]
    max_depth = 4
    
    # Encode each of the 5 blocks
    num_blocks = 5
    kernel_idx = 0
    expansion_idx = 0
    
    for block_i in range(num_blocks):
        # Get depth for this block
        if block_i < len(depths):
            block_depth = depths[block_i]
            # Convert to index (0, 1, or 2)
            if block_depth in depth_options:
                depth_idx = depth_options.index(block_depth)
            else:
                depth_idx = 0
        else:
            block_depth = 0
            depth_idx = 0
        
        features.append(depth_idx)
        
        # Get kernels for this block (padded to max_depth=4)
        # Encode as INDICES
        block_kernels = []
        for i in range(max_depth):
            if i < block_depth and kernel_idx < len(kernels):
                k_val = kernels[kernel_idx]
                k_idx = kernel_options.index(k_val) if k_val in kernel_options else 0
                block_kernels.append(k_idx)
                kernel_idx += 1
            else:
                block_kernels.append(0)
        features.extend(block_kernels)
        
        # Get expansions for this block (padded to max_depth=4)
        # Encode as INDICES
        block_expansions = []
        for i in range(max_depth):
            if i < block_depth and expansion_idx < len(expansions):
                e_val = expansions[expansion_idx]
                e_idx = exp_options.index(e_val) if e_val in exp_options else 0
                block_expansions.append(e_idx)
                expansion_idx += 1
            else:
                block_expansions.append(0)
        features.extend(block_expansions)
    
    # Now at position 45 (5 blocks * 9 elements each)
    
    # Now at position 45 (5 blocks * 9 elements each)
    
    if search_space == 'mobilenetv3':
        # Add width multiplier - position 45
        if 'w' in arch:
            features.append(arch['w'])
        else:
            features.append(1.0)  # default width
        
        # Add resolution - position 46 (only if not in config already as 'r')
        if 'r' in arch:
            # Resolution is already in the config, encode as index
            resolution_options = list(range(32, 257, 4))  # Common range
            r_val = arch['r']
            if r_val in resolution_options:
                features.append(resolution_options.index(r_val))
            else:
                features.append(r_val)  # Use raw value if not in options
        # Note: If 'r' not in arch, we don't append (fix_res case in search_space.py)
            
    elif search_space == 'layerskipping':
        # For skippingmobilenetv3_extended
        # Gate hidden sizes - positions 45-64 (20 elements) - encode as INDICES
        gate_hidden_size_options = [16, 32, 64]
        max_gatable_blocks = 20
        if 'gate_hidden_sizes' in arch:
            ghs_vals = arch['gate_hidden_sizes'][:max_gatable_blocks]
            ghs_indices = []
            for ghs in ghs_vals:
                if ghs in gate_hidden_size_options:
                    ghs_indices.append(gate_hidden_size_options.index(ghs))
                else:
                    ghs_indices.append(0)
            features.extend(ghs_indices)
            # Pad if needed
            if len(ghs_indices) < max_gatable_blocks:
                features.extend([0] * (max_gatable_blocks - len(ghs_indices)))
        else:
            features.extend([0] * max_gatable_blocks)
        
        # Target sparsities - positions 65-84 (20 elements) - encode as INDICES
        target_sparsity_options = [0, 0.3, 0.5, 0.7]
        if 'target_sparsities' in arch:
            ts_vals = arch['target_sparsities'][:max_gatable_blocks]
            ts_indices = []
            for ts in ts_vals:
                # Find closest match in options
                closest_idx = min(range(len(target_sparsity_options)), 
                                key=lambda i: abs(target_sparsity_options[i] - ts))
                ts_indices.append(closest_idx)
            features.extend(ts_indices)
            # Pad if needed
            if len(ts_indices) < max_gatable_blocks:
                features.extend([0] * (max_gatable_blocks - len(ts_indices)))
        else:
            features.extend([0] * max_gatable_blocks)
        
        # Resolution - position 85 (only if NOT fix_res)
        if not fix_res and 'r' in arch:
            # Resolution is in the config, encode as index
            resolution_options = list(range(32, 257, 4))
            r_val = arch['r']
            if r_val in resolution_options:
                features.append(resolution_options.index(r_val))
            else:
                # Use a normalized value if not in standard range
                features.append((r_val - 32) // 4)
        # Note: In fix_res mode, resolution is NOT appended to encoding
    
    return np.array(features)


def create_plots(importance, correlations, feature_names, y, y_pred_flat, args, output_dir):
    """
    Create comprehensive visualization plots for surrogate importance analysis.
    """
    print("\nGenerating visualization plots...")
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    
    # 1. Top-K Feature Importance Bar Plot
    print("  - Creating feature importance bar plot...")
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get top-k features
    sorted_indices = np.argsort(importance)[::-1][:args.top_k]
    top_importance = importance[sorted_indices]
    top_correlations = correlations[sorted_indices]
    top_names = [feature_names[i] if i < len(feature_names) else f"Feature_{i}" for i in sorted_indices]
    
    # Create signed importance: negative for hurts (red), positive for helps (green)
    signed_importance = np.array([imp if corr < 0 else -imp for imp, corr in zip(top_importance, top_correlations)])
    
    # Create bar plot with colors based on correlation
    colors = ['green' if corr < 0 else 'red' for corr in top_correlations]
    bars = ax.barh(range(len(top_names)), signed_importance, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names)
    ax.set_xlabel('Decision tree feature Importance', fontsize=12)
    ax.set_title(f'Surrogate model feature importance', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
    
    # Add value labels on bars
    for i, (bar, val, unsigned_val) in enumerate(zip(bars, signed_importance, top_importance)):
        x_pos = val + (0.01 if val > 0 else -0.01)
        ha = 'left' if val > 0 else 'right'
        ax.text(x_pos, i, f'{unsigned_val:.4f}', va='center', ha=ha, fontsize=9)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.7, edgecolor='black', label='Negative correlation'),
        Patch(facecolor='green', alpha=0.7, edgecolor='black', label='Positive correlation')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance_top_k.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Feature Importance vs Correlation Scatter Plot
    print("  - Creating importance vs correlation scatter plot...")
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Filter out zero-importance features for better visualization
    non_zero_mask = importance > 1e-6
    scatter = ax.scatter(correlations[non_zero_mask], importance[non_zero_mask], 
                        c=correlations[non_zero_mask], cmap='RdYlGn_r', 
                        s=120, alpha=0.7, edgecolors='black', linewidth=0.8)
    
    # Annotate top features with simpler positioning
    for idx in sorted_indices[:15]:
        if importance[idx] > 1e-6:
            feat_name = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
            # Simplify long names
            if len(feat_name) > 25:
                feat_name = feat_name[:22] + "..."
            ax.annotate(feat_name, (correlations[idx], importance[idx]), 
                       fontsize=9, fontweight='bold',
                       xytext=(5, 5), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='gray'))
    
    ax.set_xlabel('Correlation with Error\n← Helps Accuracy (Negative) | Hurts Accuracy (Positive) →', 
                  fontsize=13, fontweight='bold')
    ax.set_ylabel('Feature Importance', fontsize=13, fontweight='bold')
    ax.set_title('Feature Importance vs Correlation with Error', fontsize=14, fontweight='bold', pad=15)
    
    # Add vertical line at zero with labels
    ax.axvline(x=0, color='black', linestyle='--', linewidth=2, alpha=0.5)
    
    # Add background shading
    ylim = ax.get_ylim()
    ax.axvspan(ax.get_xlim()[0], 0, alpha=0.1, color='green', label='Helps Region')
    ax.axvspan(0, ax.get_xlim()[1], alpha=0.1, color='red', label='Hurts Region')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Correlation with Error', fontsize=11, fontweight='bold')
    
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'importance_vs_correlation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Predicted vs Actual Error Plot
    print("  - Creating predicted vs actual plot...")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot
    ax.scatter(y, y_pred_flat, alpha=0.5, s=30, edgecolors='black', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(y.min(), y_pred_flat.min())
    max_val = max(y.max(), y_pred_flat.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Calculate and display metrics
    rmse = np.sqrt(((y_pred_flat - y) ** 2).mean())
    mae = np.abs(y_pred_flat - y).mean()
    r2 = 1 - (np.sum((y - y_pred_flat) ** 2) / np.sum((y - y.mean()) ** 2))
    correlation = np.corrcoef(y, y_pred_flat)[0, 1]
    
    # Add text box with metrics
    textstr = f'RMSE = {rmse:.6f}\nMAE = {mae:.6f}\nR² = {r2:.4f}\nCorr = {correlation:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    ax.set_xlabel('Actual Error', fontsize=12)
    ax.set_ylabel('Predicted Error', fontsize=12)
    ax.set_title('Surrogate Model: Predicted vs Actual Error', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'predicted_vs_actual.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Residual Plot
    print("  - Creating residual plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    residuals = y_pred_flat - y
    ax.scatter(y_pred_flat, residuals, alpha=0.5, s=30, edgecolors='black', linewidth=0.5)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax.axhline(y=residuals.std(), color='orange', linestyle='--', alpha=0.7, label=f'±1 Std ({residuals.std():.4f})')
    ax.axhline(y=-residuals.std(), color='orange', linestyle='--', alpha=0.7)
    
    ax.set_xlabel('Predicted Error', fontsize=12)
    ax.set_ylabel('Residual (Predicted - Actual)', fontsize=12)
    ax.set_title('Residual Plot', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residual_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Feature Category Importance (grouped by type)
    print("  - Creating feature category importance plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Group features by category - adapt based on search space
    # Note: Encoding is INTERLEAVED per block: [depth, ks[0-3], e[0-3]] repeated 5 times
    if args.search_space == 'mobilenetv3':
        categories = {
            'Depth (all blocks)': [i*9 for i in range(5)],  # Every 9th position starting at 0
            'Kernel Size (all layers)': [i*9+j for i in range(5) for j in range(1, 5)],  # Positions 1-4, 10-13, 19-22, 28-31, 37-40
            'Expansion (all layers)': [i*9+j for i in range(5) for j in range(5, 9)],  # Positions 5-8, 14-17, 23-26, 32-35, 41-44
            'Width & Resolution': [45, 46]
        }
    elif args.search_space == 'layerskipping':
        categories = {
            'Depth (all blocks)': [i*9 for i in range(5)],
            'Kernel Size (all layers)': [i*9+j for i in range(5) for j in range(1, 5)],
            'Expansion (all layers)': [i*9+j for i in range(5) for j in range(5, 9)],
            'Gate Hidden Sizes': list(range(45, 65)),  # Positions 45-64
            'Target Sparsities': list(range(65, 85)),  # Positions 65-84
            'Resolution': [85]
        }
    else:
        # Default fallback
        categories = {
            'Depth (all blocks)': [i*9 for i in range(5)],
            'Kernel Size (all layers)': [i*9+j for i in range(5) for j in range(1, 5)],
            'Expansion (all layers)': [i*9+j for i in range(5) for j in range(5, 9)],
            'Other': list(range(45, len(importance)))
        }
    
    category_importance = {}
    for cat_name, indices in categories.items():
        # Sum importance for features in this category
        cat_imp = sum(importance[i] for i in indices if i < len(importance))
        category_importance[cat_name] = cat_imp
    
    # Sort by importance
    sorted_categories = sorted(category_importance.items(), key=lambda x: x[1], reverse=True)
    cat_names = [c[0] for c in sorted_categories]
    cat_values = [c[1] for c in sorted_categories]
    
    colors_cat = plt.cm.viridis(np.linspace(0, 0.9, len(cat_names)))
    bars = ax.barh(range(len(cat_names)), cat_values, color=colors_cat, alpha=0.8)
    
    ax.set_yticks(range(len(cat_names)))
    ax.set_yticklabels(cat_names)
    ax.set_xlabel('Total Importance', fontsize=12)
    ax.set_title('Feature Category Importance (Grouped)', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, cat_values)):
        ax.text(val, i, f' {val:.4f}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'category_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Error Distribution
    print("  - Creating error distribution plot...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram of actual errors
    ax1.hist(y, bins=30, alpha=0.7, color='blue', edgecolor='black', label='Actual Error')
    ax1.axvline(y.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean = {y.mean():.4f}')
    ax1.set_xlabel('Error Rate', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution of Actual Errors', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Histogram of prediction errors (residuals)
    ax2.hist(residuals, bins=30, alpha=0.7, color='orange', edgecolor='black', label='Prediction Error')
    ax2.axvline(residuals.mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean = {residuals.mean():.6f}')
    ax2.set_xlabel('Prediction Error (Residual)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of Prediction Errors', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ All plots saved to: {output_dir}")
    print("  Generated plots:")
    print("    - feature_importance_top_k.png")
    print("    - importance_vs_correlation.png")
    print("    - predicted_vs_actual.png")
    print("    - residual_plot.png")
    print("    - category_importance.png")
    print("    - error_distributions.png")


def main(args):
    print("\n" + "="*80)
    print("SURROGATE MODEL FEATURE IMPORTANCE ANALYSIS")
    print("="*80)
    
    # Load archive data
    expr_path = args.expr
    if not expr_path.endswith('.json'):
        expr_path, _ = os.path.splitext(expr_path)
    
    print(f"\nLoading data from: {expr_path}")
    archive = load_archive_data(expr_path)
    
    # Extract and encode architectures
    print("\nEncoding architectures...")
    architectures = [v[0] for v in archive]
    accuracies = np.array([v[1] for v in archive])  # Top-1 error
    
    # Detect if resolution is fixed (all archs have same resolution)
    resolutions = [arch.get('r', None) for arch in architectures]
    unique_resolutions = set(r for r in resolutions if r is not None)
    fix_res = len(unique_resolutions) <= 1
    if fix_res and len(unique_resolutions) == 1:
        print(f"Detected fixed resolution mode: r={list(unique_resolutions)[0]}")
    else:
        print(f"Detected variable resolution mode: {len(unique_resolutions)} different resolutions")
    
    # Encode to feature vectors
    X = np.array([encode_architecture(arch, args.search_space, fix_res=fix_res) for arch in architectures])
    y = accuracies
    
    print(f"Feature vector shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Top-1 error range: [{y.min():.4f}, {y.max():.4f}]")
    
    # Get feature names
    feature_names = get_feature_names(args.search_space)
    if feature_names and len(feature_names) > X.shape[1]:
        feature_names = feature_names[:X.shape[1]]
    
    # Train surrogate model
    print(f"\nTraining {args.predictor.upper()} surrogate model...")
    surrogate = get_acc_predictor(args.predictor, X, y)
    
    # Test predictions and calculate goodness of fit
    y_pred = surrogate.predict(X)
    y_pred_flat = y_pred.flatten()
    
    # Calculate metrics
    rmse = np.sqrt(((y_pred_flat - y) ** 2).mean())
    mae = np.abs(y_pred_flat - y).mean()
    
    # R-squared (coefficient of determination)
    ss_res = np.sum((y - y_pred_flat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Pearson correlation coefficient
    correlation = np.corrcoef(y, y_pred_flat)[0, 1]
    
    print("\n" + "="*80)
    print("GOODNESS OF FIT METRICS")
    print("="*80)
    print(f"RMSE (Root Mean Squared Error):     {rmse:.6f}")
    print(f"MAE (Mean Absolute Error):          {mae:.6f}")
    print(f"R² (Coefficient of Determination):  {r2:.6f}")
    print(f"Pearson Correlation:                {correlation:.6f}")
    print(f"Mean Prediction Error:              {(y_pred_flat - y).mean():.6f}")
    print(f"Std Prediction Error:               {(y_pred_flat - y).std():.6f}")
    print("="*80)
    
    # Analyze feature importance
    print("\nAnalyzing feature importance...")
    
    if args.predictor == 'carts':
        # Get feature importance
        importance_data = surrogate.get_feature_importance(X.shape[1])
        importance = importance_data['importance']
        
        # Calculate correlation between each feature and target
        # Positive correlation = higher feature value -> higher accuracy (lower error)
        # Negative correlation = higher feature value -> lower accuracy (higher error)
        feature_correlations = np.zeros(X.shape[1])
        for i in range(X.shape[1]):
            if np.std(X[:, i]) > 1e-10:  # Only for non-constant features
                feature_correlations[i] = np.corrcoef(X[:, i], y)[0, 1]
        
        surrogate.print_feature_importance(
            n_features=X.shape[1],
            feature_names=feature_names,
            top_k=args.top_k
        )
        
        # Print additional analysis showing direction of effect
        print("\n" + "="*80)
        print("FEATURE EFFECT DIRECTION ANALYSIS")
        print("="*80)
        print("Note: Lower error = Better accuracy")
        print("Negative correlation → Higher feature value helps (improves accuracy)")
        print("Positive correlation → Higher feature value hurts (reduces accuracy)")
        print("\n" + "-"*80)
        print(f"{'Rank':<6} {'Feature':<30} {'Importance':<12} {'Correlation':<14} {'Effect':<15}")
        print("-"*80)
        
        sorted_indices = np.argsort(importance)[::-1]
        for rank, idx in enumerate(sorted_indices[:args.top_k], 1):
            if feature_names and idx < len(feature_names):
                feat_name = feature_names[idx]
            else:
                feat_name = f"Feature_{idx}"
            
            corr = feature_correlations[idx]
            if abs(corr) < 0.01:
                effect = "Negligible"
            elif corr < 0:
                effect = "↑ Helps"
            else:
                effect = "↑ Hurts"
            
            print(f"{rank:<6} {feat_name:<30} {importance[idx]:<12.4f} {corr:>+12.4f}  {effect:<15}")
        
        print("-"*80)
        print("="*80 + "\n")
        
        # Save detailed importance data and create plots
        if args.save:
            output = {
                'predictor': args.predictor,
                'n_features': X.shape[1],
                'feature_names': feature_names if feature_names else [f"Feature_{i}" for i in range(X.shape[1])],
                'importance': importance.tolist(),
                'usage_percentage': importance_data['usage_percentage'].tolist(),
                'correlations': feature_correlations.tolist(),
                'goodness_of_fit': {
                    'rmse': float(rmse),
                    'mae': float(mae),
                    'r2': float(r2),
                    'pearson_correlation': float(correlation)
                }
            }
            output_file = os.path.join(args.save, 'surrogate_feature_importance.json')
            os.makedirs(args.save, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(output, f, indent=2)
            print(f"Saved detailed importance data to: {output_file}")
            
            # Create visualization plots
            if args.plot:
                create_plots(importance, feature_correlations, feature_names, 
                           y, y_pred_flat, args, args.save)
    
    elif args.predictor == 'mlp':
        surrogate.print_feature_importance(
            x_data=X,
            feature_names=feature_names,
            top_k=args.top_k,
            method='gradient',
            device=args.device
        )
        
        # Save detailed importance data
        if args.save:
            importance_data = surrogate.get_feature_importance(X, method='gradient', device=args.device)
            output = {
                'predictor': args.predictor,
                'n_features': X.shape[1],
                'feature_names': feature_names if feature_names else [f"Feature_{i}" for i in range(X.shape[1])],
                'importance': importance_data['importance'].tolist(),
                'method': importance_data['method'],
            }
            output_file = os.path.join(args.save, 'surrogate_feature_importance.json')
            os.makedirs(args.save, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(output, f, indent=2)
            print(f"\nSaved detailed importance data to: {output_file}")
            
            # Create visualization plots
            if args.plot:
                importance_data = surrogate.get_feature_importance(X, method='gradient', device=args.device)
                # For MLP, we don't have correlations readily available, compute them
                feature_correlations = np.zeros(X.shape[1])
                for i in range(X.shape[1]):
                    if np.std(X[:, i]) > 1e-10:
                        feature_correlations[i] = np.corrcoef(X[:, i], y)[0, 1]
                
                create_plots(importance_data['importance'], feature_correlations, 
                           feature_names, y, y_pred_flat, args, args.save)
    
    else:
        print(f"Feature importance analysis not implemented for {args.predictor}")
    
    print("\nAnalysis complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze surrogate model feature importance')
    parser.add_argument('--expr', type=str, required=True,
                        help='Path to search results directory or JSON file')
    parser.add_argument('--predictor', type=str, default='carts',
                        choices=['carts', 'mlp'],
                        help='Type of surrogate predictor to analyze (only carts and mlp support feature importance)')
    parser.add_argument('--search_space', type=str, default='layerskipping',
                        help='Search space type')
    parser.add_argument('--top_k', type=int, default=20,
                        help='Number of top features to display')
    parser.add_argument('--save', type=str, default=None,
                        help='Directory to save importance analysis results')
    parser.add_argument('--plot', action='store_true',
                        help='Generate visualization plots (requires --save)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device for MLP gradient calculation')
    
    args = parser.parse_args()
    main(args)

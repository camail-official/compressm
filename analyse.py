#!/usr/bin/env python3
"""
Analysis script to compare test metrics from multiple directories.
Groups runs by initial SSM dimension and tolerance (if any) from all directories.
Shows results from each directory with different markers for comparison.
Supports arbitrary number of input directories.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import re
import argparse
from pathlib import Path
from matplotlib.patches import Rectangle

def parse_run_name(run_dir):
    """Parse run directory name to extract key parameters."""
    # Extract SSM dimension
    ssm_match = re.search(r'ssm_dim_(\d+)', run_dir)
    ssm_dim = int(ssm_match.group(1)) if ssm_match else None
    
    # Extract tolerance if present
    tol_match = re.search(r'tol_([\d.e-]+)', run_dir)
    tolerance = float(tol_match.group(1)) if tol_match else None
    
    # Extract seed
    seed_match = re.search(r'seed_(\d+)', run_dir)
    seed = int(seed_match.group(1)) if seed_match else None

    # Extract number of blocks
    num_blocks_match = re.search(r'num_blocks_(\d+)', run_dir)
    num_blocks = int(num_blocks_match.group(1)) if num_blocks_match else None

    return {
        'ssm_dim': ssm_dim,
        'tolerance': tolerance,
        'seed': seed,
        'run_name': run_dir,
        'num_blocks': num_blocks
    }

def load_test_metric(run_path):
    """Load the test metric from a run directory."""
    test_metric_file = os.path.join(run_path, 'test_metric.npy')
    if os.path.exists(test_metric_file):
        try:
            test_metric = np.load(test_metric_file)
            # Handle both scalar and array cases
            if np.isscalar(test_metric):
                return float(test_metric)
            elif len(test_metric.shape) == 0:
                return float(test_metric.item())
            else:
                return float(test_metric[0]) if len(test_metric) > 0 else None
        except:
            return None
    return None

def load_time_metric(run_path):
    """Load the time metric from a run directory."""
    time_metric_file = os.path.join(run_path, 'all_time.npy')
    if os.path.exists(time_metric_file):
        try:
            time_metric = np.load(time_metric_file)
            # Handle both scalar and array cases
            if np.isscalar(time_metric):
                return float(time_metric)
            elif len(time_metric.shape) == 0:
                return float(time_metric.item())
            else:
                return float(sum(time_metric)) if len(time_metric) > 0 else None
        except:
            return None
    return None

def collect_results(base_path, directory_label=None):
    """Collect all results from the output directory."""
    results = []
    
    if not os.path.exists(base_path):
        print(f"Base path {base_path} does not exist")
        return results
    
    # Check if base_path contains run directories directly (like outputs/lru/mnist/)
    # or if it contains model/dataset subdirectories
    base_path_items = os.listdir(base_path)
    
    # Check if any item in base_path looks like a run directory (contains experiment parameters)
    has_run_dirs = any(re.search(r'ssm_dim_\d+', item) for item in base_path_items 
                       if os.path.isdir(os.path.join(base_path, item)))
    
    if has_run_dirs:
        # Base path already points to dataset level, process run directories directly
        print(f"Processing run directories directly from: {base_path}")
        for run_dir in base_path_items:
            run_path = os.path.join(base_path, run_dir)
            if not os.path.isdir(run_path):
                continue
            
            # Parse run parameters
            run_info = parse_run_name(run_dir)
            if run_info['ssm_dim'] is None:
                continue
            
            # Load test metric
            test_metric = load_test_metric(run_path)
            if test_metric is None:
                continue

            # Load time metric
            time_metric = load_time_metric(run_path)
            if time_metric is None:
                continue

            # Calculate final dimension
            final_dim = run_info['ssm_dim']  # Default to initial dimension
            if run_info['tolerance'] is not None:  # This is a reduced run
                reduction_history_path = os.path.join(run_path, "reduction_history.npy")
                if os.path.exists(reduction_history_path):
                    try:
                        reduction_history = np.load(reduction_history_path, allow_pickle=True)
                        if reduction_history is not None and len(reduction_history) > 0:
                            # Get the number of blocks from config or infer from reduction history
                            config_path = os.path.join(run_path, "config.json")
                            num_blocks = 1  # Default
                            if os.path.exists(config_path):
                                with open(config_path, 'r') as f:
                                    config = json.load(f)
                                    num_blocks = int(config.get('num_blocks', 1))
                            
                            # Calculate final dimension by checking each block
                            block_dims = {}
                            for entry in reduction_history:
                                if 'block' in entry and 'new_dim' in entry:
                                    block_dims[entry['block']] = entry['new_dim']
                            
                            # Calculate average final dimension across all blocks
                            total_final_dim = 0
                            for block_id in range(num_blocks):
                                if block_id in block_dims:
                                    # This block was reduced
                                    total_final_dim += block_dims[block_id]
                                else:
                                    # This block kept its original dimension
                                    total_final_dim += run_info['ssm_dim']
                            
                            final_dim = total_final_dim / num_blocks
                    except Exception as e:
                        print(f"Warning: Could not load reduction history for {run_path}: {e}")
            
            # Extract model and dataset from path
            path_parts = base_path.rstrip('/').split('/')
            model = path_parts[-2] if len(path_parts) >= 2 else 'unknown'
            dataset = path_parts[-1] if len(path_parts) >= 1 else 'unknown'
            
            # Add to results
            result = {
                'model': model,
                'dataset': dataset,
                'ssm_dim': run_info['ssm_dim'],
                'tolerance': run_info['tolerance'],
                'seed': run_info['seed'],
                'test_metric': test_metric,
                'final_dim': final_dim,
                'time_metric': time_metric,
                'run_name': run_dir,
                'directory_label': directory_label or base_path.split('/')[-1]
            }
            results.append(result)
    else:
        # Original nested structure: base_path/model_dir/dataset_dir/run_dir
        print(f"Processing nested directory structure from: {base_path}")
        for model_dir in os.listdir(base_path):
            model_path = os.path.join(base_path, model_dir)
            if not os.path.isdir(model_path):
                continue
                
            for dataset_dir in os.listdir(model_path):
                dataset_path = os.path.join(model_path, dataset_dir)
                if not os.path.isdir(dataset_path):
                    continue
                    
                for run_dir in os.listdir(dataset_path):
                    run_path = os.path.join(dataset_path, run_dir)
                    if not os.path.isdir(run_path):
                        continue
                    
                    # Parse run parameters
                    run_info = parse_run_name(run_dir)
                    if run_info['ssm_dim'] is None:
                        continue
                    
                    # Load test metric
                    test_metric = load_test_metric(run_path)
                    if test_metric is None:
                        continue

                    # Load time metric
                    time_metric = load_time_metric(run_path)
                    if time_metric is None:
                        continue
                    
                    # Calculate final dimension
                    final_dim = run_info['ssm_dim']  # Default to initial dimension
                    if run_info['tolerance'] is not None:  # This is a reduced run
                        reduction_history_path = os.path.join(run_path, "reduction_history.npy")
                        if os.path.exists(reduction_history_path):
                            try:
                                reduction_history = np.load(reduction_history_path, allow_pickle=True)
                                if reduction_history is not None and len(reduction_history) > 0:
                                    # Get the number of blocks from config or infer from reduction history
                                    config_path = os.path.join(run_path, "config.json")
                                    num_blocks = run_info['num_blocks']

                                    # Calculate final dimension by checking each block
                                    block_dims = {}
                                    for entry in reduction_history:
                                        if 'block' in entry and 'new_dim' in entry:
                                            block_dims[entry['block']] = entry['new_dim']
                                    
                                    # Calculate average final dimension across all blocks
                                    total_final_dim = 0
                                    for block_id in range(num_blocks):
                                        if block_id in block_dims:
                                            # This block was reduced
                                            total_final_dim += block_dims[block_id]
                                        else:
                                            # This block kept its original dimension
                                            total_final_dim += run_info['ssm_dim']
                                    
                                    final_dim = total_final_dim / num_blocks
                            except Exception as e:
                                print(f"Warning: Could not load reduction history for {run_path}: {e}")
                    
                    # Add to results
                    result = {
                        'model': model_dir,
                        'dataset': dataset_dir,
                        'ssm_dim': run_info['ssm_dim'],
                        'tolerance': run_info['tolerance'],
                        'seed': run_info['seed'],
                        'test_metric': test_metric,
                        'time_metric': time_metric,
                        'final_dim': final_dim,
                        'run_name': run_dir,
                        'directory_label': directory_label or base_path.split('/')[-1]
                    }
                    results.append(result)
    
    return results

def create_category_labels(results):
    """Create category labels for grouping runs."""
    categories = {}
    
    for result in results:
        ssm_dim = result['ssm_dim']
        tolerance = result['tolerance']
        
        if tolerance is None:
            # No reduction
            category = f"SSM-{ssm_dim} (No reduction)"
        else:
            # With reduction
            if tolerance >= 1e-1:
                tol_str = f"{tolerance:.1f}"
            elif tolerance >= 1e-2:
                tol_str = f"{tolerance:.2f}"
            else:
                tol_str = f"{tolerance:.0e}"
            category = f"SSM-{ssm_dim} (tol={tol_str})"
        
        result['category'] = category
        if category not in categories:
            categories[category] = []
        categories[category].append(result)
    
    return categories

def plot_test_metrics(results, save_path=None):
    """Create a comprehensive plot of test metrics showing min/max/mean by category."""
    
    if not results:
        print("No results found to plot")
        return
    
    # Get unique directory labels, models and datasets from results
    directory_labels = list(set(r['directory_label'] for r in results))
    model_names = list(set(r['model'] for r in results))
    dataset_names = list(set(r['dataset'] for r in results))
    
    # Create title based on available data
    if len(model_names) == 1 and len(dataset_names) == 1:
        title_suffix = f"{model_names[0]} on {dataset_names[0]}"
    elif len(model_names) == 1:
        title_suffix = f"{model_names[0]} on {', '.join(dataset_names)}"
    elif len(dataset_names) == 1:
        title_suffix = f"{', '.join(model_names)} on {dataset_names[0]}"
    else:
        title_suffix = f"Multiple models and datasets"
    
    # Group results by initial SSM dimension, tolerance, AND directory for aggregation
    grouped_results = {}
    
    for result in results:
        ssm_dim = result['ssm_dim']
        tolerance = result['tolerance']
        directory_label = result['directory_label']
        
        # Create a key for grouping that includes directory
        if tolerance is None:
            group_key = f"SSM-{ssm_dim}_no_reduction_{directory_label}"
        else:
            group_key = f"SSM-{ssm_dim}_tol-{tolerance}_{directory_label}"
        
        if group_key not in grouped_results:
            grouped_results[group_key] = []
        grouped_results[group_key].append(result)
    
    # Create figure matching the Pareto plot style
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Process each group and calculate statistics
    plot_data = []
    
    for group_key, group_results in grouped_results.items():
        # Extract information from group key (now includes directory)
        if "_no_reduction_" in group_key:
            parts = group_key.split("_no_reduction_")
            initial_ssm_dim = int(parts[0].replace('SSM-', ''))
            tolerance = None
            directory_label = parts[1]
            label = f"SSM-{initial_ssm_dim} (No reduction) - {directory_label}"
        else:
            # Split by directory label (assumed to be at the end)
            parts = group_key.rsplit('_', 1)
            directory_label = parts[1]
            remaining = parts[0].split('_tol-')
            initial_ssm_dim = int(remaining[0].replace('SSM-', ''))
            tolerance = float(remaining[1])
            label = f"SSM-{initial_ssm_dim} (tol={tolerance:.0e}) - {directory_label}"
        
        # Calculate test metric statistics
        test_metrics = [r['test_metric'] for r in group_results]
        # Get indices of the three largest test metrics
        top_3_indices = np.argsort(test_metrics)[-3:]
        best_idx = top_3_indices[-1]  # Index of the best (highest) test metric

        test_min = np.min(test_metrics)
        test_max = np.max(test_metrics) 
        test_mean = np.mean(test_metrics)
        top_3_mean = np.mean([test_metrics[i] for i in top_3_indices])

        # calculate the time metric statistics
        time_metrics = [r['time_metric'] for r in group_results]
        time_min = np.min(time_metrics)
        time_max = np.max(time_metrics)
        time_mean = np.mean(time_metrics)
        top_3_time = np.mean([time_metrics[i] for i in top_3_indices])
        best_time = time_metrics[best_idx]

        if tolerance is None:
            # No reduction: use initial SSM dimension as x-coordinate
            x_coord = initial_ssm_dim
            # For final dimensions, they're the same as initial
            final_dims = [result['final_dim'] for result in group_results]
        else:
            # With reduction: extract final dimensions from results and use mean as x-coordinate
            final_dims = [result['final_dim'] for result in group_results]
            # Use mean final dimension as x-coordinate
            x_coord = np.mean(final_dims)

        # calculate final dimension statistics
        final_min = np.min(final_dims)
        final_max = np.max(final_dims)
        final_mean = np.mean(final_dims)
        best_final_dim = group_results[best_idx]['final_dim']
        top_3_final = np.mean([final_dims[i] for i in top_3_indices])
        
        if final_min >= 2**4:
            plot_data.append({
                'x_coord': x_coord,
                'test_min': test_min,
                'test_max': test_max,
                'test_mean': test_mean,
                'time_min': time_min,
                'time_max': time_max,
                'time_mean': time_mean,
                'final_min': final_min,
                'final_max': final_max, 
                'final_mean': final_mean,
                'final_best': best_final_dim,
                'time_best': best_time,
                'top_3_mean': top_3_mean,
                'top_3_final': top_3_final,
                'top_3_time': top_3_time,
                'initial_ssm_dim': initial_ssm_dim,
                'tolerance': tolerance,
                'label': label,
                'directory_label': directory_label,
                'n_runs': len(group_results),
                'init_dim': initial_ssm_dim
            })
    
    # Sort plot data by initial SSM dimension, then by reduction status
    plot_data.sort(key=lambda x: (x['initial_ssm_dim'], x['tolerance'] is not None, x['tolerance'] or 0))
    
    # Plot the data
    # Define markers and colors for different directories
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x']
    dir_to_marker = {}
    unique_dirs = list(set(data['directory_label'] for data in plot_data))
    for i, dir_label in enumerate(unique_dirs):
        dir_to_marker[dir_label] = markers[i % len(markers)]
    
    for i, data in enumerate(plot_data):
        x = data['x_coord']
        
        # Choose color based on reduction status
        if data['tolerance'] is None:
            color = 'black'
            alpha = 0.6
        else:
            # Use autumn colormap for reduced runs
            color_intensity = min(1.0, (i / len(plot_data))) - 0.1
            color = plt.cm.autumn(color_intensity)
            alpha = 0.6
        
        # Use different markers for different directories
        marker = dir_to_marker[data['directory_label']]
        
        # Plot rectangle showing min/max ranges
        
        # Calculate rectangle parameters
        rect_width = data['final_max'] - data['final_min']
        rect_height = data['test_max'] - data['test_min']
        rect_x = data['final_min']
        rect_y = data['test_min']
        
        # Draw rectangle with transparent background
        if rect_width == 0:
            rect = Rectangle((rect_x, rect_y), rect_width, rect_height,
                    facecolor=color, alpha=0.2, edgecolor=color, 
                    linewidth=5, linestyle='-')
        else:
            rect = Rectangle((rect_x, rect_y), rect_width, rect_height,
                    facecolor=color, alpha=0.2, edgecolor=color, 
                    linewidth=1, linestyle='-')
        # ax.add_patch(rect)
        
        # # Plot mean point on top
        ax.scatter(x, data['test_mean'], color=color, alpha=alpha, 
              marker=marker, s=150, label=data['label'] if i < 10 else "",
              edgecolors='black', linewidths=1, zorder=3)

        # ax.scatter(data['final_median'], data['test_median'], color=color, alpha=alpha, marker="D", s=100)

        ax.scatter(data['final_best'], data['test_max'], color=color, alpha=alpha, marker="*", s=250, label=data['label'])

        ax.scatter(data['top_3_final'], data['top_3_mean'], color=color, alpha=alpha, marker="D", s=50)
        
        # # Add small text showing number of runs under the mean point, and the average mean final dim if reduced
        if data['tolerance'] is None:
            ax.annotate(f"N={data['n_runs']}\nDim={int(x)}", 
                   (x, data['test_mean']* 0.997),
                   ha='center', va='top',
                   fontsize=8, color='gray')
        if data['tolerance'] is not None:
            ax.annotate(f"N={data['n_runs']}\nDim={int(x)}\ntol: {data['tolerance']:.1e}", 
                       (x, data['test_mean'] * 0.997),
                       ha='center', va='top',
                       fontsize=8, color='gray')

    ax.set_xlabel('SSM Dimension')
    ax.set_ylabel('Test Metric')
    ax.set_title(f'{title_suffix}: Test Metrics from {len(directory_labels)} directories')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    
    # Add legend with marker explanation
    # legend1 = ax.legend(loc='best')
    
    # Add a second legend for marker meanings (dynamic based on number of directories)
    from matplotlib.lines import Line2D
    legend_elements = []
    for dir_label in unique_dirs:
        marker = dir_to_marker[dir_label]
        legend_elements.append(
            Line2D([0], [0], marker=marker, color='gray', linestyle='None', markersize=8, label=dir_label)
        )
    
    # ax.add_artist(legend1)  # Keep the first legend
    ax.legend(handles=legend_elements, loc='lower right', title='Directories')

    # make y ticks at 0.01 intervals
    ax.set_yticks(np.arange(0.01, 1.01, 0.01))
    ax.set_yticklabels([f"{y:.2f}" for y in np.arange(0.01, 1.01, 0.01)])
    
    # set y limits to closest 0.02 tick below and above min and max
    if plot_data:
        y_min = min(data['test_mean'] for data in plot_data)
        y_max = max(data['test_max'] for data in plot_data)
        
        # Round to nearest 0.02 tick below and above
        y_min_tick = np.floor(y_min / 0.02) * 0.02
        y_max_tick = np.ceil(y_max / 0.02) * 0.02
        
        ax.set_ylim(y_min_tick, y_max_tick)
    else:
        ax.set_ylim(0.9, 1.0)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path + 'test_metrics_by_category.png', dpi=300, bbox_inches='tight')
    
    plt.close(fig)

    # find the longest time among all runs for normalization
    max_time = max(data['time_mean'] for data in plot_data if data['time_mean'] is not None)
    print(f"Max time across all runs: {max_time/3600:.2f} hours")

    # make the time figure
    fig, ax = plt.subplots(figsize=(12, 8))
    for i, data in enumerate(plot_data):
            # Choose color based on reduction status
            if data['tolerance'] is None:
                color = 'black'
                alpha = 0.6
            else:
                # Use autumn colormap for reduced runs
                color_intensity = data['x_coord']/data['init_dim']  # Normalize by initial dimension
                color = plt.cm.autumn(color_intensity)
                alpha = 0.6
            x = data['time_mean']/max_time
            y = data['test_mean']
            scatter = plt.scatter(x, y, 
                        color=color, alpha=alpha, 
                        marker=marker, s=data['final_mean']**2/75, label=data['label'] if i < 10 else "",
                        edgecolors='black', linewidths=1, zorder=3, cmap='autumn')
            
            # # do the same with the best runs
            # x_best = data['time_best']/max_time
            # y_best = data['test_max']
            # ax.scatter(x_best, y_best, color=color, alpha=alpha, marker="*", s=data['final_best']**2/50, label=data['label'])

            # # Add small text showing number of runs under the mean point, and the average mean final dim if reduced
            if data['tolerance'] is None:
                ax.annotate(f"N={data['n_runs']}\nDim={int(data['final_mean'])}", 
                    (x, y * 0.997),
                    ha='center', va='top',
                    fontsize=8, color='gray')
            if data['tolerance'] is not None:
                ax.annotate(f"N={data['n_runs']}\nDim={int(data['final_mean'])}\ntol: {data['tolerance']:.1e}", 
                        (x, y * 0.997),
                        ha='center', va='top',
                        fontsize=8, color='gray')
                
    plt.xlabel('Training Time (normalized vs longest run)')
    plt.ylabel('Test Metric')
    plt.title(f'{title_suffix}: Test Metric vs Training Time')
    plt.grid(True, alpha=0.3)
    
    # Add colorbar
    # cbar = plt.colorbar(scatter)
    # cbar.set_label('Reduction Percentage (%)')
    
    plt.savefig(os.path.join(save_path, 'test_metric_vs_time.png'), dpi=300, bbox_inches='tight')
    print(f"Plot saved in {save_path}")
    plt.close()
    
    return

def print_summary_stats(results):
    """Print summary statistics."""
    categories = create_category_labels(results)
    
    # Group by directory for summary
    directory_stats = defaultdict(list)
    for result in results:
        directory_stats[result['directory_label']].append(result)
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS BY DIRECTORY")
    print("="*80)
    
    for directory_label, dir_results in directory_stats.items():
        print(f"\n{directory_label}: {len(dir_results)} runs")
        print("-" * 40)
        
        # Group by category within this directory
        dir_categories = create_category_labels(dir_results)
        for category in sorted(dir_categories.keys()):
            test_metrics = [r['test_metric'] for r in dir_categories[category]]
            print(f"  {category}:")
            print(f"    Count: {len(test_metrics)}")
            print(f"    Mean:  {np.mean(test_metrics):.4f}")
            print(f"    Std:   {np.std(test_metrics):.4f}")
            print(f"    Min:   {np.min(test_metrics):.4f}")
            print(f"    Max:   {np.max(test_metrics):.4f}")
    
    print("\n" + "="*80)
    print("OVERALL SUMMARY STATISTICS")
    print("="*80)
    
    for category in sorted(categories.keys()):
        test_metrics = [r['test_metric'] for r in categories[category]]
        print(f"\n{category}:")
        print(f"  Count: {len(test_metrics)}")
        print(f"  Mean:  {np.mean(test_metrics):.4f}")
        # print the average final dimension as well
        final_dims = [r['final_dim'] for r in dir_categories[category]]
        print(f"    Avg Final Dim: {np.mean(final_dims):.1f}")

def main():
    """Main analysis function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze test metrics from experiment results')
    parser.add_argument('directories', nargs='+', help='One or more paths to directories containing experiment results')
    parser.add_argument('--save-plot', default='analysis/', 
                       help='Path to save the plot (default: analysis/)')
    parser.add_argument('--labels', nargs='*', help='Optional custom labels for each directory (must match number of directories)')
    args = parser.parse_args()
    
    # Configuration
    directories = args.directories
    custom_labels = args.labels
    
    # Generate directory labels
    if custom_labels and len(custom_labels) == len(directories):
        directory_labels = custom_labels
    else:
        if custom_labels:
            print(f"Warning: Number of labels ({len(custom_labels)}) doesn't match number of directories ({len(directories)}). Using default labels.")
        directory_labels = [Path(directory).name for directory in directories]
    
    # Collect results from all directories
    all_results = []
    for i, directory in enumerate(directories):
        print(f"Collecting results from: {directory}")
        dir_results = collect_results(directory, directory_label=directory_labels[i])
        print(f"Found {len(dir_results)} runs from {directory_labels[i]}")
        all_results.extend(dir_results)
    
    # Combine all results
    results = all_results
    
    if not results:
        print("No results found from any directory!")
        return
    
    print(f"Total: {len(results)} completed runs from {len(directories)} directories combined")
    
    # # Print summary statistics
    # print_summary_stats(results)
    
    # Create plots
    print("\nCreating plots...")
    os.makedirs(args.save_plot, exist_ok=True)
    plot_test_metrics(results, save_path=args.save_plot)
    
    print(f"\nAnalysis complete!")

if __name__ == "__main__":
    main()

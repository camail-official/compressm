#!/usr/bin/env python3
"""
Analysis script for Compre-SSM paper reproduction results.

Aggregates results across seeds and generates publication-quality plots
matching the paper figures.

Usage:
    # Analyse all results in outputs/paper/
    python scripts/analyse_results.py outputs/paper/ --output results/paper_reproduction/
    
    # Analyse specific config
    python scripts/analyse_results.py outputs/paper/smnist_tau0.01/ --output results/
    
    # Generate only tables (no plots)
    python scripts/analyse_results.py outputs/paper/ --tables-only

Output:
    - summary_table.csv: Results table with mean±std across seeds
    - accuracy_vs_dim.pdf: Publication-quality accuracy vs dimension plot
    - accuracy_vs_speedup.pdf: Publication-quality accuracy vs speedup plot
"""

import argparse
import os
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np


@dataclass
class RunResult:
    """Results from a single training run."""
    config_name: str
    seed: int
    test_accuracy: float
    final_dim: float  # Average across blocks
    initial_dim: float
    training_time: float  # seconds
    reduction_history: List[dict]
    is_baseline: bool
    tolerance: Optional[float]  # Paper τ value


def load_run_result(run_dir: str) -> Optional[RunResult]:
    """Load results from a single run directory."""
    run_path = Path(run_dir)
    
    # Extract config name and seed from path
    # Expected: outputs/paper/{config_name}/seed_{N}/
    parts = run_path.parts
    try:
        seed_dir = parts[-1]
        config_name = parts[-2]
        seed = int(seed_dir.replace("seed_", ""))
    except (IndexError, ValueError):
        print(f"Warning: Could not parse run directory: {run_dir}")
        return None
    
    # Load test metric
    test_metric_path = run_path / "test_metric.npy"
    if not test_metric_path.exists():
        print(f"Warning: No test_metric.npy in {run_dir}")
        return None
    
    test_metric = np.load(test_metric_path)
    if test_metric.ndim == 0:
        test_accuracy = float(test_metric)
    else:
        test_accuracy = float(test_metric[0]) if len(test_metric) > 0 else 0.0
    
    # Load reduction history
    reduction_history = []
    reduction_path = run_path / "reduction_history.npy"
    if reduction_path.exists():
        reduction_history = np.load(reduction_path, allow_pickle=True).tolist()
        if not isinstance(reduction_history, list):
            reduction_history = []
    
    # Load config/data.json to get initial dimensions
    config_path = run_path / "config.json"
    data_path = run_path / "data.json"
    initial_dim = 256  # Default
    num_blocks = 1
    
    for cfg_file in [config_path, data_path]:
        if cfg_file.exists():
            try:
                with open(cfg_file) as f:
                    cfg = json.load(f)
                initial_dim = cfg.get("model", {}).get("state_dim", cfg.get("ssm_dim", 256))
                if isinstance(initial_dim, str):
                    initial_dim = int(initial_dim)
                num_blocks = cfg.get("model", {}).get("num_blocks", cfg.get("num_blocks", 1))
                if isinstance(num_blocks, str):
                    num_blocks = int(num_blocks)
                break
            except (json.JSONDecodeError, KeyError):
                pass
    
    # Calculate final dimension from reduction history
    # Track last dimension for each block
    block_dims = {i: initial_dim for i in range(num_blocks)}
    for entry in reduction_history:
        if isinstance(entry, dict) and 'block' in entry and 'new_dim' in entry:
            block_dims[entry['block']] = entry['new_dim']
    
    final_dim = sum(block_dims.values()) / num_blocks
    
    # Load training time if available
    training_time = 0.0
    time_path = run_path / "all_time.npy"
    if time_path.exists():
        times = np.load(time_path)
        training_time = np.sum(times)
    
    # Parse config name to get tolerance
    is_baseline = "baseline" in config_name
    tolerance = None
    if not is_baseline:
        # Parse tau from config name (e.g., smnist_tau0.01)
        tau_match = re.search(r"tau(\d+\.?\d*)", config_name)
        if tau_match:
            tolerance = float(tau_match.group(1))
    
    return RunResult(
        config_name=config_name,
        seed=seed,
        test_accuracy=test_accuracy,
        final_dim=final_dim,
        initial_dim=initial_dim,
        training_time=training_time,
        reduction_history=reduction_history,
        is_baseline=is_baseline,
        tolerance=tolerance,
    )


def discover_runs(results_dir: str) -> Dict[str, List[RunResult]]:
    """Discover all run directories and group by config name."""
    results_path = Path(results_dir)
    grouped = {}
    
    # Look for pattern: {results_dir}/{config_name}/seed_{N}/
    for config_dir in results_path.iterdir():
        if not config_dir.is_dir():
            continue
        
        config_name = config_dir.name
        runs = []
        
        for seed_dir in config_dir.iterdir():
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                continue
            
            result = load_run_result(str(seed_dir))
            if result is not None:
                runs.append(result)
        
        if runs:
            grouped[config_name] = runs
    
    return grouped


@dataclass
class AggregatedResult:
    """Aggregated results across seeds."""
    config_name: str
    n_seeds: int
    accuracy_mean: float
    accuracy_std: float
    accuracy_top3_mean: float
    final_dim_mean: float
    final_dim_std: float
    initial_dim: float
    time_mean: float
    is_baseline: bool
    tolerance: Optional[float]


def aggregate_results(runs: List[RunResult]) -> AggregatedResult:
    """Aggregate results across seeds using top-3 mean (as in paper)."""
    accuracies = [r.test_accuracy for r in runs]
    final_dims = [r.final_dim for r in runs]
    times = [r.training_time for r in runs]
    
    # Top-3 mean (paper's aggregation method)
    sorted_acc = sorted(accuracies, reverse=True)
    top3_acc = sorted_acc[:3] if len(sorted_acc) >= 3 else sorted_acc
    
    return AggregatedResult(
        config_name=runs[0].config_name,
        n_seeds=len(runs),
        accuracy_mean=np.mean(accuracies),
        accuracy_std=np.std(accuracies),
        accuracy_top3_mean=np.mean(top3_acc),
        final_dim_mean=np.mean(final_dims),
        final_dim_std=np.std(final_dims),
        initial_dim=runs[0].initial_dim,
        time_mean=np.mean(times),
        is_baseline=runs[0].is_baseline,
        tolerance=runs[0].tolerance,
    )


def generate_summary_table(
    grouped_results: Dict[str, List[RunResult]],
    output_path: str,
) -> str:
    """Generate summary table in CSV format."""
    aggregated = {}
    for config_name, runs in grouped_results.items():
        aggregated[config_name] = aggregate_results(runs)
    
    # Sort by dataset then tolerance
    def sort_key(name):
        agg = aggregated[name]
        dataset = 0 if "smnist" in name else 1
        tol = 0 if agg.is_baseline else (agg.tolerance or 0)
        return (dataset, tol)
    
    sorted_names = sorted(aggregated.keys(), key=sort_key)
    
    # Generate CSV
    lines = ["config,n_seeds,tolerance,final_dim,final_dim_std,accuracy,accuracy_std,top3_accuracy"]
    for name in sorted_names:
        agg = aggregated[name]
        tol_str = "baseline" if agg.is_baseline else f"{agg.tolerance:.2f}"
        lines.append(
            f"{name},{agg.n_seeds},{tol_str},"
            f"{agg.final_dim_mean:.1f},{agg.final_dim_std:.1f},"
            f"{agg.accuracy_mean*100:.2f},{agg.accuracy_std*100:.2f},"
            f"{agg.accuracy_top3_mean*100:.2f}"
        )
    
    csv_content = "\n".join(lines)
    
    with open(output_path, 'w') as f:
        f.write(csv_content)
    
    return csv_content


def generate_plots(
    grouped_results: Dict[str, List[RunResult]],
    output_dir: str,
):
    """Generate publication-quality plots."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
    except ImportError:
        print("Warning: matplotlib not available, skipping plots")
        return
    
    # LaTeX font configuration
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Computer Modern']
    
    # Font sizes
    TITLE_SIZE = 16
    LABEL_SIZE = 14
    TICK_SIZE = 12
    LEGEND_SIZE = 11
    
    # Aggregate results
    aggregated = {
        name: aggregate_results(runs)
        for name, runs in grouped_results.items()
    }
    
    # Separate by dataset
    smnist_results = {k: v for k, v in aggregated.items() if "smnist" in k}
    scifar_results = {k: v for k, v in aggregated.items() if "scifar" in k}
    
    # Colors
    colors = {
        'baseline': '#666666',
        'compressed': '#D62728',  # Red
    }
    
    for dataset_name, results in [("smnist", smnist_results), ("scifar", scifar_results)]:
        if not results:
            continue
        
        # Sort by tolerance (baseline first, then ascending tolerance)
        def sort_key(name):
            agg = results[name]
            return (0 if agg.is_baseline else 1, agg.tolerance or 0)
        
        sorted_names = sorted(results.keys(), key=sort_key)
        
        # ========== Accuracy vs Dimension Plot ==========
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for name in sorted_names:
            agg = results[name]
            color = colors['baseline'] if agg.is_baseline else colors['compressed']
            marker = 's' if agg.is_baseline else 'o'
            
            if agg.is_baseline:
                label = r'Baseline (full)'
            else:
                label = rf'$\tau={agg.tolerance}$'
            
            ax.errorbar(
                agg.final_dim_mean, agg.accuracy_top3_mean * 100,
                xerr=agg.final_dim_std,
                yerr=agg.accuracy_std * 100,
                fmt=marker, color=color, markersize=10,
                capsize=5, capthick=1.5, elinewidth=1.5,
                label=label,
            )
        
        ax.set_xlabel(r'State Dimension', fontsize=LABEL_SIZE)
        ax.set_ylabel(r'Test Accuracy (\%)', fontsize=LABEL_SIZE)
        ax.set_title(rf'{dataset_name.upper()} Accuracy vs. Dimension', fontsize=TITLE_SIZE)
        ax.tick_params(axis='both', labelsize=TICK_SIZE)
        ax.legend(fontsize=LEGEND_SIZE)
        ax.grid(True, alpha=0.3)
        
        # Use log scale for x-axis if range is large
        dims = [results[n].final_dim_mean for n in sorted_names]
        if max(dims) / min(dims) > 5:
            ax.set_xscale('log')
        
        plt.tight_layout()
        plot_path = Path(output_dir) / f"{dataset_name}_accuracy_vs_dim.pdf"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.savefig(plot_path.with_suffix('.png'), dpi=300, bbox_inches='tight', transparent=True)
        plt.close()
        print(f"Saved: {plot_path}")
        
        # ========== Accuracy vs Speedup Plot ==========
        # Calculate speedup relative to baseline
        baseline_time = None
        for name in sorted_names:
            if results[name].is_baseline:
                baseline_time = results[name].time_mean
                break
        
        if baseline_time and baseline_time > 0:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            for name in sorted_names:
                agg = results[name]
                speedup = baseline_time / agg.time_mean if agg.time_mean > 0 else 1.0
                color = colors['baseline'] if agg.is_baseline else colors['compressed']
                marker = 's' if agg.is_baseline else 'o'
                
                if agg.is_baseline:
                    label = r'Baseline (1$\times$)'
                else:
                    label = rf'$\tau={agg.tolerance}$ ({speedup:.1f}$\times$)'
                
                ax.errorbar(
                    speedup, agg.accuracy_top3_mean * 100,
                    yerr=agg.accuracy_std * 100,
                    fmt=marker, color=color, markersize=10,
                    capsize=5, capthick=1.5, elinewidth=1.5,
                    label=label,
                )
            
            ax.set_xlabel(r'Speedup', fontsize=LABEL_SIZE)
            ax.set_ylabel(r'Test Accuracy (\%)', fontsize=LABEL_SIZE)
            ax.set_title(rf'{dataset_name.upper()} Accuracy vs. Speedup', fontsize=TITLE_SIZE)
            ax.tick_params(axis='both', labelsize=TICK_SIZE)
            ax.legend(fontsize=LEGEND_SIZE)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = Path(output_dir) / f"{dataset_name}_accuracy_vs_speedup.pdf"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.savefig(plot_path.with_suffix('.png'), dpi=300, bbox_inches='tight', transparent=True)
            plt.close()
            print(f"Saved: {plot_path}")


def print_comparison_table(grouped_results: Dict[str, List[RunResult]]):
    """Print comparison table to console."""
    aggregated = {
        name: aggregate_results(runs)
        for name, runs in grouped_results.items()
    }
    
    # Sort by dataset then tolerance
    def sort_key(name):
        agg = aggregated[name]
        dataset = 0 if "smnist" in name else 1
        tol = 0 if agg.is_baseline else (agg.tolerance or 0)
        return (dataset, tol)
    
    sorted_names = sorted(aggregated.keys(), key=sort_key)
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"{'Config':<25} {'Seeds':<6} {'τ':<10} {'Final Dim':<15} {'Top-3 Acc':<12}")
    print("-"*80)
    
    current_dataset = None
    for name in sorted_names:
        agg = aggregated[name]
        dataset = "smnist" if "smnist" in name else "scifar"
        
        if dataset != current_dataset:
            if current_dataset is not None:
                print("-"*80)
            current_dataset = dataset
        
        tol_str = "baseline" if agg.is_baseline else f"{agg.tolerance:.2f}"
        dim_str = f"{agg.final_dim_mean:.1f}±{agg.final_dim_std:.1f}"
        acc_str = f"{agg.accuracy_top3_mean*100:.2f}%"
        
        print(f"{name:<25} {agg.n_seeds:<6} {tol_str:<10} {dim_str:<15} {acc_str:<12}")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Analyse Compre-SSM paper reproduction results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "results_dir",
        type=str,
        help="Directory containing run results (e.g., outputs/paper/)",
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for analysis results (default: {results_dir}/analysis/)",
    )
    
    parser.add_argument(
        "--tables-only",
        action="store_true",
        help="Generate only tables, skip plots",
    )
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output is None:
        output_dir = Path(args.results_dir) / "analysis"
    else:
        output_dir = Path(args.output)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Discovering runs in: {args.results_dir}")
    grouped_results = discover_runs(args.results_dir)
    
    if not grouped_results:
        print("No results found!")
        return
    
    total_runs = sum(len(runs) for runs in grouped_results.values())
    print(f"Found {len(grouped_results)} configs with {total_runs} total runs")
    
    # Print console summary
    print_comparison_table(grouped_results)
    
    # Generate CSV summary
    csv_path = output_dir / "summary_table.csv"
    csv_content = generate_summary_table(grouped_results, str(csv_path))
    print(f"\nSaved summary table: {csv_path}")
    
    # Generate plots
    if not args.tables_only:
        print("\nGenerating plots...")
        generate_plots(grouped_results, str(output_dir))
    
    print(f"\nAnalysis complete. Results in: {output_dir}")


if __name__ == "__main__":
    main()

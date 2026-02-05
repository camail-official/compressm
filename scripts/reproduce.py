#!/usr/bin/env python3
"""
Reproduction script for CompreSSM paper results.

This script runs experiments from pre-defined paper configurations.

Usage:
    # Run single config with explicit seeds
    python scripts/reproduce.py configs/paper/smnist_tau0.01.yaml --seeds 8 42 123 --gpu 0
    
    # Dry run to preview command
    python scripts/reproduce.py configs/paper/scifar_baseline.yaml --seeds 42 --dry-run
    
    # Run all seeds for sMNIST
    python scripts/reproduce.py configs/paper/smnist_tau0.01.yaml --seeds 8 42 123 456 789 101 202 303 404 505 --gpu 0

Paper seed conventions:
    - sMNIST: 10 seeds (8, 42, 123, 456, 789, 101, 202, 303, 404, 505)
    - sCIFAR: 5 seeds (8, 42, 123, 456, 789)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def get_config_name(config_path: str) -> str:
    """Extract config name from path (e.g., 'smnist_tau0.01' from 'configs/paper/smnist_tau0.01.yaml')."""
    return Path(config_path).stem


def get_output_dir(config_path: str, seed: int) -> str:
    """Generate output directory for paper reproduction runs."""
    config_name = get_config_name(config_path)
    return f"outputs/paper/{config_name}/seed_{seed}"


def run_experiment(
    config_path: str,
    seed: int,
    gpu: int = None,
    dry_run: bool = False,
    extra_args: list = None,
) -> str:
    """
    Run a single experiment with specified config and seed.
    
    Args:
        config_path: Path to YAML config file
        seed: Random seed
        gpu: GPU device ID (sets CUDA_VISIBLE_DEVICES)
        dry_run: If True, print command without executing
        extra_args: Additional command-line arguments
        
    Returns:
        Command string that was/would be run
    """
    script_dir = Path(__file__).parent
    train_script = script_dir / "train.py"
    output_dir = get_output_dir(config_path, seed)
    
    cmd = [
        sys.executable, str(train_script),
        "--config", config_path,
        "--seed", str(seed),
        "--output-dir", output_dir,
    ]
    
    if extra_args:
        cmd.extend(extra_args)
    
    cmd_str = " ".join(cmd)
    
    # Prepare environment with GPU setting
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        cmd_prefix = f"CUDA_VISIBLE_DEVICES={gpu} "
    else:
        cmd_prefix = ""
    
    full_cmd_str = cmd_prefix + cmd_str
    
    if dry_run:
        print(f"[DRY RUN] {full_cmd_str}")
    else:
        print(f"\n{'='*60}")
        print(f"Running: {full_cmd_str}")
        print(f"Output: {output_dir}")
        print('='*60)
        subprocess.run(cmd, check=True, env=env)
    
    return full_cmd_str


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce CompreSSM paper results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # sMNIST baseline with single seed
    python scripts/reproduce.py configs/paper/smnist_baseline.yaml --seeds 42 --gpu 0
    
    # sMNIST τ=0.01 with all 10 seeds
    python scripts/reproduce.py configs/paper/smnist_tau0.01.yaml \\
        --seeds 8 42 123 456 789 101 202 303 404 505 --gpu 0
    
    # sCIFAR τ=0.05 with 5 seeds (dry run)
    python scripts/reproduce.py configs/paper/scifar_tau0.05.yaml \\
        --seeds 8 42 123 456 789 --gpu 1 --dry-run
        
Paper configurations available in configs/paper/:
    smnist_baseline.yaml  - sMNIST baseline (no reduction)
    smnist_tau0.01.yaml   - sMNIST τ=0.01 (tol=0.99)
    smnist_tau0.02.yaml   - sMNIST τ=0.02 (tol=0.98)
    smnist_tau0.04.yaml   - sMNIST τ=0.04 (tol=0.96)
    scifar_baseline.yaml  - sCIFAR baseline (no reduction)
    scifar_tau0.05.yaml   - sCIFAR τ=0.05 (tol=0.95)
    scifar_tau0.10.yaml   - sCIFAR τ=0.10 (tol=0.90)
    scifar_tau0.15.yaml   - sCIFAR τ=0.15 (tol=0.85)
"""
    )
    
    parser.add_argument(
        "config",
        type=str,
        help="Path to YAML config file (e.g., configs/paper/smnist_tau0.01.yaml)",
    )
    
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        required=True,
        help="Random seeds to run (e.g., --seeds 8 42 123)",
    )
    
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU device ID (sets CUDA_VISIBLE_DEVICES)",
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    
    args = parser.parse_args()
    
    # Validate config exists
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    config_name = get_config_name(args.config)
    print(f"\n{'='*60}")
    print(f"REPRODUCING: {config_name}")
    print(f"Seeds: {args.seeds}")
    print(f"GPU: {args.gpu if args.gpu is not None else 'default'}")
    print(f"{'='*60}")
    
    # Run experiments for each seed
    commands = []
    for seed in args.seeds:
        cmd = run_experiment(
            config_path=args.config,
            seed=seed,
            gpu=args.gpu,
            dry_run=args.dry_run,
        )
        commands.append((seed, cmd))
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    print(f"Config: {config_name}")
    print(f"Runs: {len(commands)}")
    print(f"Output directory: outputs/paper/{config_name}/")
    if args.dry_run:
        print("\n[DRY RUN] No experiments were actually executed.")
    else:
        print("\nAll experiments completed.")
        print(f"Run analysis with: python scripts/analyse_results.py outputs/paper/{config_name}/")


if __name__ == "__main__":
    main()

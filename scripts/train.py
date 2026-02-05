#!/usr/bin/env python3
"""
Compre-SSM Training CLI

Train LRU models with optional in-training compression on sMNIST/sCIFAR.

Usage:
    # Train with default config
    python scripts/train.py --config configs/scifar.yaml
    
    # Train with tolerance-based reduction
    python scripts/train.py --config configs/scifar.yaml --mode tolerance --tol 0.99
    
    # Train with fixed-fraction reduction
    python scripts/train.py --config configs/scifar.yaml --mode fixed --reduction-fraction 0.1
    
    # Ablation: random state selection
    python scripts/train.py --config configs/scifar.yaml --mode tolerance --selection random
"""

import argparse
import os
import sys

import yaml
import jax.random as jr

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compressm.data.datasets import create_dataset
from compressm.training.trainer import (
    train,
    TrainingConfig,
    ModelConfig,
    ReductionConfig,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Compre-SSM models with in-training compression",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Config file
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    
    # Dataset override
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        choices=["smnist", "scifar"],
        help="Dataset name (overrides config)",
    )
    
    # Training overrides
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--num-steps", type=int, help="Training steps")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    
    # Model overrides
    parser.add_argument("--state-dim", type=int, help="State dimension per block")
    parser.add_argument("--hidden-dim", type=int, help="Hidden dimension")
    parser.add_argument("--num-blocks", type=int, help="Number of LRU blocks")
    
    # Reduction overrides
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["none", "tolerance", "fixed", "pragmatic"],
        help="Reduction mode",
    )
    parser.add_argument(
        "--selection", "-s",
        type=str,
        choices=["largest", "smallest", "random"],
        help="State selection mode for reduction",
    )
    parser.add_argument("--tol", type=float, help="Energy conservation tolerance")
    parser.add_argument("--red-start", type=int, help="Step to start reductions")
    parser.add_argument("--red-end", type=int, help="Step to stop reductions")
    parser.add_argument("--red-interval", type=int, help="Steps between reductions")
    parser.add_argument("--hsv-interval", type=int, help="Steps between HSV logging (0=off)")
    parser.add_argument("--reduction-fraction", type=float, help="Fraction to reduce")
    
    # Output
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        help="Output directory (auto-generated if not specified)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Directory for dataset downloads",
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    
    # Load base config
    config = load_config(args.config)
    
    # Apply overrides
    dataset_name = args.dataset or config.get("dataset", "scifar")
    
    # Training config with overrides
    train_cfg = config.get("training", {})
    training_config = TrainingConfig(
        num_steps=args.num_steps or train_cfg.get("num_steps", 100000),
        batch_size=args.batch_size or train_cfg.get("batch_size", 64),
        lr=args.lr or train_cfg.get("lr", 0.004),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        ssm_lr_factor=train_cfg.get("ssm_lr_factor", 0.5),
        use_warmup_cosine=train_cfg.get("use_warmup_cosine", True),
        eval_steps=train_cfg.get("eval_steps", 1000),
        seed=args.seed or train_cfg.get("seed", 42),
    )
    
    # Model config with overrides
    model_cfg = config.get("model", {})
    model_config = ModelConfig(
        num_blocks=args.num_blocks or model_cfg.get("num_blocks", 6),
        state_dim=args.state_dim or model_cfg.get("state_dim", 512),
        hidden_dim=args.hidden_dim or model_cfg.get("hidden_dim", 256),
        r_min=model_cfg.get("r_min", 0.9),
        r_max=model_cfg.get("r_max", 0.999),
        drop_rate=model_cfg.get("drop_rate", 0.1),
    )
    
    # Reduction config with overrides
    red_cfg = config.get("reduction", {})
    reduction_config = ReductionConfig(
        mode=args.mode or red_cfg.get("mode", "none"),
        selection=args.selection or red_cfg.get("selection", "largest"),
        tol=args.tol if args.tol is not None else red_cfg.get("tol", 0.99),
        red_start=args.red_start if args.red_start is not None else red_cfg.get("red_start", 0),
        red_end=args.red_end if args.red_end is not None else red_cfg.get("red_end", 100000),
        red_interval=args.red_interval if args.red_interval is not None else red_cfg.get("red_interval", 2000),
        hsv_interval=args.hsv_interval if args.hsv_interval is not None else red_cfg.get("hsv_interval", 0),
        reduction_fraction=args.reduction_fraction if args.reduction_fraction is not None else red_cfg.get("reduction_fraction", 0.1),
        performance_tolerance=red_cfg.get("performance_tolerance", 0.02),
        method=red_cfg.get("method", "sqrtm"),
    )
    
    # Print configuration
    print("\n" + "="*60)
    print("Compre-SSM Training")
    print("="*60)
    print(f"\nDataset: {dataset_name}")
    print(f"Model: {model_config.num_blocks} blocks × {model_config.state_dim} state dim × {model_config.hidden_dim} hidden dim")
    print(f"Reduction mode: {reduction_config.mode}")
    if reduction_config.mode != "none":
        print(f"  Selection: {reduction_config.selection}")
        if reduction_config.mode == "tolerance":
            print(f"  Tolerance: {reduction_config.tol}")
        else:
            print(f"  Fraction: {reduction_config.reduction_fraction}")
    print(f"Training: {training_config.num_steps} steps, lr={training_config.lr}")
    print(f"Seed: {training_config.seed}")
    print("="*60 + "\n")
    
    # Create dataset
    dataset_key = jr.PRNGKey(training_config.seed)
    data_dir = args.data_dir or config.get("data_dir", "./data")
    dataset = create_dataset(dataset_name, key=dataset_key, data_dir=data_dir)
    
    # Train
    model = train(
        dataset=dataset,
        training_config=training_config,
        model_config=model_config,
        reduction_config=reduction_config,
        output_dir=args.output_dir,
    )
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()

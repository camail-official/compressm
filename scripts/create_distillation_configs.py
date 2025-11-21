"""
Script to create distillation config files based on existing non-reduced-adapted configs.
"""

import json
import os
from pathlib import Path

# Teacher checkpoint path
TEACHER_CHECKPOINT = "outputs/lru/imdb_stuff/lr_0.001_num_blocks_1_hidden_dim_256_vf_depth_None_vf_width_None_ssm_dim_192_ssm_blocks_None_scale_1_lambd_None_use_embedding_True_drop_rate_0.2_dual_False_seed_123_warmup_3000/best_model.eqx"

# Distillation hyperparameters
TEMPERATURE = 2.0
ALPHA = 0.5

# Seed to use (should match teacher)
SEED = 123

# Student dimensions
STUDENT_DIMS = [95, 119, 136, 150, 165]

# Base config directory
BASE_DIR = Path("experiment_configs/repeats/lru/imdb/non-reduced-adapted")
OUTPUT_DIR = Path("experiment_configs/repeats/lru/imdb/distilled")

def create_distillation_config(student_dim, seed=SEED):
    """Create a distillation config file for a given student dimension."""
    
    # Load base config
    base_config_path = BASE_DIR / f"imdb-{seed}-{student_dim}.json"
    
    if not base_config_path.exists():
        print(f"Warning: Base config not found: {base_config_path}")
        return
    
    with open(base_config_path, 'r') as f:
        config = json.load(f)
    
    # Add distillation parameters
    config["teacher_checkpoint"] = TEACHER_CHECKPOINT
    config["distill_temperature"] = TEMPERATURE
    config["distill_alpha"] = ALPHA
    
    # Disable reduction during distillation (optional, you can keep it if you want)
    config["tol"] = -1
    config["red_steps"] = 0
    config["red_wait_steps"] = 0
    
    # Save distillation config
    output_path = OUTPUT_DIR / f"imdb-{seed}-{student_dim}.json"
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Created: {output_path}")

def main():
    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating distillation configs for seed {SEED}")
    print(f"Teacher checkpoint: {TEACHER_CHECKPOINT}")
    print(f"Temperature: {TEMPERATURE}, Alpha: {ALPHA}")
    print()
    
    for dim in STUDENT_DIMS:
        create_distillation_config(dim, SEED)
    
    print()
    print(f"Created {len(STUDENT_DIMS)} distillation config files in {OUTPUT_DIR}")
    print()
    print("To run distillation experiments:")
    print(f"  python -m run_experiment --dataset_name imdb/distilled/imdb-{SEED}-119")

if __name__ == "__main__":
    main()

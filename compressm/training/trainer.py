"""
Unified training loop for Compre-SSM.

This module provides the main training function with integrated
support for different reduction modes:
- tolerance: Reduce based on Hankel energy conservation threshold
- fixed: Reduce by fixed percentage at regular intervals
- pragmatic: Fixed reduction with performance-based rollback
- none: No reduction (baseline training)
"""

import os
import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Dict, Any, List

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import yaml
from tqdm import tqdm

from compressm.models.lru import LRU
from compressm.reduction.hsv import SelectionMode
from compressm.data.datasets import Dataset
from compressm.training.utils import (
    create_optimizer,
    truncate_optimizer_state,
    classification_loss,
    make_step,
    compute_accuracy,
)


@dataclass
class ReductionConfig:
    """
    Configuration for in-training model reduction.
    
    Attributes:
        mode: Reduction strategy
            - "none": No reduction (baseline)
            - "tolerance": Reduce to preserve Hankel energy fraction
            - "fixed": Reduce by fixed percentage at intervals
            - "pragmatic": Fixed reduction with rollback on degradation
        selection: Which states to keep during reduction
            - "largest": Keep highest HSV states (recommended)
            - "smallest": Keep lowest HSV states (ablation)
            - "random": Random selection (ablation)
        tol: Fraction of Hankel energy to preserve (e.g., 0.95 = keep 95% of energy)
             Higher values = less aggressive reduction, more states kept
        red_start: Training step to begin reductions (default: 0)
        red_end: Training step to stop reductions (e.g., 20000 for 10% of 200k)
        red_interval: Steps between reduction checks (all modes)
        hsv_interval: Steps between HSV logging (for analysis plots, 0 = off)
        reduction_fraction: Fraction to reduce each time (fixed/pragmatic modes)
        performance_tolerance: Max allowed accuracy drop for pragmatic rollback
        method: Balanced truncation method ("sqrtm" or "chol")
    """
    mode: str = "none"
    selection: str = "largest"
    tol: float = 0.95
    red_start: int = 0
    red_end: int = 100000
    red_interval: int = 2000
    hsv_interval: int = 0
    reduction_fraction: float = 0.1
    performance_tolerance: float = 0.02
    method: str = "sqrtm"
    
    def get_selection_mode(self) -> SelectionMode:
        """Convert string selection to SelectionMode enum."""
        return SelectionMode(self.selection)


@dataclass
class TrainingConfig:
    """
    Full training configuration.
    
    Attributes:
        num_steps: Total training steps
        batch_size: Batch size
        lr: Base learning rate
        weight_decay: Weight decay for AdamW
        ssm_lr_factor: Learning rate multiplier for SSM parameters
        use_warmup_cosine: Use warmup + cosine annealing schedule
        eval_steps: How often to evaluate and print metrics
        seed: Random seed
    """
    num_steps: int = 100000
    batch_size: int = 64
    lr: float = 0.004
    weight_decay: float = 0.01
    ssm_lr_factor: float = 0.5
    use_warmup_cosine: bool = True
    eval_steps: int = 1000
    seed: int = 42


@dataclass
class ModelConfig:
    """
    LRU model configuration.
    
    Attributes:
        num_blocks: Number of LRU blocks
        state_dim: Dimension of recurrent state (N)
        hidden_dim: Model hidden dimension (H)
        r_min: Minimum eigenvalue magnitude
        r_max: Maximum eigenvalue magnitude
        drop_rate: Dropout probability
    """
    num_blocks: int = 6
    state_dim: int = 512
    hidden_dim: int = 256
    r_min: float = 0.9
    r_max: float = 0.999
    drop_rate: float = 0.1


def save_config(
    output_dir: str,
    training_config: TrainingConfig,
    model_config: ModelConfig,
    reduction_config: ReductionConfig,
    dataset_name: str
):
    """Save all configs to a YAML file."""
    config = {
        "dataset": dataset_name,
        "training": asdict(training_config),
        "model": asdict(model_config),
        "reduction": asdict(reduction_config),
    }
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def save_metrics(output_dir: str, metrics: Dict[str, Any]):
    """Save training metrics to JSON."""
    # Convert numpy arrays to lists for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.ndarray, jnp.ndarray)):
            return obj.tolist()
        elif isinstance(obj, (np.floating, jnp.floating)):
            return float(obj)
        elif isinstance(obj, (np.integer, jnp.integer)):
            return int(obj)
        return obj
    
    serializable = {k: convert(v) for k, v in metrics.items()}
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(serializable, f, indent=2)


def train(
    dataset: Dataset,
    training_config: TrainingConfig,
    model_config: ModelConfig,
    reduction_config: ReductionConfig,
    output_dir: Optional[str] = None,
) -> LRU:
    """
    Train an LRU model with optional in-training compression.
    
    This is the main training function that supports all reduction modes
    and handles the complete training loop including:
    - Model initialization
    - Optimizer setup with differential learning rates
    - Training loop with periodic evaluation
    - Reduction based on configured mode
    - Metric logging and checkpointing
    
    Args:
        dataset: Dataset object with train/val/test dataloaders
        training_config: Training hyperparameters
        model_config: Model architecture config
        reduction_config: Reduction strategy config
        output_dir: Directory for outputs (auto-generated if None)
        
    Returns:
        Trained LRU model
    """
    # Setup output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"outputs/{dataset.name}/{reduction_config.mode}/{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configs
    save_config(output_dir, training_config, model_config, reduction_config, dataset.name)
    
    # Initialize random keys
    key = jr.PRNGKey(training_config.seed)
    model_key, train_key, batch_key = jr.split(key, 3)
    
    # Create model
    model = LRU(
        num_blocks=model_config.num_blocks,
        input_dim=dataset.input_dim,
        state_dim=model_config.state_dim,
        hidden_dim=model_config.hidden_dim,
        output_dim=dataset.output_dim,
        classification=True,
        r_min=model_config.r_min,
        r_max=model_config.r_max,
        drop_rate=model_config.drop_rate,
        key=model_key,
    )
    
    # Get batch norm state (for stateful modules like BatchNorm)
    state = eqx.nn.State(model)
    
    # Create optimizer
    use_multi = training_config.ssm_lr_factor != 1.0
    opt = create_optimizer(
        lr=training_config.lr,
        num_steps=training_config.num_steps,
        weight_decay=training_config.weight_decay,
        ssm_lr_factor=training_config.ssm_lr_factor,
        use_warmup_cosine=training_config.use_warmup_cosine,
    )
    
    # Initialize optimizer state
    model_params = eqx.filter(model, eqx.is_inexact_array)
    if use_multi:
        opt_state = opt.init([model_params])
    else:
        opt_state = opt.init(model_params)
    
    # Filter spec for training
    filter_spec = jax.tree_util.tree_map(lambda _: True, model)
    
    # Training state
    train_metrics: List[float] = []
    val_metrics: List[float] = []
    test_metrics: List[float] = []
    steps_recorded: List[int] = []
    reduction_history: List[Dict[str, Any]] = []
    hankel_singular_values: Dict[int, Dict[str, Any]] = {}  # step -> {block_i: {g: array}}
    best_val_acc = 0.0
    best_test_acc = 0.0
    best_step = 0
    
    # For pragmatic mode
    checkpoint_model = None
    checkpoint_opt_state = None
    checkpoint_val_acc = 0.0
    reductions_stopped = False
    
    # Training loop
    print(f"\nTraining on {dataset.name} with {reduction_config.mode} reduction mode")
    print(f"Initial state dimensions: {model.get_state_dims()}")
    print(f"Output directory: {output_dir}\n")
    
    running_loss = 0.0
    
    with tqdm(total=training_config.num_steps, desc="Training", ncols=120) as pbar:
        for step, (X, y) in zip(
            range(training_config.num_steps),
            dataset.dataloaders["train"].loop(training_config.batch_size, key=batch_key),
        ):
            # Training step
            train_key, step_key = jr.split(train_key)
            model, state, opt_state, loss = make_step(
                model, filter_spec, X, y, classification_loss,
                state, opt, opt_state, step_key, use_multi_optimizer=use_multi
            )
            running_loss += float(loss)
            
            # Periodic evaluation and reduction
            if (step + 1) % training_config.eval_steps == 0:
                train_key, eval_key = jr.split(train_key)
                
                # Evaluate
                train_acc = compute_accuracy(
                    model, dataset.dataloaders["train"], state,
                    training_config.batch_size, eval_key
                )
                val_acc = compute_accuracy(
                    model, dataset.dataloaders["val"], state,
                    training_config.batch_size, eval_key
                )
                
                # Save HSVs if requested (for analysis plots)
                if reduction_config.hsv_interval > 0 and (step + 1) % reduction_config.hsv_interval == 0:
                    hsv_dict_for_save = model.get_all_hankel_singular_values()
                    # Store only the g values (Hankel singular values), not P/Q matrices
                    hankel_singular_values[step + 1] = {
                        f'block_{i}': {'g': np.asarray(hsv_dict_for_save[f'block_{i}']['g'])}
                        for i in range(len(model.blocks))
                    }
                
                # Apply reduction based on mode
                reduction_applied = False
                
                # Check if we're in the reduction window and at a reduction interval
                in_reduction_window = reduction_config.red_start <= step < reduction_config.red_end
                at_reduction_interval = (step + 1) % reduction_config.red_interval == 0
                should_try_reduction = in_reduction_window and at_reduction_interval and not reductions_stopped
                
                if should_try_reduction:
                    
                    if reduction_config.mode == "tolerance":
                        # Tolerance-based reduction
                        hsv_dict = model.get_all_hankel_singular_values()
                        analysis = model.get_reduction_analysis(hsv_dict, reduction_config.tol)
                        
                        ranks = []
                        for i, block in enumerate(model.blocks):
                            rec_rank = analysis[f'block_{i}']['recommended_ranks']['threshold']
                            current_rank = block.state_dim
                            # Only reduce if new dim < 95% of current (5% gate)
                            if rec_rank < 0.95 * current_rank:
                                reduction_applied = True
                                ranks.append(rec_rank)
                                reduction_history.append({
                                    'step': step + 1,
                                    'block': i,
                                    'old_dim': current_rank,
                                    'new_dim': rec_rank,
                                })
                            else:
                                ranks.append(current_rank)
                        
                        if reduction_applied:
                            model = model.reduce(
                                ranks, hsv_dict,
                                method=reduction_config.method,
                                selection=reduction_config.get_selection_mode()
                            )
                    
                    elif reduction_config.mode == "fixed":
                        # Fixed fraction reduction
                        hsv_dict = model.get_all_hankel_singular_values()
                        
                        ranks = []
                        for i, block in enumerate(model.blocks):
                            current_rank = block.state_dim
                            new_rank = max(1, int(current_rank * (1 - reduction_config.reduction_fraction)))
                            # Only reduce if new dim < 95% of current (5% gate)
                            if new_rank < 0.95 * current_rank:
                                reduction_applied = True
                                ranks.append(new_rank)
                                reduction_history.append({
                                    'step': step + 1,
                                    'block': i,
                                    'old_dim': current_rank,
                                    'new_dim': new_rank,
                                })
                            else:
                                ranks.append(current_rank)
                        
                        if reduction_applied:
                            model = model.reduce(
                                ranks, hsv_dict,
                                method=reduction_config.method,
                                selection=reduction_config.get_selection_mode()
                            )
                    
                    elif reduction_config.mode == "pragmatic":
                        # Pragmatic: reduce but rollback if performance degrades
                        # Check if we should rollback previous reduction
                        if checkpoint_model is not None:
                            if val_acc < checkpoint_val_acc - reduction_config.performance_tolerance:
                                # Rollback
                                print(f"\n  Rollback: val_acc {val_acc:.4f} < {checkpoint_val_acc:.4f} - {reduction_config.performance_tolerance}")
                                model = checkpoint_model
                                opt_state = checkpoint_opt_state
                                reductions_stopped = True
                                continue
                        
                        # Save checkpoint before reduction
                        checkpoint_model = model
                        checkpoint_opt_state = opt_state
                        checkpoint_val_acc = val_acc
                        
                        # Apply reduction
                        hsv_dict = model.get_all_hankel_singular_values()
                        
                        ranks = []
                        for i, block in enumerate(model.blocks):
                            current_rank = block.state_dim
                            new_rank = max(1, int(current_rank * (1 - reduction_config.reduction_fraction)))
                            # Only reduce if new dim < 95% of current (5% gate)
                            if new_rank < 0.95 * current_rank:
                                reduction_applied = True
                                ranks.append(new_rank)
                                reduction_history.append({
                                    'step': step + 1,
                                    'block': i,
                                    'old_dim': current_rank,
                                    'new_dim': new_rank,
                                })
                            else:
                                ranks.append(current_rank)
                        
                        if reduction_applied:
                            model = model.reduce(
                                ranks, hsv_dict,
                                method=reduction_config.method,
                                selection=reduction_config.get_selection_mode()
                            )
                
                # Reinitialize optimizer if reduction was applied
                if reduction_applied:
                    model_params = eqx.filter(model, eqx.is_inexact_array)
                    new_opt = create_optimizer(
                        lr=training_config.lr,
                        num_steps=training_config.num_steps,
                        weight_decay=training_config.weight_decay,
                        ssm_lr_factor=training_config.ssm_lr_factor,
                        use_warmup_cosine=training_config.use_warmup_cosine,
                    )
                    if use_multi:
                        new_opt_state = new_opt.init([model_params])
                    else:
                        new_opt_state = new_opt.init(model_params)
                    opt_state = truncate_optimizer_state(opt_state, new_opt_state)
                    opt = new_opt
                
                # Test evaluation on improvement or reduction
                if val_acc > best_val_acc or reduction_applied:
                    best_val_acc = val_acc
                    best_test_acc = compute_accuracy(
                        model, dataset.dataloaders["test"], state,
                        training_config.batch_size, eval_key
                    )
                    best_step = step + 1
                
                # Record metrics
                train_metrics.append(train_acc)
                val_metrics.append(val_acc)
                test_metrics.append(best_test_acc)
                steps_recorded.append(step + 1)
                
                # Update progress bar
                avg_dim = sum(model.get_state_dims()) / len(model.get_state_dims())
                pbar.set_postfix({
                    'Loss': f'{running_loss:.2f}',
                    'Train': f'{train_acc*100:.1f}%',
                    'Val': f'{val_acc*100:.1f}%',
                    'Test': f'{best_test_acc*100:.1f}%',
                    'Dim': f'{avg_dim:.0f}',
                })
                
                running_loss = 0.0
            
            pbar.update(1)
    
    # Save final metrics
    metrics = {
        'steps': steps_recorded,
        'train_accuracy': train_metrics,
        'val_accuracy': val_metrics,
        'test_accuracy': test_metrics,
        'best_test_accuracy': best_test_acc,
        'best_step': best_step,
        'final_state_dims': model.get_state_dims(),
        'reduction_history': reduction_history,
    }
    save_metrics(output_dir, metrics)
    
    # Save test metric for analysis script (legacy format)
    np.save(os.path.join(output_dir, "test_metric.npy"), best_test_acc)
    
    # Save reduction history separately for analysis
    np.save(os.path.join(output_dir, "reduction_history.npy"), reduction_history)
    
    # Save config for analysis
    config_dict = {
        'model': {
            'num_blocks': model_config.num_blocks,
            'state_dim': model_config.state_dim,
            'hidden_dim': model_config.hidden_dim,
        },
        'training': {
            'num_steps': training_config.num_steps,
            'lr': training_config.lr,
            'seed': training_config.seed,
        },
        'reduction': {
            'mode': reduction_config.mode,
            'tol': reduction_config.tol,
        },
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)
    
    # Save Hankel singular values if collected
    if reduction_config.hsv_interval > 0 and hankel_singular_values:
        np.save(os.path.join(output_dir, "all_hankel_singular_values.npy"), hankel_singular_values)
        print(f"Saved HSVs at {len(hankel_singular_values)} checkpoints")
    
    print(f"\nTraining complete!")
    print(f"Best test accuracy: {best_test_acc*100:.2f}% at step {best_step}")
    print(f"Final state dimensions: {model.get_state_dims()}")
    print(f"Total state dim: {model.get_total_state_dim()} (started at {model_config.state_dim * model_config.num_blocks})")
    
    return model

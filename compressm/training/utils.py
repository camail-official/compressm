"""
Training utilities for Compre-SSM.

This module provides optimizer creation, loss functions, and
the core training step function.
"""

from dataclasses import dataclass
from typing import Optional, Callable, Any, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax


@dataclass
class TrainState:
    """Container for training state."""
    model: eqx.Module
    opt_state: optax.OptState
    batch_norm_state: Any
    step: int


def create_warmup_cosine_schedule(
    peak_lr: float,
    num_steps: int,
    warmup_ratio: float = 0.1,
    final_lr: float = 1e-7
) -> optax.Schedule:
    """
    Create warmup + cosine annealing learning rate schedule.
    
    Args:
        peak_lr: Peak learning rate after warmup
        num_steps: Total training steps
        warmup_ratio: Fraction of steps for warmup (default 10%)
        final_lr: Final learning rate after decay
        
    Returns:
        Optax schedule function
    """
    warmup_steps = int(num_steps * warmup_ratio)
    cosine_steps = num_steps - warmup_steps
    
    warmup_schedule = optax.linear_schedule(
        init_value=1e-7,
        end_value=peak_lr,
        transition_steps=warmup_steps
    )
    
    cosine_schedule = optax.cosine_decay_schedule(
        init_value=peak_lr,
        decay_steps=cosine_steps,
        alpha=final_lr / peak_lr
    )
    
    return optax.join_schedules(
        schedules=[warmup_schedule, cosine_schedule],
        boundaries=[warmup_steps]
    )


def create_ssm_label_fn():
    """
    Create label function for multi-transform optimizer.
    
    Labels SSM parameters (nu_log, theta_log, B_re, B_im, gamma_log)
    separately from other parameters for differential learning rates.
    """
    ssm_params = ['nu_log', 'theta_log', 'B_re', 'B_im', 'gamma_log']
    
    def label_fn(listed_params):
        if not isinstance(listed_params, list):
            listed_params = [listed_params]
        
        params = listed_params[0]
        
        def get_label(path, param):
            path_str = '.'.join(str(key) for key in path)
            if any(p in path_str for p in ssm_params):
                return 'ssm'
            return 'main'
        
        labels = jax.tree_util.tree_map_with_path(get_label, params)
        return [labels]
    
    return label_fn


def create_optimizer(
    lr: float,
    num_steps: int,
    weight_decay: float = 0.01,
    ssm_lr_factor: float = 1.0,
    use_warmup_cosine: bool = True
) -> optax.GradientTransformation:
    """
    Create optimizer with optional differential learning rates.
    
    Args:
        lr: Base learning rate
        num_steps: Total training steps
        weight_decay: Weight decay coefficient
        ssm_lr_factor: Learning rate multiplier for SSM parameters
        use_warmup_cosine: Whether to use warmup + cosine schedule
        
    Returns:
        Configured optax optimizer
    """
    if use_warmup_cosine:
        main_schedule = create_warmup_cosine_schedule(lr, num_steps)
        ssm_schedule = create_warmup_cosine_schedule(lr * ssm_lr_factor, num_steps)
    else:
        main_schedule = lr
        ssm_schedule = lr * ssm_lr_factor
    
    if ssm_lr_factor != 1.0:
        # Multi-transform optimizer for differential learning rates
        label_fn = create_ssm_label_fn()
        
        main_optimizer = optax.adamw(learning_rate=main_schedule, weight_decay=weight_decay)
        ssm_optimizer = optax.adamw(learning_rate=ssm_schedule, weight_decay=0.0)
        
        optimizer = optax.multi_transform(
            {'ssm': ssm_optimizer, 'main': main_optimizer},
            label_fn
        )
    else:
        # Single optimizer
        optimizer = optax.adamw(learning_rate=main_schedule, weight_decay=weight_decay)
    
    return optimizer


def truncate_optimizer_state(old_state: optax.OptState, new_state: optax.OptState) -> optax.OptState:
    """
    Transfer optimizer state to reduced model dimensions.
    
    When model dimensions change during reduction, we need to
    reinitialize the optimizer but preserve momentum where possible.
    
    Args:
        old_state: Optimizer state from before reduction
        new_state: Freshly initialized optimizer state for reduced model
        
    Returns:
        New optimizer state with preserved momentum where dimensions match
    """
    def truncate(old, new):
        if isinstance(old, jax.Array) and isinstance(new, jax.Array):
            if old.shape == new.shape:
                return old
            elif len(old.shape) == len(new.shape):
                # Try to truncate along all dimensions
                can_truncate = all(
                    old_dim >= new_dim 
                    for old_dim, new_dim in zip(old.shape, new.shape)
                )
                if can_truncate:
                    slices = tuple(
                        slice(None, new_dim) if new_dim < old_dim else slice(None)
                        for old_dim, new_dim in zip(old.shape, new.shape)
                    )
                    return old[slices]
        return new
    
    try:
        return jax.tree_util.tree_map(truncate, old_state, new_state)
    except Exception:
        return new_state


@eqx.filter_jit
def calc_output(
    model: eqx.Module,
    X: jnp.ndarray,
    state,
    key: jax.random.PRNGKey
) -> Tuple[jnp.ndarray, Any]:
    """
    Compute model output with batching.
    
    Args:
        model: The LRU model
        X: Input batch, shape (batch_size, seq_len, input_dim)
           OR (batch_size, 2, seq_len, input_dim) if dual
        state: Batch norm state
        key: Random key for dropout
        
    Returns:
        predictions: Model outputs
        state: Updated batch norm state
    """
def calc_output(
    model: eqx.Module,
    X: jnp.ndarray,
    state,
    key: jax.random.PRNGKey
) -> Tuple[jnp.ndarray, Any]:
    """
    Compute model output with batching.
    
    Args:
        model: The LRU model
        X: Input batch, shape (batch_size, seq_len, input_dim)
           OR (batch_size, 2, seq_len, input_dim) if dual
        state: Batch norm state
        key: Random key for dropout
        
    Returns:
        predictions: Model outputs
        state: Updated batch norm state
    """
    if hasattr(model, "dual_head") and model.dual_head is not None:
        # X is (B, 2, L, D)
        B, dual_factor, L, D = X.shape
        assert dual_factor == 2
        X_reshaped = X.reshape(B * dual_factor, L, D)
        
        # Process through model blocks (which return features because dual_head is present)
        # vmap over 2B sequences
        features, state = jax.vmap(
            model, axis_name="batch", in_axes=(0, None, None), out_axes=(0, None)
        )(X_reshaped, state, key)
        
        # features is (2B, H)
        # Reshape back to (B, 2, H) and concatenate to (B, 2*H)
        features_combined = features.reshape(B, 2, -1).reshape(B, -1)
        
        # Final classification head
        output = jax.vmap(model.dual_head)(features_combined)
        output = jax.nn.softmax(output, axis=1)
    else:
        # Normal processing
        output, state = jax.vmap(
            model, axis_name="batch", in_axes=(0, None, None), out_axes=(0, None)
        )(X, state, key)
        
    return output, state


@eqx.filter_jit
@eqx.filter_value_and_grad(has_aux=True)
def classification_loss(
    diff_model: eqx.Module,
    static_model: eqx.Module,
    X: jnp.ndarray,
    y: jnp.ndarray,
    state,
    key: jax.random.PRNGKey
) -> Tuple[Tuple[float, Any], eqx.Module]:
    """
    Compute cross-entropy loss for classification.
    
    Args:
        diff_model: Differentiable part of model
        static_model: Static part of model
        X: Input batch
        y: Target labels (one-hot)
        state: Batch norm state
        key: Random key
        
    Returns:
        (loss, state), gradients
    """
    model = eqx.combine(diff_model, static_model)
    pred_y, state = calc_output(model, X, state, key)
    loss = jnp.mean(-jnp.sum(y * jnp.log(pred_y + 1e-8), axis=1))
    return loss, state


@eqx.filter_jit
def make_step(
    model: eqx.Module,
    filter_spec,
    X: jnp.ndarray,
    y: jnp.ndarray,
    loss_fn: Callable,
    state,
    opt: optax.GradientTransformation,
    opt_state: optax.OptState,
    key: jax.random.PRNGKey,
    use_multi_optimizer: bool = False
) -> Tuple[eqx.Module, Any, optax.OptState, float]:
    """
    Perform a single training step.
    
    Args:
        model: The model to train
        filter_spec: Specifies which parameters are differentiable
        X: Input batch
        y: Target batch
        loss_fn: Loss function
        state: Batch norm state
        opt: Optimizer
        opt_state: Optimizer state
        key: Random key
        use_multi_optimizer: Whether using multi-transform optimizer
        
    Returns:
        Updated model, state, optimizer state, and loss value
    """
    diff_model, static_model = eqx.partition(model, filter_spec)
    (value, state), grads = loss_fn(diff_model, static_model, X, y, state, key)
    params = eqx.filter(model, eqx.is_inexact_array)
    
    if use_multi_optimizer:
        updates, opt_state = opt.update([grads], opt_state, [params])
        model = eqx.apply_updates(model, updates[0])
    else:
        updates, opt_state = opt.update(grads, opt_state, params)
        model = eqx.apply_updates(model, updates)
    
    return model, state, opt_state, value


def compute_accuracy(
    model: eqx.Module,
    dataloader,
    state,
    batch_size: int,
    key: jax.random.PRNGKey
) -> float:
    """
    Compute classification accuracy over a dataloader.
    
    Args:
        model: The model to evaluate
        dataloader: Dataloader to iterate over
        state: Batch norm state
        batch_size: Batch size for evaluation
        key: Random key
        
    Returns:
        Accuracy as a float between 0 and 1
    """
    inference_model = eqx.tree_inference(model, value=True)
    
    predictions = []
    labels = []
    
    for X, y in dataloader.loop_epoch(batch_size):
        key, subkey = jax.random.split(key)
        pred, _ = calc_output(inference_model, X, state, subkey)
        predictions.append(pred)
        labels.append(y)
    
    predictions = jnp.vstack(predictions)
    labels = jnp.vstack(labels)
    
    accuracy = jnp.mean(jnp.argmax(predictions, axis=1) == jnp.argmax(labels, axis=1))
    return float(accuracy)

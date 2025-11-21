import equinox as eqx
import jax
import jax.numpy as jnp
import optax

def create_warmup_cosine_schedule(peak_lr, num_steps, warmup_ratio=0.1, final_lr=1e-7):
    """
    Creates warmup + cosine annealing schedule.
    
    Args:
        peak_lr: Peak learning rate to reach after warmup
        num_steps: Total number of training steps
        warmup_ratio: Fraction of training for warmup (default 0.1 = 10%)
        final_lr: Final learning rate after cosine decay (default 1e-7)
    Returns:
        Optax schedule function
    """
    warmup_steps = int(num_steps * warmup_ratio)
    cosine_steps = num_steps - warmup_steps
    
    # Create individual schedules
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
    
    # Join the schedules
    schedule = optax.join_schedules(
        schedules=[warmup_schedule, cosine_schedule],
        boundaries=[warmup_steps]
    )
    
    return schedule
    

def create_ssm_label_fn(model_name):
    """
    Create a label function that identifies SSM parameters for multi-transform optimizer.
    Uses the length-1 list wrapping trick to avoid callable issues with Equinox models.
    
    Args:
        model_name: Name of the model architecture
    
    Returns:
        Label function for optax.multi_transform (returns length-1 list)
    """
    def label_fn(listed_params):
        """
        Label function that works with length-1 list wrapped parameters.
        """
        if not isinstance(listed_params, list):
            listed_params = [listed_params]
        
        params = listed_params[0]  # Extract from length-1 list
        
        def get_label(path, param):
            # Convert path to string for pattern matching
            path_str = '.'.join(str(key) for key in path)
            
            label = 'main'  # default
            if model_name.lower() == 'lru':
                # LRU-specific SSM parameters
                ssm_params = ['nu_log', 'theta_log', 'B_re', 'B_im', 'gamma_log']
                if any(ssm_param in path_str for ssm_param in ssm_params):
                    label = 'ssm'
            elif model_name.lower() == 's5':
                # S5-specific SSM parameters
                ssm_params = ['Lambda', 'B', 'C', 'log_Lambda']
                if any(ssm_param in path_str for ssm_param in ssm_params):
                    label = 'ssm'
            elif model_name.lower() == 'rnn':
                # RNN-specific parameters
                ssm_params = ['recurrent', 'hidden_to_hidden', 'state']
                if any(ssm_param in path_str for ssm_param in ssm_params):
                    label = 'ssm'
            
            return label
        
        labels = jax.tree_util.tree_map_with_path(get_label, params)
        
        return [labels]  # Return as length-1 list
    
    return label_fn


def create_multi_optimizer_with_equinox(model_name, base_lr, ssm_lr_factor, weight_decay, num_steps, use_warmup_cosine):
    """
    Create multi-transform optimizer using Equinox-compatible approach with length-1 list wrapping.
    
    Args:
        model: The model to optimize
        model_name: Name of the model architecture  
        base_lr: Base learning rate for main parameters
        ssm_lr_factor: Learning rate factor for SSM parameters
        weight_decay: Weight decay for main parameters
        num_steps: Total training steps
        use_warmup_cosine: Whether to use warmup + cosine schedule

    Returns:
        Configured optax multi-transform optimizer
    """
    # Create schedules and clipped optimizers
    if use_warmup_cosine:
        main_schedule = create_warmup_cosine_schedule(base_lr, num_steps)
        ssm_schedule = create_warmup_cosine_schedule(base_lr * ssm_lr_factor, num_steps)
    else:
        main_schedule = base_lr
        ssm_schedule = base_lr * ssm_lr_factor
    
    # Create label function for parameter grouping
    label_fn = create_ssm_label_fn(model_name)
    
    # Create optimizers for each parameter group
    main_optimizer = optax.adamw(learning_rate=main_schedule, weight_decay=weight_decay)
    ssm_optimizer = optax.adamw(learning_rate=ssm_schedule, weight_decay=0.0)

    optimizers = {
        'ssm': ssm_optimizer,
        'main': main_optimizer
    }
    
    # Create multi-transform optimizer
    optimizer = optax.multi_transform(optimizers, label_fn)
    
    return optimizer, main_schedule, ssm_schedule

def create_optimizer(model_name, lr, ssm_lr_factor, weight_decay, num_steps, use_warmup_cosine, lr_scheduler, verbose=False):
    """
    Create optimizer with optional differential learning rates for SSM parameters.
    
    Args:
        model_name: Name of the model architecture
        lr: Base learning rate
        ssm_lr_factor: Learning rate factor for SSM parameters
        weight_decay: Weight decay coefficient
        num_steps: Total training steps
        use_warmup_cosine: Whether to use warmup + cosine schedule
        lr_scheduler: Learning rate scheduler function
        verbose: Whether to print optimizer information
    
    Returns:
        Configured optimizer
    """
    # Create optimizer with differential learning rates for SSM parameters
    if ssm_lr_factor != 1.0:
        # Use multi-optimizer approach for differential learning rates
        opt, main_schedule, ssm_schedule = create_multi_optimizer_with_equinox(model_name, lr, ssm_lr_factor, weight_decay, num_steps, use_warmup_cosine)
        if verbose:
            print(f"Using multi-optimizer: main_lr={lr}, ssm_lr={lr * ssm_lr_factor}, weight_decay={weight_decay}")
            print(f"SSM parameters will use lr_factor={ssm_lr_factor} (no weight decay)")
    else:
        # Use single optimizer (original approach)
        if use_warmup_cosine:
            schedule = create_warmup_cosine_schedule(lr, num_steps)
            if verbose:
                print(f"Using AdamW with warmup + cosine: peak_lr={lr}, warmup_steps={int(num_steps * 0.1)}")
                print(f"Weight_decay={weight_decay}")
            opt = optax.adamw(learning_rate=schedule, weight_decay=weight_decay)
        else:
            schedule = lr_scheduler(lr)
            if weight_decay > 0:
                if verbose:
                    print(f"Using AdamW with base_lr={lr} and weight_decay={weight_decay}")
                opt = optax.adamw(learning_rate=schedule, weight_decay=weight_decay)
            else:   
                if verbose:
                    print(f"Using Adam and base_lr={lr}")
                opt = optax.adam(learning_rate=schedule)

        main_schedule = schedule
        ssm_schedule = schedule  # Same schedule for SSM in single-optimizer case
    
    return opt, main_schedule, ssm_schedule

def truncate_optimizer_state(old_state, new_state):
    def truncate(old, new):
        # If both arrays and shapes match exactly → copy
        if isinstance(old, jax.Array) and isinstance(new, jax.Array):
            if old.shape == new.shape:
                return old
            # Try to truncate along all dimensions
            elif len(old.shape) == len(new.shape):
                # Check if all dimensions can be truncated (old >= new for each dim)
                can_truncate = all(old_dim >= new_dim for old_dim, new_dim in zip(old.shape, new.shape))
                if can_truncate:
                    # Build slicing tuple for all dimensions
                    slices = tuple(slice(None, new_dim) if new_dim < old_dim else slice(None) 
                                 for old_dim, new_dim in zip(old.shape, new.shape))
                    return old[slices]
                else:
                    raise ValueError(f"Cannot map {old.shape} → {new.shape}: some dimensions would need to expand")
            else:
                raise ValueError(f"Cannot map {old.shape} → {new.shape}: different number of dimensions")
        # For non-arrays (ints, floats, None, etc.), just use new
        return new
    return jax.tree_util.tree_map(truncate, old_state, new_state)


@eqx.filter_jit
def calc_output(model, X, state, key, stateful, nondeterministic, dual=False):
    """
    Compute model output with optional dual sequence handling.
    
    Args:
        model: The model to evaluate
        X: Input data 
        state: Model state (if stateful)
        key: JAX random key
        stateful: Whether model is stateful
        nondeterministic: Whether model is nondeterministic  
        dual: Whether to apply dual sequence processing
        
    Returns:
        output: Model predictions
        state: Updated model state
    """
    # Regular model processing (model handles dual internally)
    if stateful:
        if nondeterministic:
            output, state = jax.vmap(
                model, axis_name="batch", in_axes=(0, None, None), out_axes=(0, None)
            )(X, state, key)
        else:
            output, state = jax.vmap(
                model, axis_name="batch", in_axes=(0, None), out_axes=(0, None)
            )(X, state)
    elif nondeterministic:
        output = jax.vmap(model, in_axes=(0, None))(X, key)
    else:
        output = jax.vmap(model)(X)

    # Apply dual processing if needed (model outputs features when dual_head present)
    if dual and hasattr(model, 'dual_head') and model.dual_head is not None:
        from models.dual_processing import process_dual_outputs
        output = process_dual_outputs(output, model.dual_head)

    return output, state


@eqx.filter_jit
@eqx.filter_value_and_grad(has_aux=True)
def classification_loss(diff_model, static_model, X, y, state, key, dual=False):
    model = eqx.combine(diff_model, static_model)
    pred_y, state = calc_output(
        model, X, state, key, model.stateful, model.nondeterministic, dual
    )
    norm = 0
    if model.lip2:
        for layer in model.vf.mlp.layers:
            norm += jnp.mean(
                jnp.linalg.norm(layer.weight, axis=-1)
                + jnp.linalg.norm(layer.bias, axis=-1)
            )
        norm *= model.lambd
    return (
        jnp.mean(-jnp.sum(y * jnp.log(pred_y + 1e-8), axis=1)) + norm,
        state,
    )


@eqx.filter_jit
@eqx.filter_value_and_grad(has_aux=True)
def regression_loss(diff_model, static_model, X, y, state, key, dual=False):
    model = eqx.combine(diff_model, static_model)
    pred_y, state = calc_output(
        model, X, state, key, model.stateful, model.nondeterministic, dual
    )
    pred_y = pred_y[:, :, 0]
    norm = 0
    if model.lip2:
        for layer in model.vf.mlp.layers:
            norm += jnp.mean(
                jnp.linalg.norm(layer.weight, axis=-1)
                + jnp.linalg.norm(layer.bias, axis=-1)
            )
        norm *= model.lambd
    return (
        jnp.mean(jnp.mean((pred_y - y) ** 2, axis=1)) + norm,
        state,
    )


import jax
import jax.numpy as jnp
import equinox as eqx
# Assuming 'calc_output' is defined elsewhere and returns PROBABILITIES

@eqx.filter_jit
@eqx.filter_value_and_grad(has_aux=True)
def distillation_loss(
    diff_student,
    static_student,
    teacher_model,
    teacher_state,
    X,
    y,
    state,
    key,
    temperature=2.0,
    alpha=0.5,
    dual=False,
):
    """
    Computes distillation loss combining hard labels and soft teacher predictions.
    
    Since the model outputs probabilities (post-softmax), we need to work backwards
    to get logits for proper temperature scaling in distillation.
    
    Note: teacher_state is separate from student state since they have different dimensions.
    """
    student_model = eqx.combine(diff_student, static_student)
    
    # Get student probabilities from forward pass
    student_probs, state = calc_output(
        student_model, X, state, key, student_model.stateful, student_model.nondeterministic, dual
    )
    
    # Get teacher probabilities (no gradient, using teacher's own state)
    teacher_probs, _ = calc_output(
        teacher_model, X, teacher_state, key, teacher_model.stateful, teacher_model.nondeterministic, dual
    )
    
    # Clip for numerical stability
    epsilon = 1e-8
    student_probs = jnp.clip(student_probs, epsilon, 1.0 - epsilon)
    teacher_probs = jnp.clip(teacher_probs, epsilon, 1.0 - epsilon)
    
    # Hard loss: standard cross-entropy with true labels
    hard_loss = jnp.mean(-jnp.sum(y * jnp.log(student_probs), axis=1))
    
    # Soft loss: We need to work with logits for temperature scaling
    # Since model outputs softmax(logits), we recover logits via inverse:
    # If p = softmax(z), then z = log(p) + C (where C cancels in softmax)
    # For numerical stability with temperature, we use log-probabilities
    student_log_probs = jnp.log(student_probs)
    teacher_log_probs = jnp.log(teacher_probs)
    
    # Temperature-scaled softmax from log-probabilities
    # softmax(log(p) / T) = softmax((z + C) / T) where z are the original logits
    # This is approximately correct for temperature scaling
    soft_student = jax.nn.softmax(student_log_probs / temperature, axis=1)
    soft_teacher = jax.nn.softmax(teacher_log_probs / temperature, axis=1)
    
    # KL divergence: -sum(p_teacher * log(p_student))
    soft_loss = jnp.mean(-jnp.sum(soft_teacher * jnp.log(soft_student + epsilon), axis=1))
    
    # Scale by T^2 (standard in distillation)
    soft_loss = soft_loss * (temperature ** 2)
    
    # Combined loss
    loss = alpha * soft_loss + (1 - alpha) * hard_loss
    
    return loss, state


@eqx.filter_jit
def make_step(model, filter_spec, X, y, loss_fn, state, opt, opt_state, key, use_multi_optimizer=False, dual=False):
    diff_model, static_model = eqx.partition(model, filter_spec)
    (value, state), grads = loss_fn(diff_model, static_model, X, y, state, key, dual)
    params = eqx.filter(model, eqx.is_inexact_array)
    
    if use_multi_optimizer:
        # For multi-optimizer, wrap parameters and gradients in length-1 lists
        updates, opt_state = opt.update([grads], opt_state, [params])
        model = eqx.apply_updates(model, updates[0])  # Extract from length-1 list
    else:
        # For single optimizer, use parameters directly
        updates, opt_state = opt.update(grads, opt_state, params)
        model = eqx.apply_updates(model, updates)
    
    return model, state, opt_state, value
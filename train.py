"""
This module defines functions for creating datasets, building models, and training them using JAX
and Equinox. The main function, `create_dataset_model_and_train`, is designed to initialise the
dataset, construct the model, and execute the training process.

The function `create_dataset_model_and_train` takes the following arguments:

- `seed`: A random seed for reproducibility.
- `data_dir`: The directory where the dataset is stored.
- `use_presplit`: A boolean indicating whether to use a pre-split dataset.
- `dataset_name`: The name of the dataset to load and use for training.
- `output_step`: For regression tasks, the number of steps to skip before outputting a prediction.
- `metric`: The metric to use for evaluation. Supported values are `'mse'` for regression and `'accuracy'` for
            classification.
- `include_time`: A boolean indicating whether to include time as a channel in the time series data.
- `T`: The maximum time value to scale time data to [0, T].
- `model_name`: The name of the model architecture to use.
- `stepsize`: The size of the intervals for the Log-ODE method.
- `logsig_depth`: The depth of the Log-ODE method. Currently implemented for depths 1 and 2.
- `model_args`: A dictionary of additional arguments to customise the model.
- `num_steps`: The number of steps to train the model.
- `print_steps`: How often to print the loss during training.
- `lr`: The learning rate for the optimiser.
- `lr_scheduler`: The learning rate scheduler function (ignored if use_warmup_cosine=True).
- `batch_size`: The number of samples per batch during training.
- `output_parent_dir`: The parent directory where the training outputs will be saved.
- `weight_decay`: Weight decay coefficient for AdamW optimizer (default: 0.01).
- `use_warmup_cosine`: Whether to use warmup + cosine annealing schedule (default: False).
- `ssm_lr_factor`: Learning rate factor for SSM parameters when using multi-transform optimizer (default: 1.0).
- `energy_tol`: Tolerance for Hankel energy conservation (determines if we attempt to reduce and by how much)
- `red_warmup_steps`: Number of steps to warm up before attempting reduction (default: 0, starts at first print step)

The module also includes the following key functions:

- `create_warmup_cosine_schedule`: Creates warmup + cosine annealing schedule as specified in optimization guidelines.
- `calc_output`: Computes the model output, handling stateful and nondeterministic models with JAX's `vmap` for
                 batching.
- `classification_loss`: Computes the loss for classification tasks, including optional regularisation.
- `regression_loss`: Computes the loss for regression tasks, including optional regularisation.
- `make_step`: Performs a single optimisation step, updating model parameters based on the computed gradients.
- `train_model`: Handles the training loop, managing metrics, early stopping, and saving progress at regular intervals.
"""

import os
import shutil
import time
import json
import numpy as np

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from data_dir.datasets import create_dataset
from models.generate_model import create_model
from tqdm import tqdm


def create_warmup_cosine_schedule(peak_lr, num_steps, current_step=0, warmup_ratio=0.1, final_lr=1e-7):
    """
    Creates warmup + cosine annealing schedule starting from a specific step.
    
    Args:
        peak_lr: Peak learning rate to reach after warmup
        num_steps: Total number of training steps
        current_step: Step to start from (default 0 for original behavior)
        warmup_ratio: Fraction of training for warmup (default 0.05 = 5%)
        final_lr: Final learning rate after cosine decay (default 1e-7)
    
    Returns:
        Optax schedule function that continues the original schedule from current_step
    """
    warmup_steps = int(num_steps * warmup_ratio)
    
    # Get the original schedule function
    original_schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-7,
        peak_value=peak_lr,
        warmup_steps=warmup_steps,
        decay_steps=num_steps - warmup_steps,
        end_value=final_lr
    )
    
    # Calculate remaining steps
    remaining_steps = num_steps - current_step
    
    if remaining_steps <= 0:
        # If we're at or past the end, return constant final learning rate
        return lambda step: final_lr
    
    def adjusted_schedule(step):
        # Map the new step range [0, remaining_steps] to original range [current_step, num_steps]
        original_step = step + current_step
        return original_schedule(original_step)
    
    return adjusted_schedule


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


def create_multi_optimizer_with_equinox(model_name, base_lr, ssm_lr_factor, weight_decay, num_steps, use_warmup_cosine, current_step=0):
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
        main_schedule = create_warmup_cosine_schedule(base_lr, num_steps, current_step=current_step)
        ssm_schedule = create_warmup_cosine_schedule(base_lr * ssm_lr_factor, num_steps, current_step=current_step)
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
    
    return optimizer




@eqx.filter_jit
def calc_output(model, X, state, key, stateful, nondeterministic):
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

    return output, state


@eqx.filter_jit
@eqx.filter_value_and_grad(has_aux=True)
def classification_loss(diff_model, static_model, X, y, state, key):
    model = eqx.combine(diff_model, static_model)
    pred_y, state = calc_output(
        model, X, state, key, model.stateful, model.nondeterministic
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
def regression_loss(diff_model, static_model, X, y, state, key):
    model = eqx.combine(diff_model, static_model)
    pred_y, state = calc_output(
        model, X, state, key, model.stateful, model.nondeterministic
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


@eqx.filter_jit
def make_step(model, filter_spec, X, y, loss_fn, state, opt, opt_state, key, use_multi_optimizer=False):
    diff_model, static_model = eqx.partition(model, filter_spec)
    (value, state), grads = loss_fn(diff_model, static_model, X, y, state, key)
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


def train_model(
    dataset_name,
    model,
    metric,
    filter_spec,
    state,
    dataloaders,
    num_steps,
    print_steps,
    lr,
    lr_scheduler,
    batch_size,
    key,
    output_dir,
    id,
    data=None,
    weight_decay=0.01,
    use_warmup_cosine=False,
    ssm_lr_factor=1.0,
    model_name='lru',
    energy_tol=None,
    red_warmup_steps=0
):

    if metric == "accuracy":
        best_val = max
        operator_improv = lambda x, y: x >= y
        operator_no_improv = lambda x, y: x <= y
    elif metric == "mse":
        best_val = min
        operator_improv = lambda x, y: x <= y
        operator_no_improv = lambda x, y: x >= y
    else:
        raise ValueError(f"Unknown metric: {metric}")

    if os.path.isdir(output_dir):
        user_input = input(
            f"Warning: Output directory {output_dir} already exists. Do you want to delete it? (yes/no): "
        )
        if user_input.lower() == "yes":
            shutil.rmtree(output_dir)
            os.makedirs(output_dir)
            print(f"Directory {output_dir} has been deleted and recreated.")
        else:
            raise ValueError(f"Directory {output_dir} already exists. Exiting.")
    else:
        os.makedirs(output_dir)
        print(f"Directory {output_dir} has been created.")

    # Save data as a json file in the output directory
    if data is not None:
        with open(output_dir + "/data.json", "w") as f:
            json.dump(data, f)

    batchkey, key = jr.split(key, 2)

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
            opt = create_multi_optimizer_with_equinox(model_name, lr, ssm_lr_factor, weight_decay, num_steps, use_warmup_cosine
            )
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
        
        return opt

    opt = create_optimizer(model_name, lr, ssm_lr_factor, weight_decay, num_steps, use_warmup_cosine, lr_scheduler, verbose=True)

    # Initialize optimizer state with proper parameter wrapping
    model_params = eqx.filter(model, eqx.is_inexact_array)
    if ssm_lr_factor != 1.0:
        # For multi-optimizer, wrap parameters in length-1 list
        opt_state = opt.init([model_params])
    else:
        # For single optimizer, use parameters directly
        opt_state = opt.init(model_params)

    if model.classification:
        loss_fn = classification_loss
    else:
        loss_fn = regression_loss

    running_loss = 0.0
    if metric == "accuracy":
        all_val_metric = [0.0]
        all_train_metric = [0.0]
        val_metric_for_best_model = [0.0]
    elif metric == "mse":
        all_val_metric = [100.0]
        all_train_metric = [100.0]
        val_metric_for_best_model = [100.0]
    no_val_improvement = 0
    all_time = []
    test_metric = 0.0
    has_crashed = False
    has_stagnated = False
    start = time.time()

    # Add tracking for dimension reductions
    reduction_history = []
    ssm_dimensions = {}
    hankel_singular_values = {}

    for blc, block in enumerate(model.blocks):
        if hasattr(block, 'lru'):
            ssm_dimensions[blc] = block.get_dimension()

    with tqdm(total=num_steps, desc="Training", ncols=140) as pbar:
        for step, data in zip(
            range(num_steps),
            dataloaders["train"].loop(batch_size, key=batchkey),
        ):
            stepkey, key = jr.split(key, 2)
            X, y = data
            model, state, opt_state, value = make_step(
                model, filter_spec, X, y, loss_fn, state, opt, opt_state, stepkey, use_multi_optimizer=(ssm_lr_factor != 1.0)
            )
            running_loss += value
            
            if (step + 1) % print_steps == 0:
                # Compute training metric
                predictions = []
                labels = []
                for data in dataloaders["train"].loop_epoch(batch_size):
                    stepkey, key = jr.split(key, 2)
                    inference_model = eqx.tree_inference(model, value=True)
                    X, y = data
                    prediction, _ = calc_output(
                        inference_model,
                        X,
                        state,
                        stepkey,
                        model.stateful,
                        model.nondeterministic,
                    )
                    predictions.append(prediction)
                    labels.append(y)
                prediction = jnp.vstack(predictions)
                y = jnp.vstack(labels)
                if model.classification:
                    train_metric = jnp.mean(
                        jnp.argmax(prediction, axis=1) == jnp.argmax(y, axis=1)
                    )
                else:
                    prediction = prediction[:, :, 0]
                    train_metric = jnp.mean(jnp.mean((prediction - y) ** 2, axis=1), axis=0)
                # if the training metric is less than 0.15 kill training run
                if train_metric < 0.15:
                    has_crashed = True
                    break

                # Compute validation metric
                predictions = []
                labels = []
                for data in dataloaders["val"].loop_epoch(batch_size):
                    stepkey, key = jr.split(key, 2)
                    inference_model = eqx.tree_inference(model, value=True)
                    X, y = data
                    prediction, _ = calc_output(
                        inference_model,
                        X,
                        state,
                        stepkey,
                        model.stateful,
                        model.nondeterministic,
                    )
                    predictions.append(prediction)
                    labels.append(y)
                prediction = jnp.vstack(predictions)
                y = jnp.vstack(labels)
                if model.classification:
                    val_metric = jnp.mean(
                        jnp.argmax(prediction, axis=1) == jnp.argmax(y, axis=1)
                    )
                else:
                    prediction = prediction[:, :, 0]
                    val_metric = jnp.mean(jnp.mean((prediction - y) ** 2, axis=1), axis=0)
                end = time.time()
                total_time = end - start


                ### Dimensionality reduction ###
                dico = model.get_all_hankel_singular_values()
                hankel_singular_values[step] = dico

                reduction = False

                if energy_tol is not None and step > red_warmup_steps:
                    # Get reduction analysis for all blocks
                    reduction_analysis = model.get_reduction_analysis(dico, hankel_tol=energy_tol)

                    # Extract (1-tol)% threshold ranks for each block
                    ranks = []
                    for i, block in enumerate(model.blocks):
                        block_analysis = reduction_analysis[f'block_{i}']
                        rank = block_analysis['recommended_ranks']['threshold']
                        current_rank = block.get_dimension()
                    
                        # Only reduce if there's meaningful compression
                        if rank < current_rank * 0.95:  # At least 5% reduction
                            reduction = True

                            ranks.append(rank)

                            # Record the reduction
                            reduction_info = {
                                'step': step,
                                'block': i,
                                'old_dim': current_rank,
                                'new_dim': rank,
                                'error_bound': None,  # Add this if available from reduce_discrete_LTI
                            }
                            reduction_history.append(reduction_info)
                            # Update dimensions tracker
                            ssm_dimensions[i] = rank

                        else:
                            ranks.append(current_rank)  # No reduction
                    
                    if reduction:
                        # Apply reduction
                        model = model.reduce_model_balanced_truncation(ranks, dico, method="sqrtm")
                        # Reinitialize optimizer
                        opt = create_optimizer(model_name, lr, ssm_lr_factor, weight_decay, num_steps, use_warmup_cosine, lr_scheduler)
                        # Reinitialize optimizer state
                        opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

                start = time.time()
                if operator_no_improv(val_metric, best_val(val_metric_for_best_model)):
                    no_val_improvement += 1
                    if no_val_improvement > 10:
                        has_stagnated = True
                        break

                # overwrite the test metric if the validation improves or if the model has been reduced
                # this ensures the best model is restrained to the latest reduced model size
                if operator_improv(val_metric, best_val(val_metric_for_best_model)) or reduction:
                    val_metric_for_best_model.append(val_metric)
                    predictions = []
                    labels = []
                    for data in dataloaders["test"].loop_epoch(batch_size):
                        stepkey, key = jr.split(key, 2)
                        inference_model = eqx.tree_inference(model, value=True)
                        X, y = data
                        prediction, _ = calc_output(
                            inference_model,
                            X,
                            state,
                            stepkey,
                            model.stateful,
                            model.nondeterministic,
                        )
                        predictions.append(prediction)
                        labels.append(y)
                    prediction = jnp.vstack(predictions)
                    y = jnp.vstack(labels)
                    if model.classification:
                        test_metric = jnp.mean(
                            jnp.argmax(prediction, axis=1) == jnp.argmax(y, axis=1)
                        )
                    else:
                        prediction = prediction[:, :, 0]
                        test_metric = jnp.mean(
                            jnp.mean((prediction - y) ** 2, axis=1), axis=0
                        )
                    best_step = step + 1
                    # Reset no validation improvement counter either to show improvement or to give reduced model time to converge
                    no_val_improvement = 0
                        

                # Update progress bar with metrics
                pbar.set_postfix({
                    'Loss': f'{running_loss:.3f}',
                    'Train': f'{train_metric:.3f}',
                    'Val': f'{val_metric:.3f}',
                    'Top test': f'{test_metric:.3f} at step: {best_step}',
                    'Avg dim': f'{int(sum(ranks)/len(ranks))}' if 'ranks' in locals() else f'{ssm_dimensions[0]}'
                })

                
                # Performance metrics
                all_train_metric.append(train_metric)
                all_val_metric.append(val_metric)
                all_time.append(total_time)
                steps = jnp.arange(0, step + 1, print_steps)
                all_train_metric_save = jnp.array(all_train_metric)
                all_val_metric_save = jnp.array(all_val_metric)
                all_time_save = jnp.array(all_time)
                test_metric_save = jnp.array(test_metric)
                jnp.save(output_dir + "/steps.npy", steps)
                jnp.save(output_dir + "/all_train_metric.npy", all_train_metric_save)
                jnp.save(output_dir + "/all_val_metric.npy", all_val_metric_save)
                jnp.save(output_dir + "/all_time.npy", all_time_save)
                jnp.save(output_dir + "/test_metric.npy", test_metric_save)

                # Dimensionality reduction metrics
                np.save(output_dir + "/all_hankel_singular_values.npy", hankel_singular_values)
                np.save(output_dir + "/reduction_history.npy", np.array(reduction_history, dtype=object))
                np.save(output_dir + "/ssm_dimensions.npy", np.array(list(ssm_dimensions.items()), dtype=object))
                running_loss = 0.0
            
            pbar.update(1)

    if has_crashed:
        print(f"Unstable training has been aborted at step {step + 1}.")
    elif has_stagnated:
        print(f"Training has stagnated and stopped early at step {step + 1}.")
    else:
        print(f"Training completed successfully with best test metric: {test_metric:.3f} at step {best_step}.")

    return model


def create_dataset_model_and_train(
    seed,
    data_dir,
    use_presplit,
    dataset_name,
    output_step,
    metric,
    include_time,
    T,
    model_name,
    stepsize,
    logsig_depth,
    linoss_discretization,
    model_args,
    num_steps,
    print_steps,
    lr,
    lr_scheduler,
    batch_size,
    output_parent_dir="",
    id=None,
    data=None,
    weight_decay=0.01,
    use_warmup_cosine=False,
    ssm_lr_factor=1.0,
    energy_tol=None,
    red_warmup_steps=0
):
    if model_name == 'LinOSS':
        model_name_directory = model_name+'_'+linoss_discretization
    else:
        model_name_directory = model_name
    output_parent_dir += "outputs/" + model_name_directory + "/" + dataset_name
    output_dir = f"T_{T:.2f}_time_{include_time}_nsteps_{num_steps}_lr_{lr}"
    if model_name == "log_ncde" or model_name == "nrde":
        output_dir += f"_stepsize_{stepsize:.2f}_depth_{logsig_depth}"
    for k, v in model_args.items():
        name = str(v)
        if "(" in name:
            name = name.split("(", 1)[0]
        if name == "dt0":
            output_dir += f"_{k}_" + f"{v:.2f}"
        else:
            output_dir += f"_{k}_" + name
        if name == "PIDController":
            output_dir += f"_rtol_{v.rtol}_atol_{v.atol}"
    output_dir += f"_seed_{seed}"
    if energy_tol is not None:
        output_dir += f"_tol_{energy_tol:.3e}"
    if red_warmup_steps > 0:
        output_dir += f"_warmup_{red_warmup_steps}"

    key = jr.PRNGKey(seed)

    datasetkey, modelkey, trainkey, key = jr.split(key, 4)
    print(f"Creating dataset {dataset_name}")

    dataset = create_dataset(
        data_dir,
        dataset_name,
        stepsize=stepsize,
        depth=logsig_depth,
        include_time=include_time,
        T=T,
        use_idxs=False,
        use_presplit=use_presplit,
        key=datasetkey,
    )

    print(f"Creating model {model_name}")
    classification = metric == "accuracy"
    model, state = create_model(
        model_name,
        dataset.data_dim,
        dataset.logsig_dim,
        logsig_depth,
        dataset.intervals,
        dataset.label_dim,
        classification=classification,
        output_step=output_step,
        linoss_discretization=linoss_discretization,
        **model_args,
        key=modelkey,
    )
    filter_spec = jax.tree_util.tree_map(lambda _: True, model)
    if model_name == "nrde" or model_name == "log_ncde":
        dataloaders = dataset.path_dataloaders
        if model_name == "log_ncde":
            where = lambda model: (model.intervals, model.pairs)
            filter_spec = eqx.tree_at(
                where, filter_spec, replace=(False, False), is_leaf=lambda x: x is None
            )
        elif model_name == "nrde":
            where = lambda model: (model.intervals,)
            filter_spec = eqx.tree_at(where, filter_spec, replace=(False,))
    elif model_name == "ncde":
        dataloaders = dataset.coeff_dataloaders
    else:
        dataloaders = dataset.raw_dataloaders

    return train_model(
        dataset_name,
        model,
        metric,
        filter_spec,
        state,
        dataloaders,
        num_steps,
        print_steps,
        lr,
        lr_scheduler,
        batch_size,
        trainkey,
        output_parent_dir + "/" + output_dir,
        id,
        data=data,
        weight_decay=weight_decay,
        use_warmup_cosine=use_warmup_cosine,
        ssm_lr_factor=ssm_lr_factor,
        energy_tol=energy_tol,
        red_warmup_steps=red_warmup_steps,
        model_name=model_name
    )

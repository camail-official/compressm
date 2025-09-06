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
- `tol`: Tolerance for Hankel energy conservation (determines if we attempt to reduce and by how much)
- `red_steps`: Number of initial steps to do reduction before just training (default: 1e10, i.e. reduce during all training)

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
import time
import json
import numpy as np

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from data_dir.datasets import create_dataset
from models.generate_model import create_model
from training.train_utils import create_optimizer, truncate_optimizer_state, classification_loss, regression_loss, calc_output, make_step

from tqdm import tqdm

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
    tol=None,
    red_steps=1e10,
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
        print(f"Directory {output_dir} already exists. Skipping this run.")
        return None
    else:
        os.makedirs(output_dir)
        print(f"Directory {output_dir} has been created.")

    # Save data as a json file in the output directory
    if data is not None:
        with open(output_dir + "/data.json", "w") as f:
            json.dump(data, f)

    batchkey, key = jr.split(key, 2)

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

                if tol is not None and step < red_steps + 1:
                    # Get reduction analysis for all blocks
                    reduction_analysis = model.get_reduction_analysis(dico, hankel_tol=tol)

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
                        model_params = eqx.filter(model, eqx.is_inexact_array)
                        # Reinitialize optimizer
                        opt = create_optimizer(model_name, lr, ssm_lr_factor, weight_decay, num_steps, use_warmup_cosine, lr_scheduler)
                        # For multi-optimizer, wrap parameters in length-1 list, For single optimizer, use parameters directly
                        new_opt_state = opt.init([model_params]) if ssm_lr_factor != 1.0 else opt.init(model_params)
                        # Truncate/copy old state into new for learning optimization continuity
                        opt_state = truncate_optimizer_state(opt_state, new_opt_state)

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
                    'Loss': f'{running_loss:.2f}',
                    'Train': f'{train_metric:.2f}',
                    'Val': f'{val_metric:.2f}',
                    'Top test': f'{test_metric:.2f} at step: {best_step}',
                    'Avg dim': f'{int(sum(ranks)/len(ranks))}' if 'ranks' in locals() else f'{ssm_dimensions[0]}',
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
        print(f"Training completed successfully with best test metric: {test_metric:.2f} at step {best_step}.")

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
    tol=None,
    red_steps=0
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
    if tol is not None:
        output_dir += f"_tol_{tol:.3e}"
    if red_steps > 0:
        output_dir += f"_warmup_{red_steps}"

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
        tol=tol,
        red_steps=red_steps,
        model_name=model_name
    )

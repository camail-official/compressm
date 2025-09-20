"""
This module defines the `Dataset` class and functions for generating datasets tailored to different model types.
A `Dataset` object in this module contains three different dataloaders, each providing a specific version of the data
required by different models:

- `raw_dataloaders`: Returns the raw time series data, suitable for recurrent neural networks (RNNs) and structured
  state space models (SSMs).
- `coeff_dataloaders`: Provides the coefficients of an interpolation of the data, used by Neural Controlled Differential
  Equations (NCDEs).
- `path_dataloaders`: Provides the log-signature of the data over intervals, used by Neural Rough Differential Equations
  (NRDEs) and Log-NCDEs.

The module also includes utility functions for processing and generating these datasets, ensuring compatibility with
different model requirements.
"""

import os
import pickle
from dataclasses import dataclass
from typing import Dict
from types import SimpleNamespace

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from data_dir.dataloaders import Dataloader
from data_dir.generate_coeffs import calc_coeffs
from data_dir.generate_paths import calc_paths
from data_dir.lra.lra import IMDB, AAN, ListOps, PathFinder


@dataclass
class Dataset:
    name: str
    raw_dataloaders: Dict[str, Dataloader]
    coeff_dataloaders: Dict[str, Dataloader] | None
    path_dataloaders: Dict[str, Dataloader] | None
    data_dim: int
    logsig_dim: int | None
    intervals: jnp.ndarray | None
    label_dim: int


def batch_calc_paths(data, stepsize, depth, inmemory=False):
    N = len(data)
    batchsize = 128
    num_batches = N // batchsize
    remainder = N % batchsize
    path_data = []
    if inmemory:
        out_func = lambda x: x
        in_func = lambda x: x
    else:
        out_func = lambda x: np.array(x)
        in_func = lambda x: jnp.array(x)
    for i in range(num_batches):
        path_data.append(
            out_func(
                calc_paths(
                    in_func(data[i * batchsize : (i + 1) * batchsize]), stepsize, depth
                )
            )
        )
    if remainder > 0:
        path_data.append(
            out_func(calc_paths(in_func(data[-remainder:]), stepsize, depth))
        )
    if inmemory:
        path_data = jnp.concatenate(path_data)
    else:
        path_data = np.concatenate(path_data)
    return path_data


def batch_calc_coeffs(data, include_time, T, inmemory=False):
    N = len(data)
    batchsize = 128
    num_batches = N // batchsize
    remainder = N % batchsize
    coeffs = []
    if inmemory:
        out_func = lambda x: x
        in_func = lambda x: x
    else:
        out_func = lambda x: np.array(x)
        in_func = lambda x: jnp.array(x)
    for i in range(num_batches):
        coeffs.append(
            out_func(
                calc_coeffs(
                    in_func(data[i * batchsize : (i + 1) * batchsize]), include_time, T
                )
            )
        )
    if remainder > 0:
        coeffs.append(
            out_func(calc_coeffs(in_func(data[-remainder:]), include_time, T))
        )
    if inmemory:
        coeffs = jnp.concatenate(coeffs)
    else:
        coeffs = np.concatenate(coeffs)
    return coeffs


def dataset_generator(
    name,
    data,
    labels,
    stepsize,
    depth,
    include_time,
    T,
    inmemory=False,
    idxs=None,
    use_presplit=False,
    *,
    key,
):
    N = len(data)
    if idxs is None:
        if use_presplit:
            train_data, val_data, test_data = data
            train_labels, val_labels, test_labels = labels
        else:
            permkey, key = jr.split(key)
            bound1 = int(N * 0.7)
            bound2 = int(N * 0.85)
            idxs_new = jr.permutation(permkey, N)
            train_data, train_labels = (
                data[idxs_new[:bound1]],
                labels[idxs_new[:bound1]],
            )
            val_data, val_labels = (
                data[idxs_new[bound1:bound2]],
                labels[idxs_new[bound1:bound2]],
            )
            test_data, test_labels = data[idxs_new[bound2:]], labels[idxs_new[bound2:]]
    else:
        train_data, train_labels = data[idxs[0]], labels[idxs[0]]
        val_data, val_labels = data[idxs[1]], labels[idxs[1]]
        test_data, test_labels = None, None

    intervals = jnp.arange(0, train_data.shape[1], stepsize)
    intervals = jnp.concatenate((intervals, jnp.array([train_data.shape[1]])))
    intervals = intervals * (T / train_data.shape[1])

    train_paths = batch_calc_paths(train_data, stepsize, depth)
    val_paths = batch_calc_paths(val_data, stepsize, depth)
    test_paths = batch_calc_paths(test_data, stepsize, depth)
    logsig_dim = train_paths.shape[-1]

    train_coeffs = calc_coeffs(train_data, include_time, T)
    val_coeffs = calc_coeffs(val_data, include_time, T)
    test_coeffs = calc_coeffs(test_data, include_time, T)
    train_coeff_data = (
        (T / train_data.shape[1])
        * jnp.repeat(
            jnp.arange(train_data.shape[1])[None, :], train_data.shape[0], axis=0
        ),
        train_coeffs,
        train_data[:, 0, :],
    )
    val_coeff_data = (
        (T / val_data.shape[1])
        * jnp.repeat(jnp.arange(val_data.shape[1])[None, :], val_data.shape[0], axis=0),
        val_coeffs,
        val_data[:, 0, :],
    )
    if idxs is None:
        test_coeff_data = (
            (T / test_data.shape[1])
            * jnp.repeat(
                jnp.arange(test_data.shape[1])[None, :], test_data.shape[0], axis=0
            ),
            test_coeffs,
            test_data[:, 0, :],
        )

    train_path_data = (
        (T / train_data.shape[1])
        * jnp.repeat(
            jnp.arange(train_data.shape[1])[None, :], train_data.shape[0], axis=0
        ),
        train_paths,
        train_data[:, 0, :],
    )
    val_path_data = (
        (T / val_data.shape[1])
        * jnp.repeat(jnp.arange(val_data.shape[1])[None, :], val_data.shape[0], axis=0),
        val_paths,
        val_data[:, 0, :],
    )
    if idxs is None:
        test_path_data = (
            (T / test_data.shape[1])
            * jnp.repeat(
                jnp.arange(test_data.shape[1])[None, :], test_data.shape[0], axis=0
            ),
            test_paths,
            test_data[:, 0, :],
        )

    data_dim = train_data.shape[-1]
    if len(train_labels.shape) == 1 or name == "ppg":
        label_dim = 1
    else:
        label_dim = train_labels.shape[-1]
    raw_dataloaders = {
        "train": Dataloader(train_data, train_labels, inmemory),
        "val": Dataloader(val_data, val_labels, inmemory),
        "test": Dataloader(test_data, test_labels, inmemory),
    }

    coeff_dataloaders = {
        "train": Dataloader(train_coeff_data, train_labels, inmemory),
        "val": Dataloader(val_coeff_data, val_labels, inmemory),
        "test": Dataloader(test_coeff_data, test_labels, inmemory),
    }
    path_dataloaders = {
        "train": Dataloader(train_path_data, train_labels, inmemory),
        "val": Dataloader(val_path_data, val_labels, inmemory),
        "test": Dataloader(test_path_data, test_labels, inmemory),
    }
    return Dataset(
        name,
        raw_dataloaders,
        coeff_dataloaders,
        path_dataloaders,
        data_dim,
        logsig_dim,
        intervals,
        label_dim,
    )


def create_scifar_dataset(
    data_dir, use_presplit, stepsize, depth, include_time, T, *, key
):
    import torchvision
    import torchvision.transforms as transforms

    # Transform: ToTensor + Normalize + Reshape to (1024, 3)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # (32,32,3) -> (3,32,32) and /255
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Lambda(lambda x: x.view(3, 1024).t()),  # (3,32,32) -> (1024,3)
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    # Apply transforms by iterating through dataset
    x_train = []
    y_train = []
    for i in range(len(trainset)):
        img, label = trainset[i]  # This applies the transform!
        x_train.append(img.numpy())
        y_train.append(label)

    x_test = []
    y_test = []
    for i in range(len(testset)):
        img, label = testset[i]  # This applies the transform!
        x_test.append(img.numpy())
        y_test.append(label)

    # Convert lists to arrays
    x_train = np.array(x_train)  # Shape: (50000, 1024, 3)
    y_train = np.array(y_train)
    x_test = np.array(x_test)  # Shape: (10000, 1024, 3)
    y_test = np.array(y_test)

    # Convert to JAX arrays
    x_train = jnp.array(x_train)
    x_test = jnp.array(x_test)
    y_train = jnp.array(y_train)
    y_test = jnp.array(y_test)

    # Create one-hot labels
    train_onehot = jnp.zeros((len(y_train), 10))
    train_onehot = train_onehot.at[jnp.arange(len(y_train)), y_train].set(1)
    test_onehot = jnp.zeros((len(y_test), 10))
    test_onehot = test_onehot.at[jnp.arange(len(y_test)), y_test].set(1)

    if use_presplit:
        # Use presplit train/test, split train into train/val
        split_key, key = jr.split(key)
        n_train = len(x_train)
        val_size = int(0.1 * n_train)  # 10% for validation
        train_size = n_train - val_size

        idxs = jr.permutation(split_key, n_train)
        train_idxs = idxs[:train_size]
        val_idxs = idxs[train_size:]

        train_data = x_train[train_idxs]
        val_data = x_train[val_idxs]
        test_data = x_test

        train_labels = train_onehot[train_idxs]
        val_labels = train_onehot[val_idxs]
        test_labels = test_onehot

        data = (train_data, val_data, test_data)
        labels = (train_labels, val_labels, test_labels)
    else:
        # Combine train and test data for random split
        data = jnp.concatenate((x_train, x_test), axis=0)
        labels = jnp.concatenate((train_onehot, test_onehot), axis=0)

    if include_time:
        if use_presplit:
            ts_train = (T / train_data.shape[1]) * jnp.repeat(
                jnp.arange(train_data.shape[1])[None, :], train_data.shape[0], axis=0
            )
            train_data = jnp.concatenate([ts_train[:, :, None], train_data], axis=2)

            ts_val = (T / val_data.shape[1]) * jnp.repeat(
                jnp.arange(val_data.shape[1])[None, :], val_data.shape[0], axis=0
            )
            val_data = jnp.concatenate([ts_val[:, :, None], val_data], axis=2)

            ts_test = (T / test_data.shape[1]) * jnp.repeat(
                jnp.arange(test_data.shape[1])[None, :], test_data.shape[0], axis=0
            )
            test_data = jnp.concatenate([ts_test[:, :, None], test_data], axis=2)

            data = (train_data, val_data, test_data)
        else:
            ts = (T / data.shape[1]) * jnp.repeat(
                jnp.arange(data.shape[1])[None, :], data.shape[0], axis=0
            )
            data = jnp.concatenate([ts[:, :, None], data], axis=2)

    return dataset_generator(
        "scifar",
        data,
        labels,
        stepsize,
        depth,
        include_time,
        T,
        inmemory=False,
        use_presplit=use_presplit,
        key=key,
    )


def create_mnist_dataset(
    data_dir, use_presplit, stepsize, depth, include_time, T, *, key
):
    import torchvision
    import torchvision.transforms as transforms

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x.view(1, 784).t())]
    )
    transform_train = transform_test = transform

    trainset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform_train
    )

    testset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform_test
    )

    # Apply transforms by iterating through dataset
    x_train = []
    y_train = []
    for i in range(len(trainset)):
        img, label = trainset[i]  # This applies the transform!
        x_train.append(img.numpy())
        y_train.append(label)

    x_test = []
    y_test = []
    for i in range(len(testset)):
        img, label = testset[i]  # This applies the transform!
        x_test.append(img.numpy())
        y_test.append(label)

    # Convert lists to arrays
    x_train = np.array(x_train)  # Shape: (60000, 784, 1)
    y_train = np.array(y_train)
    x_test = np.array(x_test)  # Shape: (10000, 784, 1)
    y_test = np.array(y_test)

    # Convert to JAX arrays
    x_train = jnp.array(x_train)
    x_test = jnp.array(x_test)
    y_train = jnp.array(y_train)
    y_test = jnp.array(y_test)

    # Create one-hot labels
    train_onehot = jnp.zeros((len(y_train), 10))
    train_onehot = train_onehot.at[jnp.arange(len(y_train)), y_train].set(1)
    test_onehot = jnp.zeros((len(y_test), 10))
    test_onehot = test_onehot.at[jnp.arange(len(y_test)), y_test].set(1)

    if use_presplit:
        # Use presplit train/test, split train into train/val
        split_key, key = jr.split(key)
        n_train = len(x_train)
        val_size = int(0.10 * n_train)  # 10% for validation
        train_size = n_train - val_size

        idxs = jr.permutation(split_key, n_train)
        train_idxs = idxs[:train_size]
        val_idxs = idxs[train_size:]

        train_data = x_train[train_idxs]
        val_data = x_train[val_idxs]
        test_data = x_test

        train_labels = train_onehot[train_idxs]
        val_labels = train_onehot[val_idxs]
        test_labels = test_onehot

        data = (train_data, val_data, test_data)
        labels = (train_labels, val_labels, test_labels)
    else:
        # Combine train and test data for random split
        data = jnp.concatenate((x_train, x_test), axis=0)
        labels = jnp.concatenate((train_onehot, test_onehot), axis=0)

    if include_time:
        if use_presplit:
            ts_train = (T / train_data.shape[1]) * jnp.repeat(
                jnp.arange(train_data.shape[1])[None, :], train_data.shape[0], axis=0
            )
            train_data = jnp.concatenate([ts_train[:, :, None], train_data], axis=2)

            ts_val = (T / val_data.shape[1]) * jnp.repeat(
                jnp.arange(val_data.shape[1])[None, :], val_data.shape[0], axis=0
            )
            val_data = jnp.concatenate([ts_val[:, :, None], val_data], axis=2)

            ts_test = (T / test_data.shape[1]) * jnp.repeat(
                jnp.arange(test_data.shape[1])[None, :], test_data.shape[0], axis=0
            )
            test_data = jnp.concatenate([ts_test[:, :, None], test_data], axis=2)

            data = (train_data, val_data, test_data)
        else:
            ts = (T / data.shape[1]) * jnp.repeat(
                jnp.arange(data.shape[1])[None, :], data.shape[0], axis=0
            )
            data = jnp.concatenate([ts[:, :, None], data], axis=2)

    return dataset_generator(
        "mnist",
        data,
        labels,
        stepsize,
        depth,
        include_time,
        T,
        inmemory=False,
        use_presplit=use_presplit,
        key=key,
    )


def create_imdb_dataset(data_dir, batch_size):

    dataset = IMDB(data_dir=data_dir, _name_="imdb", val_split=0.1)
    dataset.setup()

    # TODO: allow for setting the batch size in the config later on
    raw_dataloaders = {
        "train": dataset.train_dataloader(batch_size=batch_size, shuffle=True),
        "val": dataset.val_dataloader(batch_size=batch_size, shuffle=False),
        "test": dataset.test_dataloader(batch_size=batch_size, shuffle=False),
    }

    return_dict = {
        "name": "imdb",
        "raw_dataloaders": raw_dataloaders,
        "coeff_dataloaders": None,
        "path_dataloaders": None,
        "data_dim": 1,
        "label_dim": 2,
        "logsig_dim": None,
        "intervals": None,
        "vocab_size": dataset.n_tokens,
    }

    return SimpleNamespace(**return_dict)


def create_aan_dataset(data_dir, batch_size):

    dataset = AAN(data_dir=data_dir, _name_="aan", val_split=0.1)
    dataset.setup()

    # TODO: allow for setting the batch size in the config later on
    raw_dataloaders = {
        "train": dataset.train_dataloader(batch_size=batch_size, shuffle=True),
        "val": dataset.val_dataloader(batch_size=batch_size, shuffle=False),
        "test": dataset.test_dataloader(batch_size=batch_size, shuffle=False),
    }

    return_dict = {
        "name": "imdb",
        "raw_dataloaders": raw_dataloaders,
        "coeff_dataloaders": None,
        "path_dataloaders": None,
        "data_dim": 1,
        "label_dim": 2,
        "logsig_dim": None,
        "intervals": None,
        "vocab_size": dataset.n_tokens,
    }

    return SimpleNamespace(**return_dict)


def create_listops_dataset(data_dir, batch_size):

    dataset = ListOps(data_dir=data_dir, _name_="listops", val_split=0.1)
    dataset.setup()

    # TODO: allow for setting the batch size in the config later on
    raw_dataloaders = {
        "train": dataset.train_dataloader(batch_size=batch_size, shuffle=True),
        "val": dataset.val_dataloader(batch_size=batch_size, shuffle=False),
        "test": dataset.test_dataloader(batch_size=batch_size, shuffle=False),
    }

    return_dict = {
        "name": "listops",
        "raw_dataloaders": raw_dataloaders,
        "coeff_dataloaders": None,
        "path_dataloaders": None,
        "data_dim": 1,
        "label_dim": 10,
        "logsig_dim": None,
        "intervals": None,
        "vocab_size": dataset.n_tokens,
    }

    return SimpleNamespace(**return_dict)


def create_pathfinder_dataset(data_dir, batch_size):

    dataset = PathFinder(
        data_dir=data_dir, _name_="pathfinder", val_split=0.1, resolution=32
    )
    dataset.setup()

    # TODO: allow for setting the batch size in the config later on
    raw_dataloaders = {
        "train": dataset.train_dataloader(batch_size=batch_size, shuffle=True),
        "val": dataset.val_dataloader(batch_size=batch_size, shuffle=False)[None],
        "test": dataset.test_dataloader(batch_size=batch_size, shuffle=False)[None],
    }

    return_dict = {
        "name": "pathfinder",
        "raw_dataloaders": raw_dataloaders,
        "coeff_dataloaders": None,
        "path_dataloaders": None,
        "data_dim": 1,
        "label_dim": 2,
        "logsig_dim": None,
        "intervals": None,
        "vocab_size": dataset.n_tokens,
    }

    return SimpleNamespace(**return_dict)


def create_pathfinderx_dataset(data_dir, batch_size):

    dataset = PathFinder(
        data_dir=data_dir, _name_="pathfinder", val_split=0.1, resolution=128
    )
    dataset.setup()

    # TODO: allow for setting the batch size in the config later on
    raw_dataloaders = {
        "train": dataset.train_dataloader(batch_size=batch_size, shuffle=True),
        "val": dataset.val_dataloader(batch_size=batch_size, shuffle=False)[None],
        "test": dataset.test_dataloader(batch_size=batch_size, shuffle=False)[None],
    }

    return_dict = {
        "name": "pathfinderx",
        "raw_dataloaders": raw_dataloaders,
        "coeff_dataloaders": None,
        "path_dataloaders": None,
        "data_dim": 1,
        "label_dim": 2,
        "logsig_dim": None,
        "intervals": None,
        "vocab_size": dataset.n_tokens,
    }

    return SimpleNamespace(**return_dict)


# keeping this as reference example although not used
def create_toy_dataset(data_dir, name, stepsize, depth, include_time, T, *, key):
    with open(data_dir + "/processed/toy/signature/data.pkl", "rb") as f:
        data = pickle.load(f)
    with open(data_dir + "/processed/toy/signature/labels.pkl", "rb") as f:
        labels = pickle.load(f)
    if name == "signature1":
        labels = ((jnp.sign(labels[0][:, 2]) + 1) / 2).astype(int)
    elif name == "signature2":
        labels = ((jnp.sign(labels[1][:, 2, 5]) + 1) / 2).astype(int)
    elif name == "signature3":
        labels = ((jnp.sign(labels[2][:, 2, 5, 0]) + 1) / 2).astype(int)
    elif name == "signature4":
        labels = ((jnp.sign(labels[3][:, 2, 5, 0, 3]) + 1) / 2).astype(int)
    onehot_labels = jnp.zeros((len(labels), len(jnp.unique(labels))))
    onehot_labels = onehot_labels.at[jnp.arange(len(labels)), labels].set(1)
    idxs = None

    if include_time:
        ts = (T / data.shape[1]) * jnp.repeat(
            jnp.arange(data.shape[1])[None, :], data.shape[0], axis=0
        )
        data = jnp.concatenate([ts[:, :, None], data], axis=2)

    return dataset_generator(
        "toy", data, onehot_labels, stepsize, depth, include_time, T, idxs, key=key
    )


def create_dataset(
    data_dir,
    name,
    use_idxs,
    use_presplit,
    stepsize,
    batch_size,
    depth,
    include_time,
    T,
    *,
    key,
):

    if name == "scifar":
        return create_scifar_dataset(
            data_dir, use_presplit, stepsize, depth, include_time, T, key=key
        )
    elif name == "mnist":
        return create_mnist_dataset(
            data_dir, use_presplit, stepsize, depth, include_time, T, key=key
        )
    elif name == "pathfinder":
        return create_pathfinder_dataset(data_dir, batch_size)
    elif name == "pathfinderx":
        return create_pathfinderx_dataset(data_dir, batch_size)
    elif name == "imdb":
        return create_imdb_dataset(data_dir, batch_size)
    elif name == "aan":
        return create_aan_dataset(data_dir, batch_size)
    elif name == "listops":
        return create_listops_dataset(data_dir, batch_size)
    else:
        raise ValueError(f"Dataset {name} not a valid dataset")

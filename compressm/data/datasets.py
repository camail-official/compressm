"""
Dataset loading for sMNIST and sCIFAR benchmarks.

This module provides functions to create sequential image classification
datasets used for evaluating sequence models.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from compressm.data.dataloaders import Dataloader


@dataclass
class Dataset:
    """
    Container for a dataset with train/val/test splits.
    
    Attributes:
        name: Dataset identifier
        dataloaders: Dict mapping split names to Dataloader objects
        input_dim: Dimension of input features at each timestep
        output_dim: Dimension of output (number of classes)
        seq_len: Length of sequences
    """
    name: str
    dataloaders: Dict[str, Dataloader]
    input_dim: int
    output_dim: int
    seq_len: int


def create_smnist(*, key: jr.PRNGKey, data_dir: str = "./data") -> Dataset:
    """
    Create Sequential MNIST dataset.
    
    MNIST images (28x28) are flattened to sequences of length 784,
    with each timestep being a single pixel value.
    
    Args:
        key: JAX random key for train/val split
        data_dir: Directory to download/cache data
        
    Returns:
        Dataset object with train/val/test dataloaders
    """
    import torchvision
    import torchvision.transforms as transforms

    # Transform: flatten to (784, 1) sequence
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(1, 784).t())  # (1, 28, 28) -> (784, 1)
    ])

    # Load datasets
    trainset = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )
    
    # Extract data
    x_train = np.array([trainset[i][0].numpy() for i in range(len(trainset))])
    y_train = np.array([trainset[i][1] for i in range(len(trainset))])
    x_test = np.array([testset[i][0].numpy() for i in range(len(testset))])
    y_test = np.array([testset[i][1] for i in range(len(testset))])

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

    # Split train into train/val (90/10)
    split_key, key = jr.split(key)
    n_train = len(x_train)
    val_size = int(0.10 * n_train)
    
    idxs = jr.permutation(split_key, n_train)
    train_idxs = idxs[val_size:]
    val_idxs = idxs[:val_size]
    
    # Create dataloaders
    dataloaders = {
        "train": Dataloader(np.array(x_train[train_idxs]), np.array(train_onehot[train_idxs])),
        "val": Dataloader(np.array(x_train[val_idxs]), np.array(train_onehot[val_idxs])),
        "test": Dataloader(np.array(x_test), np.array(test_onehot)),
    }

    return Dataset(
        name="smnist",
        dataloaders=dataloaders,
        input_dim=1,
        output_dim=10,
        seq_len=784,
    )


def create_scifar(*, key: jr.PRNGKey, data_dir: str = "./data") -> Dataset:
    """
    Create Sequential CIFAR-10 dataset.
    
    CIFAR-10 images (32x32x3) are flattened to sequences of length 1024,
    with each timestep being 3 RGB values.
    
    Args:
        key: JAX random key for train/val split
        data_dir: Directory to download/cache data
        
    Returns:
        Dataset object with train/val/test dataloaders
    """
    import torchvision
    import torchvision.transforms as transforms

    # Transform: normalize and flatten to (1024, 3) sequence
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Lambda(lambda x: x.view(3, 1024).t())  # (3, 32, 32) -> (1024, 3)
    ])

    # Load datasets
    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )

    # Extract data
    x_train = np.array([trainset[i][0].numpy() for i in range(len(trainset))])
    y_train = np.array([trainset[i][1] for i in range(len(trainset))])
    x_test = np.array([testset[i][0].numpy() for i in range(len(testset))])
    y_test = np.array([testset[i][1] for i in range(len(testset))])

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

    # Split train into train/val (90/10)
    split_key, key = jr.split(key)
    n_train = len(x_train)
    val_size = int(0.10 * n_train)
    
    idxs = jr.permutation(split_key, n_train)
    train_idxs = idxs[val_size:]
    val_idxs = idxs[:val_size]
    
    # Create dataloaders
    dataloaders = {
        "train": Dataloader(np.array(x_train[train_idxs]), np.array(train_onehot[train_idxs])),
        "val": Dataloader(np.array(x_train[val_idxs]), np.array(train_onehot[val_idxs])),
        "test": Dataloader(np.array(x_test), np.array(test_onehot)),
    }

    return Dataset(
        name="scifar",
        dataloaders=dataloaders,
        input_dim=3,
        output_dim=10,
        seq_len=1024,
    )


def create_dataset(name: str, *, key: jr.PRNGKey, data_dir: str = "./data") -> Dataset:
    """
    Create a dataset by name.
    
    Args:
        name: Dataset name ("smnist" or "scifar")
        key: JAX random key
        data_dir: Directory for data storage
        
    Returns:
        Dataset object
        
    Raises:
        ValueError: If dataset name is not recognized
    """
    if name == "smnist":
        return create_smnist(key=key, data_dir=data_dir)
    elif name == "scifar":
        return create_scifar(key=key, data_dir=data_dir)
    else:
        raise ValueError(f"Unknown dataset: {name}. Supported: 'smnist', 'scifar'")

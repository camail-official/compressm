"""
Dataset loading for sMNIST, sCIFAR, and LRA benchmarks.

This module provides functions to create sequential image and text classification
datasets used for evaluating sequence models.
"""

from dataclasses import dataclass
from typing import Dict
import os

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import torchvision
from torchvision import transforms
from datasets import load_dataset, DatasetDict
import torchtext
from torchtext import vocab as tf_vocab
from PIL import Image
import glob
import pickle
from concurrent.futures import ThreadPoolExecutor

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
    vocab_size: Optional[int] = None
    use_embedding: bool = False


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

def create_imdb(*, key: jr.PRNGKey, data_dir: str = "./data") -> Dataset:
    """
    Create IMDB dataset (LRA version).
    """
    # Configuration
    l_max = 4096
    level = "char"
    min_freq = 15
    append_bos = False
    append_eos = True
    val_split = 0.0 # Use test as val by default if 0.0

    print(f"IMDB {level} level | min_freq {min_freq}")
    print(f"Loading IMDB dataset from: {data_dir}")
    
    cache_path = os.path.join(data_dir, f"imdb_{level}_processed")
    if os.path.exists(cache_path):
        print(f"Loading processed IMDB from cache: {cache_path}")
        dataset = DatasetDict.load_from_disk(cache_path)
        with open(os.path.join(cache_path, "vocab.pkl"), "rb") as f:
            vocab = pickle.load(f)
    else:
        # Load dataset
        dataset = load_dataset("imdb", cache_dir=data_dir)
        dataset = DatasetDict(train=dataset["train"], test=dataset["test"])
        
        # Tokenizer
        if level == "word":
            tokenizer = torchtext.data.utils.get_tokenizer("spacy", language="en_core_web_sm")
        else:
            tokenizer = list 
            
        l_max_tokens = l_max - int(append_bos) - int(append_eos)
        
        def tokenize(example):
            return {"tokens": tokenizer(example["text"])[:l_max_tokens]}
            
        dataset = dataset.map(
            tokenize,
            remove_columns=["text"],
            keep_in_memory=True,
            load_from_cache_file=False,
            num_proc=4,
        )
        
        # Build vocab
        vocab = tf_vocab.build_vocab_from_iterator(
            dataset["train"]["tokens"],
            min_freq=min_freq,
            specials=(
                ["<pad>", "<unk>"]
                + (["<bos>"] if append_bos else [])
                + (["<eos>"] if append_eos else [])
            ),
        )
        vocab.set_default_index(vocab["<unk>"])
        
        numericalize = lambda example: {
            "input_ids": vocab(
                (["<bos>"] if append_bos else [])
                + example["tokens"]
                + (["<eos>"] if append_eos else [])
            )
        }

        dataset = dataset.map(
            numericalize,
            remove_columns=["tokens"],
            keep_in_memory=True,
            load_from_cache_file=False,
            num_proc=4,
        )
        
        # Save to cache
        print(f"Saving processed IMDB to cache: {cache_path}")
        dataset.save_to_disk(cache_path)
        with open(os.path.join(cache_path, "vocab.pkl"), "wb") as f:
            pickle.dump(vocab, f)
    
    # Extract data using numpy format for speed
    dataset.set_format(type="numpy", columns=["input_ids", "label"])
    x_train = dataset["train"]["input_ids"]
    y_train = dataset["train"]["label"]
    x_test = dataset["test"]["input_ids"]
    y_test = dataset["test"]["label"]
    
    # Pad sequences to fixed length
    def pad_to_length(sequences, length, pad_value):
        padded = np.zeros((len(sequences), length), dtype=np.int64)
        for i, seq in enumerate(sequences):
            seq_len = min(len(seq), length)
            padded[i, :seq_len] = seq[:seq_len]
        return padded
    
    x_train = pad_to_length(x_train, l_max, vocab["<pad>"])
    x_test = pad_to_length(x_test, l_max, vocab["<pad>"])
    
    # Convert to JAX arrays
    x_train = jnp.array(x_train)[..., None] 
    x_test = jnp.array(x_test)[..., None]
    y_train = jnp.array(y_train).astype(int)
    y_test = jnp.array(y_test).astype(int)
    
    # One-hot labels
    train_onehot = jnp.zeros((len(y_train), 2))
    train_onehot = train_onehot.at[jnp.arange(len(y_train)), y_train].set(1)
    test_onehot = jnp.zeros((len(y_test), 2))
    test_onehot = test_onehot.at[jnp.arange(len(y_test)), y_test].set(1)

    # Split
    if val_split == 0.0:
        # Use test set as val set
        dataloaders = {
            "train": Dataloader(x_train, train_onehot, inmemory=True),
            "val": Dataloader(x_test, test_onehot, inmemory=True), # Use test as val
            "test": Dataloader(x_test, test_onehot, inmemory=True),
        }
    else:
        split_key, key = jr.split(key)
        n_train = len(x_train)
        val_size = int(val_split * n_train)
        idxs = jr.permutation(split_key, n_train)
        train_idxs = idxs[val_size:]
        val_idxs = idxs[:val_size]
        
        dataloaders = {
            "train": Dataloader(x_train[train_idxs], train_onehot[train_idxs], inmemory=True),
            "val": Dataloader(x_train[val_idxs], train_onehot[val_idxs], inmemory=True),
            "test": Dataloader(x_test, test_onehot, inmemory=True),
        }

    return Dataset(
        name="imdb",
        dataloaders=dataloaders,
        input_dim=1,
        output_dim=2,
        seq_len=l_max,
        vocab_size=len(vocab),
        use_embedding=True
    )

def create_aan(*, key: jr.PRNGKey, data_dir: str = "./data") -> Dataset:
    """
    Create AAN dataset.
    """
    l_max = 4096
    append_bos = False
    append_eos = True
    
    data_path = os.path.join(data_dir, "aan")
    print(f"Loading AAN dataset from: {data_path}")
    if not os.path.exists(data_path):
         # This should probably be handled by user downloading LRA
         # But for now we assume it exists as per reference
         pass 

    cache_path = os.path.join(data_dir, "aan_processed")
    if os.path.exists(cache_path):
        print(f"Loading processed AAN from cache: {cache_path}")
        dataset = DatasetDict.load_from_disk(cache_path)
        with open(os.path.join(cache_path, "vocab.pkl"), "rb") as f:
            vocab = pickle.load(f)
    else:
        dataset = load_dataset(
            "csv",
            data_files={
                "train": os.path.join(data_path, "new_aan_pairs.train.tsv"),
                "val": os.path.join(data_path, "new_aan_pairs.eval.tsv"),
                "test": os.path.join(data_path, "new_aan_pairs.test.tsv"),
            },
            delimiter="\t",
            column_names=["label", "input1_id", "input2_id", "text1", "text2"],
            keep_in_memory=True,
        )
        dataset = dataset.remove_columns(["input1_id", "input2_id"])
        
        tokenizer = list
        l_max_tokens = l_max - int(append_bos) - int(append_eos)

        def tokenize(example):
            return {
                "tokens1": tokenizer(example["text1"])[:l_max_tokens],
                "tokens2": tokenizer(example["text2"])[:l_max_tokens],
            }
            
        dataset = dataset.map(
            tokenize,
            remove_columns=["text1", "text2"],
            keep_in_memory=True,
            load_from_cache_file=False,
            num_proc=4,
        )

        # Build vocab (reference uses direct concatenation)
        vocab = tf_vocab.build_vocab_from_iterator(
            dataset["train"]["tokens1"] + dataset["train"]["tokens2"],
            specials=(
                ["<pad>", "<unk>"]
                + (["<bos>"] if append_bos else [])
                + (["<eos>"] if append_eos else [])
            ),
        )
        vocab.set_default_index(vocab["<unk>"])

        encode = lambda text: vocab(
            (["<bos>"] if append_bos else []) + text + (["<eos>"] if append_eos else [])
        )
        numericalize = lambda example: {
            "input_ids1": encode(example["tokens1"]),
            "input_ids2": encode(example["tokens2"]),
        }

        dataset = dataset.map(
            numericalize,
            remove_columns=["tokens1", "tokens2"],
            keep_in_memory=True,
            load_from_cache_file=False,
            num_proc=4,
        )
        
        # Save to cache
        print(f"Saving processed AAN to cache: {cache_path}")
        dataset.save_to_disk(cache_path)
        with open(os.path.join(cache_path, "vocab.pkl"), "wb") as f:
            pickle.dump(vocab, f)
    # Extract and process AAN data (two sequences per example)
    dataset.set_format(type="numpy", columns=["input_ids1", "input_ids2", "label"])
    
    def pad_aan(ids1_batch, ids2_batch, l_max, pad_value):
        """Pad two sequences separately unlike the old concatenation approach."""
        batch_size = len(ids1_batch)
        # We store as (batch, 2, l_max) to keep sequences separate
        result = np.zeros((batch_size, 2, l_max), dtype=np.int64)
        
        for i in range(batch_size):
            ids1 = np.array(ids1_batch[i])
            ids2 = np.array(ids2_batch[i])
            
            # Pad or truncate ids1 to l_max
            if len(ids1) < l_max:
                ids1 = np.pad(ids1, (0, l_max - len(ids1)), constant_values=pad_value)
            else:
                ids1 = ids1[:l_max]
                
            # Pad or truncate ids2 to l_max
            if len(ids2) < l_max:
                ids2 = np.pad(ids2, (0, l_max - len(ids2)), constant_values=pad_value)
            else:
                ids2 = ids2[:l_max]
            
            result[i, 0] = ids1
            result[i, 1] = ids2
        
        return result
    
    # Process each split
    x_train = pad_aan(
        dataset["train"]["input_ids1"],
        dataset["train"]["input_ids2"],
        l_max,
        vocab["<pad>"]
    )
    y_train = dataset["train"]["label"].astype(int)
    
    x_val = pad_aan(
        dataset["val"]["input_ids1"],
        dataset["val"]["input_ids2"],
        l_max,
        vocab["<pad>"]
    )
    y_val = dataset["val"]["label"].astype(int)
    
    x_test = pad_aan(
        dataset["test"]["input_ids1"],
        dataset["test"]["input_ids2"],
        l_max,
        vocab["<pad>"]
    )
    y_test = dataset["test"]["label"].astype(int)
    
    # Convert to JAX arrays (stack them appropriately)
    # Shape: (batch, 2, l_max, 1)
    x_train = jnp.array(x_train)[..., None]
    x_val = jnp.array(x_val)[..., None]
    x_test = jnp.array(x_test)[..., None]
    y_train = jnp.array(y_train)
    y_val = jnp.array(y_val)
    y_test = jnp.array(y_test)
    
    def to_onehot(y):
        oh = jnp.zeros((len(y), 2))
        return oh.at[jnp.arange(len(y)), y].set(1)

    dataloaders = {
        "train": Dataloader(x_train, to_onehot(y_train), inmemory=True),
        "val": Dataloader(x_val, to_onehot(y_val), inmemory=True),
        "test": Dataloader(x_test, to_onehot(y_test), inmemory=True),
    }

    return Dataset(
        name="aan", 
        dataloaders=dataloaders, 
        input_dim=1, 
        output_dim=2, 
        seq_len=l_max,
        vocab_size=len(vocab),
        use_embedding=True
    )


def create_listops(*, key: jr.PRNGKey, data_dir: str = "./data") -> Dataset:
    """
    ListOps dataset.
    """
    l_max = 2048
    append_bos = False
    append_eos = True
    
    data_path = os.path.join(data_dir, "listops")
    print(f"Loading ListOps dataset from: {data_path}")
    if not os.path.exists(data_path):
         pass

    cache_path = os.path.join(data_dir, "listops_processed")
    if os.path.exists(cache_path):
        print(f"Loading processed ListOps from cache: {cache_path}")
        dataset = DatasetDict.load_from_disk(cache_path)
        with open(os.path.join(cache_path, "vocab.pkl"), "rb") as f:
            vocab = pickle.load(f)
    else:
        dataset = load_dataset(
            "csv",
            data_files={
                "train": os.path.join(data_path, "basic_train.tsv"),
                "val": os.path.join(data_path, "basic_val.tsv"),
                "test": os.path.join(data_path, "basic_test.tsv"),
            },
            delimiter="\t",
            keep_in_memory=True,
        )

        def listops_tokenizer(s):
            return s.translate({ord("]"): ord("X"), ord("("): None, ord(")"): None}).split()

        tokenizer = listops_tokenizer
        l_max_tokens = l_max - int(append_bos) - int(append_eos)
        
        def tokenize(example):
            return {"tokens": tokenizer(example["Source"])[:l_max_tokens]}

        dataset = dataset.map(
            tokenize,
            remove_columns=["Source"],
            keep_in_memory=True,
            load_from_cache_file=False,
            num_proc=4,
        )
        
        vocab = tf_vocab.build_vocab_from_iterator(
            (tokens for tokens in dataset["train"]["tokens"]),
            specials=(
                ["<pad>", "<unk>"]
                + (["<bos>"] if append_bos else [])
                + (["<eos>"] if append_eos else [])
            ),
        )
        vocab.set_default_index(vocab["<unk>"])
        
        numericalize = lambda example: {
            "input_ids": vocab(
                (["<bos>"] if append_bos else [])
                + example["tokens"]
                + (["<eos>"] if append_eos else [])
            )
        }

        dataset = dataset.map(
            numericalize,
            remove_columns=["tokens"],
            keep_in_memory=True,
            load_from_cache_file=False,
            num_proc=4,
        )
        
        # Save to cache
        print(f"Saving processed ListOps to cache: {cache_path}")
        dataset.save_to_disk(cache_path)
        with open(os.path.join(cache_path, "vocab.pkl"), "wb") as f:
            pickle.dump(vocab, f)
    
    dataset.set_format(type="numpy", columns=["input_ids", "Target"])
    
    # Pad sequences to fixed length
    def pad_to_length(sequences, length, pad_value):
        padded = np.zeros((len(sequences), length), dtype=np.int64)
        for i, seq in enumerate(sequences):
            seq_len = min(len(seq), length)
            padded[i, :seq_len] = seq[:seq_len]
        return padded
    
    x_train = pad_to_length(dataset["train"]["input_ids"], l_max, vocab["<pad>"])
    x_val = pad_to_length(dataset["val"]["input_ids"], l_max, vocab["<pad>"])
    x_test = pad_to_length(dataset["test"]["input_ids"], l_max, vocab["<pad>"])
    
    # Convert to JAX
    x_train = jnp.array(x_train)[..., None]
    y_train = jnp.array(dataset["train"]["Target"]).astype(int)
    x_val = jnp.array(x_val)[..., None]
    y_val = jnp.array(dataset["val"]["Target"]).astype(int)
    x_test = jnp.array(x_test)[..., None]
    y_test = jnp.array(dataset["test"]["Target"]).astype(int)
    
    def to_onehot(y):
        oh = jnp.zeros((len(y), 10))
        return oh.at[jnp.arange(len(y)), y].set(1)
        
    dataloaders = {
        "train": Dataloader(x_train, to_onehot(y_train), inmemory=True),
        "val": Dataloader(x_val, to_onehot(y_val), inmemory=True),
        "test": Dataloader(x_test, to_onehot(y_test), inmemory=True),
    }
    
    return Dataset(
        name="listops", 
        dataloaders=dataloaders, 
        input_dim=1, 
        output_dim=10, 
        seq_len=l_max,
        vocab_size=len(vocab),
        use_embedding=True
    )


class LazyPathfinderData:
    """
    Lazy loader for Pathfinder images to avoid loading all images into memory.
    """
    def __init__(self, samples, resolution, seq_len):
        self.samples = samples # List of (path, label)
        self.resolution = resolution
        self.seq_len = seq_len
        self.n = len(samples)
        self.executor = ThreadPoolExecutor(max_workers=8)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):
            return self._load_single(idx)
        elif isinstance(idx, (slice, np.ndarray, list, jnp.ndarray)):
            if isinstance(idx, slice):
                indices = range(*idx.indices(self.n))
            else:
                indices = idx
            
            # Load images for the batch in parallel
            batch_x = list(self.executor.map(self._load_single, indices))
            return np.array(batch_x)
        else:
            raise TypeError(f"Invalid index type: {type(idx)}")

    def _load_single(self, i):
        p, _ = self.samples[i]
        try:
            img = Image.open(p).convert("L")
            if img.size != (self.resolution, self.resolution):
                img = img.resize((self.resolution, self.resolution))
            # Normalize to [0, 1] as expected by most sequence models
            return (np.array(img).flatten()[..., None] / 255.0).astype(np.float32)
        except Exception as e:
            # Fallback for missing/broken images
            return np.zeros((self.seq_len, 1), dtype=np.float32)

    @property
    def shape(self):
        return (self.n, self.seq_len, 1)
    
    @property
    def dtype(self):
        return np.float32


def create_pathfinder(*, key: jr.PRNGKey, data_dir: str = "./data") -> Dataset:
    """
    Pathfinder dataset.
    """
    resolution = 32
    seq_len = resolution * resolution
    input_dim = 1
    output_dim = 2
    
    # Path resolution: try multiple common LRA structures
    options = [
        os.path.join(data_dir, f"pathfinder/pathfinder{resolution}"),
        os.path.join(data_dir, f"pathfinder{resolution}"),
        data_dir
    ]
    
    data_path = None
    for opt in options:
        if os.path.exists(os.path.join(opt, "curv_baseline")) or os.path.exists(os.path.join(opt, f"pathfinder{resolution}_preprocessed.npz")):
            data_path = opt
            break
            
    if data_path is None:
        data_path = options[0] 
        
    print(f"Loading Pathfinder dataset from: {data_path}")
    
    # Try to load from preprocessed .npz file first (fastest)
    npz_path = os.path.join(data_path, f"pathfinder{resolution}_preprocessed.npz")
    if os.path.exists(npz_path):
        print(f"Loading preprocessed Pathfinder data from {npz_path}...")
        data = np.load(npz_path, allow_pickle=True)
        images = data['images'] # (N, H, W)
        labels = data['labels']
        if len(images.shape) == 3:
            images = images.reshape(images.shape[0], -1) # Flatten
        
        # Still load .npz into memory as it's typically faster than lazy disk IO
        X = jnp.array(images[..., None])
        Y = jnp.array(labels)
    else:
        # Fallback to individual image loading (Lazy)
        samples = []
        base = os.path.join(data_path, "curv_baseline")
        if not os.path.exists(base):
            base = data_path
            
        metadata_dir = os.path.join(base, "metadata")
        print(f"Checking metadata directory: {metadata_dir}")
        
        if os.path.exists(metadata_dir):
            files = sorted(glob.glob(os.path.join(metadata_dir, "*.npy")) + glob.glob(os.path.join(metadata_dir, "*.txt")))
            
            print(f"Parsing {len(files)} metadata files...")
            for fpath in files:
                 try:
                     with open(fpath, "r") as f:
                         lines = f.read().splitlines()
                         for line in lines:
                             parts = line.split()
                             if len(parts) >= 4:
                                 img_rel = os.path.join(parts[0], parts[1])
                                 label = int(parts[3])
                                 img_full = os.path.join(base, img_rel)
                                 samples.append((img_full, label))
                 except:
                     pass
        
        if not samples:
            raise ValueError(f"No valid image samples found in {data_path}")
            
        print(f"Found {len(samples)} samples. Initializing lazy loading...")
        
        # Shuffle samples upfront so train/val/test splits are randomized
        split_key, key = jr.split(key)
        idxs = np.array(jr.permutation(split_key, len(samples)))
        samples = [samples[i] for i in idxs]
        
        n = len(samples)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        
        train_samples = samples[:n_train]
        val_samples = samples[n_train:n_train+n_val]
        test_samples = samples[n_train+n_val:]
        
        def samples_to_y(s):
            return jnp.array([l for _, l in s])
            
        def to_onehot(y):
            oh = jnp.zeros((len(y), output_dim))
            return oh.at[jnp.arange(len(y)), y.astype(int)].set(1)

        dataloaders = {
            "train": Dataloader(LazyPathfinderData(train_samples, resolution, seq_len), to_onehot(samples_to_y(train_samples))),
            "val": Dataloader(LazyPathfinderData(val_samples, resolution, seq_len), to_onehot(samples_to_y(val_samples))),
            "test": Dataloader(LazyPathfinderData(test_samples, resolution, seq_len), to_onehot(samples_to_y(test_samples))),
        }
        return Dataset(name="pathfinder", dataloaders=dataloaders, input_dim=input_dim, output_dim=output_dim, seq_len=seq_len)

    # If we loaded from .npz (Not Lazy)
    n = len(X)
    split_key, key = jr.split(key)
    idxs = jr.permutation(split_key, n)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    
    train_idxs = idxs[:n_train]
    val_idxs = idxs[n_train:n_train+n_val]
    test_idxs = idxs[n_train+n_val:]
    
    def to_onehot(y):
        oh = jnp.zeros((len(y), output_dim))
        return oh.at[jnp.arange(len(y)), y.astype(int)].set(1)

    dataloaders = {
        "train": Dataloader(X[train_idxs], to_onehot(Y[train_idxs])),
        "val": Dataloader(X[val_idxs], to_onehot(Y[val_idxs])),
        "test": Dataloader(X[test_idxs], to_onehot(Y[test_idxs])),
    }
    
    return Dataset(name="pathfinder", dataloaders=dataloaders, input_dim=input_dim, output_dim=output_dim, seq_len=seq_len)


def create_dataset(name: str, *, key: jr.PRNGKey, data_dir: str = "./data") -> Dataset:
    """
    Create a dataset by name.
    
    Args:
        name: Dataset name ("smnist", "scifar", "imdb", "aan", "listops", "pathfinder")
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
    elif name == "imdb":
        return create_imdb(key=key, data_dir=data_dir)
    elif name == "aan":
        return create_aan(key=key, data_dir=data_dir)
    elif name == "listops":
        return create_listops(key=key, data_dir=data_dir)
    elif name == "pathfinder":
        return create_pathfinder(key=key, data_dir=data_dir)
    else:
        raise ValueError(f"Unknown dataset: {name}. Supported: 'smnist', 'scifar', 'imdb', 'aan', 'listops', 'pathfinder'")

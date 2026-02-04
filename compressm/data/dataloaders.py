"""
Data loading utilities for batching and iterating over datasets.

This module provides a simple Dataloader class for efficiently
batching and shuffling sequence data for training.
"""

from typing import Tuple, Generator

import jax.numpy as jnp
import jax.random as jr
import numpy as np


class Dataloader:
    """
    Simple dataloader for sequence data.
    
    Supports:
    - Infinite shuffled iteration (for training)
    - Single epoch iteration (for evaluation)
    - Lazy conversion from NumPy to JAX arrays (to save GPU memory)
    
    Attributes:
        data: Input sequences, shape (n_samples, seq_len, n_features)
        labels: Target labels, shape (n_samples, n_classes) or (n_samples,)
        size: Number of samples
    """
    
    def __init__(self, data: np.ndarray, labels: np.ndarray, inmemory: bool = False):
        """
        Initialize dataloader.
        
        Args:
            data: Input data as numpy or JAX array
            labels: Labels as numpy or JAX array
            inmemory: If True, keep as JAX arrays. If False, store as numpy
                     and convert lazily to save GPU memory.
        """
        self.data = data
        self.labels = labels
        self.size = len(data) if data is not None else 0
        
        # Function to convert to JAX arrays (or identity if already in memory)
        if inmemory:
            self._to_jax = lambda x: x
        else:
            self._to_jax = lambda x: jnp.asarray(x)

    def loop(
        self,
        batch_size: int,
        *,
        key: jr.PRNGKey
    ) -> Generator[Tuple[jnp.ndarray, jnp.ndarray], None, None]:
        """
        Generate batches indefinitely with shuffling.
        
        This generator never terminates - it reshuffles and repeats
        after each pass through the data.
        
        Args:
            batch_size: Number of samples per batch
            key: JAX random key for shuffling
            
        Yields:
            Tuple of (batch_data, batch_labels)
        """
        if self.size == 0:
            raise ValueError("Dataloader is empty")
        
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        
        if batch_size > self.size:
            raise ValueError(f"batch_size ({batch_size}) larger than dataset size ({self.size})")
        
        indices = jnp.arange(self.size)
        
        while True:
            # Shuffle at start of each epoch
            key, subkey = jr.split(key)
            perm = jr.permutation(subkey, indices)
            
            # Yield batches
            for start in range(0, self.size - batch_size + 1, batch_size):
                batch_idx = perm[start:start + batch_size]
                yield self._to_jax(self.data[batch_idx]), self._to_jax(self.labels[batch_idx])

    def loop_epoch(
        self,
        batch_size: int
    ) -> Generator[Tuple[jnp.ndarray, jnp.ndarray], None, None]:
        """
        Generate batches for one epoch (no shuffling).
        
        Useful for evaluation where we want deterministic iteration
        through all data.
        
        Args:
            batch_size: Number of samples per batch
            
        Yields:
            Tuple of (batch_data, batch_labels)
        """
        if self.size == 0:
            raise ValueError("Dataloader is empty")
        
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        
        if batch_size > self.size:
            raise ValueError(f"batch_size ({batch_size}) larger than dataset size ({self.size})")
        
        # Full batches
        for start in range(0, self.size - batch_size + 1, batch_size):
            end = start + batch_size
            yield self._to_jax(self.data[start:end]), self._to_jax(self.labels[start:end])
        
        # Final partial batch (if any)
        remainder = self.size % batch_size
        if remainder > 0:
            yield self._to_jax(self.data[-remainder:]), self._to_jax(self.labels[-remainder:])

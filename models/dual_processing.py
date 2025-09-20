"""
Dual Sequence Processing Module

This module provides pure functions for handling dual sequence datasets without requiring
model wrappers. It processes outputs from regular models to handle dual sequence logic.
"""

import jax
import jax.numpy as jnp


def process_dual_outputs(outputs, dual_head):
    """
    Process outputs from dual sequence data using pure functions.

    Args:
        outputs: Model outputs with shape (2B, feature_dim) from doubled batch
        dual_head: DualHead module for processing

    Returns:
        Final predictions with shape (B, output_dim)
    """
    batch_size = outputs.shape[0] // 2

    # Split doubled batch back into two halves
    outputs1 = outputs[:batch_size]  # First B sequences: (B, feature_dim)
    outputs2 = outputs[batch_size:]  # Second B sequences: (B, feature_dim)

    # Concatenate along feature dimension
    concatenated = jnp.concatenate([outputs1, outputs2], axis=1)  # (B, 2*feature_dim)

    # Process through DualHead
    final_outputs = jax.vmap(dual_head)(concatenated)  # (B, output_dim)

    # Apply softmax for classification
    return jax.nn.softmax(final_outputs, axis=1)

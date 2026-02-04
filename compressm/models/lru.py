"""
Linear Recurrent Unit (LRU) model implementation.

This module implements the LRU architecture as described in the paper,
with integrated support for balanced truncation-based compression.

The LRU uses complex-valued diagonal state transitions, making it
efficient for sequence modeling while maintaining strong theoretical
properties for model reduction.
"""

from typing import List, Dict, Any, Optional, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from compressm.reduction.hsv import (
    SelectionMode,
    hankel_singular_values_diagonal,
    reduction_analysis,
)
from compressm.reduction.balanced_truncation import (
    reduce_discrete_lti,
    lti_to_lru,
)


def binary_operator_diag(element_i, element_j):
    """Associative operator for parallel scan with diagonal state matrix."""
    a_i, bu_i = element_i
    a_j, bu_j = element_j
    return a_j * a_i, a_j * bu_i + bu_j


class GLU(eqx.Module):
    """Gated Linear Unit for non-linear transformations."""
    
    w1: eqx.nn.Linear
    w2: eqx.nn.Linear

    def __init__(self, input_dim: int, output_dim: int, *, key: jr.PRNGKey):
        w1_key, w2_key = jr.split(key, 2)
        self.w1 = eqx.nn.Linear(input_dim, output_dim, use_bias=True, key=w1_key)
        self.w2 = eqx.nn.Linear(input_dim, output_dim, use_bias=True, key=w2_key)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.w1(x) * jax.nn.sigmoid(self.w2(x))


class LRULayer(eqx.Module):
    """
    Single LRU layer with complex-valued diagonal state transitions.
    
    The layer maintains state using the recurrence:
        x_{k+1} = Lambda * x_k + B * u_k
        y_k = Re(C * x_k) + D * u_k
    
    where Lambda = exp(-exp(nu_log) + i*exp(theta_log)) are complex eigenvalues.
    
    Attributes:
        nu_log: Log of negative log of eigenvalue magnitudes, shape (N,)
        theta_log: Log of eigenvalue phases, shape (N,)
        B_re, B_im: Real/imaginary parts of input projection, shape (N, H)
        C_re, C_im: Real/imaginary parts of output projection, shape (H, N)
        D: Skip connection weights, shape (H,)
        gamma_log: Log of normalization factors, shape (N,)
    """
    
    nu_log: jnp.ndarray
    theta_log: jnp.ndarray
    B_re: jnp.ndarray
    B_im: jnp.ndarray
    C_re: jnp.ndarray
    C_im: jnp.ndarray
    D: jnp.ndarray
    gamma_log: jnp.ndarray

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        r_min: float = 0.0,
        r_max: float = 1.0,
        max_phase: float = 6.28,
        *,
        key: jr.PRNGKey
    ):
        """
        Initialize LRU layer.
        
        Args:
            state_dim: Dimension of the recurrent state (N)
            hidden_dim: Dimension of input/output (H)
            r_min: Minimum eigenvalue magnitude
            r_max: Maximum eigenvalue magnitude
            max_phase: Maximum eigenvalue phase
            key: JAX random key
        """
        keys = jr.split(key, 7)
        u1_key, u2_key, B_re_key, B_im_key, C_re_key, C_im_key, D_key = keys

        # Initialize eigenvalues uniformly on ring [r_min, r_max] with phase [0, max_phase]
        u1 = jr.uniform(u1_key, shape=(state_dim,))
        u2 = jr.uniform(u2_key, shape=(state_dim,))
        self.nu_log = jnp.log(-0.5 * jnp.log(u1 * (r_max**2 - r_min**2) + r_min**2))
        self.theta_log = jnp.log(max_phase * u2)

        # Glorot initialization for projection matrices
        self.B_re = jr.normal(B_re_key, shape=(state_dim, hidden_dim)) / jnp.sqrt(2 * hidden_dim)
        self.B_im = jr.normal(B_im_key, shape=(state_dim, hidden_dim)) / jnp.sqrt(2 * hidden_dim)
        self.C_re = jr.normal(C_re_key, shape=(hidden_dim, state_dim)) / jnp.sqrt(state_dim)
        self.C_im = jr.normal(C_im_key, shape=(hidden_dim, state_dim)) / jnp.sqrt(state_dim)
        self.D = jr.normal(D_key, shape=(hidden_dim,))

        # Compute normalization factor
        diag_lambda = jnp.exp(-jnp.exp(self.nu_log) + 1j * jnp.exp(self.theta_log))
        self.gamma_log = jnp.log(jnp.sqrt(1 - jnp.abs(diag_lambda) ** 2))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass through the LRU layer.
        
        Args:
            x: Input sequence, shape (seq_len, hidden_dim)
            
        Returns:
            Output sequence, shape (seq_len, hidden_dim)
        """
        # Materialize eigenvalues and projections
        Lambda = jnp.exp(-jnp.exp(self.nu_log) + 1j * jnp.exp(self.theta_log))
        B_norm = (self.B_re + 1j * self.B_im) * jnp.expand_dims(
            jnp.exp(self.gamma_log), axis=-1
        )
        C = self.C_re + 1j * self.C_im
        
        # Parallel scan for efficient sequence processing
        Lambda_elements = jnp.repeat(Lambda[None, ...], x.shape[0], axis=0)
        Bu_elements = jax.vmap(lambda u: B_norm @ u)(x)
        elements = (Lambda_elements, Bu_elements)
        _, inner_states = jax.lax.associative_scan(binary_operator_diag, elements)
        
        # Output projection
        y = jax.vmap(lambda z, u: (C @ z).real + (self.D * u))(inner_states, x)
        return y

    def to_lti(self):
        """Convert LRU parameters to LTI system representation."""
        Lambda = jnp.exp(-jnp.exp(self.nu_log) + 1j * jnp.exp(self.theta_log))
        B = (self.B_re + 1j * self.B_im) * jnp.expand_dims(
            jnp.exp(self.gamma_log), axis=-1
        )
        C = self.C_re + 1j * self.C_im
        D = self.D
        return Lambda, B, C, D
    
    @property
    def state_dim(self) -> int:
        """Get the state dimension of this layer."""
        return self.nu_log.shape[0]
    
    def get_hankel_singular_values(self):
        """Compute Hankel singular values for this layer."""
        Lambda, B, C, D = self.to_lti()
        P, Q, g = hankel_singular_values_diagonal(Lambda, B, C)
        return P, Q, g
    
    def get_reduction_analysis(self, g: jnp.ndarray, hankel_tol: float):
        """Analyze reduction potential for this layer."""
        return reduction_analysis(g, hankel_tol)

    def reduce(
        self,
        rank: int,
        P,
        Q,
        method: str = "sqrtm",
        selection: SelectionMode = SelectionMode.LARGEST
    ) -> "LRULayer":
        """
        Apply balanced truncation to reduce state dimension.
        
        Args:
            rank: Target state dimension
            P: Controllability Gramian
            Q: Observability Gramian  
            method: "sqrtm" or "chol" for transformation
            selection: Which states to keep (LARGEST, SMALLEST, RANDOM)
            
        Returns:
            New LRULayer with reduced dimension
        """
        if rank >= self.state_dim:
            raise ValueError(f"Rank ({rank}) must be smaller than current dimension ({self.state_dim}).")

        # Get current LTI representation
        Lambda, B, C, D = self.to_lti()

        # Apply balanced truncation
        A_red, B_red, C_red = reduce_discrete_lti(
            Lambda, B, C, P, Q,
            rank=rank,
            method=method,
            selection=selection
        )

        # Convert back to LRU parameterization
        nu_log_new, theta_log_new, B_re_new, B_im_new, C_re_new, C_im_new, gamma_new = lti_to_lru(
            A_red, B_red, C_red
        )

        # Create new reduced layer using equinox tree_at
        return eqx.tree_at(
            lambda x: (
                x.nu_log,
                x.theta_log,
                x.B_re,
                x.B_im,
                x.C_re,
                x.C_im,
                x.gamma_log
            ),
            self,
            (
                jnp.real(nu_log_new),
                jnp.real(theta_log_new),
                jnp.real(B_re_new / gamma_new[:, None]),
                jnp.real(B_im_new / gamma_new[:, None]),
                jnp.real(C_re_new),
                jnp.real(C_im_new),
                jnp.real(jnp.log(gamma_new))
            )
        )


class LRUBlock(eqx.Module):
    """
    LRU block with normalization, GLU, and dropout.
    
    Each block consists of:
    1. Batch normalization
    2. LRU layer
    3. GELU activation
    4. GLU (Gated Linear Unit)
    5. Dropout
    6. Residual connection
    """
    
    norm: eqx.nn.BatchNorm
    lru: LRULayer
    glu: GLU
    drop: eqx.nn.Dropout

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        r_min: float = 0.0,
        r_max: float = 1.0,
        max_phase: float = 6.28,
        drop_rate: float = 0.1,
        *,
        key: jr.PRNGKey
    ):
        """
        Initialize LRU block.
        
        Args:
            state_dim: Dimension of the recurrent state
            hidden_dim: Dimension of input/output
            r_min: Minimum eigenvalue magnitude
            r_max: Maximum eigenvalue magnitude
            max_phase: Maximum eigenvalue phase
            drop_rate: Dropout probability
            key: JAX random key
        """
        lru_key, glu_key = jr.split(key, 2)
        self.norm = eqx.nn.BatchNorm(
            input_size=hidden_dim, axis_name="batch", channelwise_affine=False
        )
        self.lru = LRULayer(state_dim, hidden_dim, r_min, r_max, max_phase, key=lru_key)
        self.glu = GLU(hidden_dim, hidden_dim, key=glu_key)
        self.drop = eqx.nn.Dropout(p=drop_rate)

    def __call__(self, x: jnp.ndarray, state, *, key: jr.PRNGKey):
        """
        Forward pass through the block.
        
        Args:
            x: Input, shape (seq_len, hidden_dim)
            state: Batch norm state
            key: Random key for dropout
            
        Returns:
            Output with residual connection, updated state
        """
        drop_key1, drop_key2 = jr.split(key, 2)
        skip = x
        x, state = self.norm(x.T, state)
        x = x.T
        x = self.lru(x)
        x = self.drop(jax.nn.gelu(x), key=drop_key1)
        x = jax.vmap(self.glu)(x)
        x = self.drop(x, key=drop_key2)
        x = skip + x
        return x, state
    
    @property
    def state_dim(self) -> int:
        """Get the state dimension of this block's LRU layer."""
        return self.lru.state_dim
    
    def get_hankel_singular_values(self):
        """Get Hankel singular values for the LRU layer in this block."""
        return self.lru.get_hankel_singular_values()
    
    def get_reduction_analysis(self, g: jnp.ndarray, hankel_tol: float):
        """Get reduction analysis for this block."""
        return self.lru.get_reduction_analysis(g, hankel_tol=hankel_tol)
    
    def reduce(
        self,
        rank: int,
        P,
        Q,
        method: str = "sqrtm",
        selection: SelectionMode = SelectionMode.LARGEST
    ) -> "LRUBlock":
        """Apply balanced truncation to the LRU layer."""
        reduced_lru = self.lru.reduce(rank, P, Q, method, selection)
        return eqx.tree_at(lambda block: block.lru, self, reduced_lru)


class LRU(eqx.Module):
    """
    Full LRU model for sequence classification/regression.
    
    Architecture:
    1. Linear encoder: input_dim -> hidden_dim
    2. Stack of LRU blocks
    3. Global average pooling (for classification) or subsampling (for regression)
    4. Linear output layer
    
    Attributes:
        linear_encoder: Input projection
        blocks: List of LRU blocks
        linear_layer: Output projection
        classification: Whether this is a classification task
        output_step: For regression, subsample every output_step steps
    """
    
    linear_encoder: eqx.nn.Linear
    blocks: List[LRUBlock]
    linear_layer: eqx.nn.Linear
    classification: bool
    output_step: int
    
    # Flags for training behavior
    stateful: bool = True
    nondeterministic: bool = True

    def __init__(
        self,
        num_blocks: int,
        input_dim: int,
        state_dim: int,
        hidden_dim: int,
        output_dim: int,
        classification: bool = True,
        output_step: int = 1,
        r_min: float = 0.9,
        r_max: float = 0.999,
        max_phase: float = 6.28,
        drop_rate: float = 0.1,
        *,
        key: jr.PRNGKey
    ):
        """
        Initialize LRU model.
        
        Args:
            num_blocks: Number of LRU blocks
            input_dim: Dimension of input features
            state_dim: Dimension of recurrent state (N)
            hidden_dim: Dimension of hidden layers (H)
            output_dim: Dimension of output (e.g., number of classes)
            classification: True for classification, False for regression
            output_step: For regression, output every output_step steps
            r_min: Minimum eigenvalue magnitude
            r_max: Maximum eigenvalue magnitude
            max_phase: Maximum eigenvalue phase
            drop_rate: Dropout probability
            key: JAX random key
        """
        encoder_key, *block_keys, output_key = jr.split(key, num_blocks + 2)
        
        self.linear_encoder = eqx.nn.Linear(input_dim, hidden_dim, key=encoder_key)
        self.blocks = [
            LRUBlock(state_dim, hidden_dim, r_min, r_max, max_phase, drop_rate, key=k)
            for k in block_keys
        ]
        self.linear_layer = eqx.nn.Linear(hidden_dim, output_dim, key=output_key)
        self.classification = classification
        self.output_step = output_step

    def __call__(self, x: jnp.ndarray, state, key: jr.PRNGKey):
        """
        Forward pass through the model.
        
        Args:
            x: Input sequence, shape (seq_len, input_dim)
            state: Batch norm states
            key: Random key for dropout
            
        Returns:
            predictions: Model output
            state: Updated batch norm states
        """
        drop_keys = jr.split(key, len(self.blocks))
        
        # Encode input
        x = jax.vmap(self.linear_encoder)(x)
        
        # Process through blocks
        for block, k in zip(self.blocks, drop_keys):
            x, state = block(x, state, key=k)
        
        # Output
        if self.classification:
            x = jnp.mean(x, axis=0)  # Global average pooling
            x = jax.nn.softmax(self.linear_layer(x), axis=0)
        else:
            x = x[self.output_step - 1 :: self.output_step]  # Subsample
            x = jax.nn.tanh(jax.vmap(self.linear_layer)(x))
        
        return x, state
    
    @property
    def num_blocks(self) -> int:
        """Number of LRU blocks."""
        return len(self.blocks)
    
    def get_state_dims(self) -> List[int]:
        """Get state dimensions for all blocks."""
        return [block.state_dim for block in self.blocks]
    
    def get_total_state_dim(self) -> int:
        """Get total state dimension across all blocks."""
        return sum(self.get_state_dims())
    
    def get_all_hankel_singular_values(self) -> Dict[str, Dict[str, Any]]:
        """
        Compute Hankel singular values for all blocks.
        
        Returns:
            Dictionary mapping block names to {P, Q, g} dicts
        """
        result = {}
        for i, block in enumerate(self.blocks):
            P, Q, g = block.get_hankel_singular_values()
            result[f'block_{i}'] = {'P': P, 'Q': Q, 'g': g}
        return result

    def get_reduction_analysis(
        self,
        hsv_dict: Dict[str, Dict[str, Any]],
        hankel_tol: float
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get reduction analysis for all blocks.
        
        Args:
            hsv_dict: Output from get_all_hankel_singular_values()
            hankel_tol: Energy conservation tolerance
            
        Returns:
            Dictionary mapping block names to analysis results
        """
        analyses = {}
        for i, block in enumerate(self.blocks):
            g = hsv_dict[f'block_{i}']['g']
            analyses[f'block_{i}'] = block.get_reduction_analysis(g, hankel_tol=hankel_tol)
        return analyses
    
    def reduce(
        self,
        ranks: Union[int, List[int]],
        hsv_dict: Dict[str, Dict[str, Any]],
        method: str = "sqrtm",
        selection: SelectionMode = SelectionMode.LARGEST
    ) -> "LRU":
        """
        Apply balanced truncation to all blocks.
        
        Args:
            ranks: Target rank(s). If int, apply to all blocks.
                   If list, must match number of blocks.
            hsv_dict: Output from get_all_hankel_singular_values()
            method: "sqrtm" or "chol" for transformation
            selection: Which states to keep
            
        Returns:
            New LRU model with reduced dimensions
        """
        if isinstance(ranks, int):
            ranks = [ranks] * len(self.blocks)
        
        if len(ranks) != len(self.blocks):
            raise ValueError(f"Number of ranks ({len(ranks)}) must match number of blocks ({len(self.blocks)}).")
        
        new_blocks = []
        for i, (block, rank) in enumerate(zip(self.blocks, ranks)):
            if rank < block.state_dim:
                P = hsv_dict[f'block_{i}']['P']
                Q = hsv_dict[f'block_{i}']['Q']
                new_blocks.append(block.reduce(rank, P, Q, method, selection))
            else:
                new_blocks.append(block)  # No reduction needed
        
        return eqx.tree_at(lambda model: model.blocks, self, new_blocks)

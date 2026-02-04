"""
Balanced truncation for LTI system reduction.

This module implements the balanced truncation algorithm for reducing
the state dimension of discrete-time LTI systems while preserving
input-output behavior.
"""

from typing import Tuple

import numpy as np
import scipy.linalg as la

from compressm.reduction.hsv import SelectionMode, diagonalize_lti


def balanced_realization_transformation(
    P: np.ndarray,
    Q: np.ndarray,
    method: str = "chol",
    eps: float = 1e-9
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the balanced realization transformation matrices.
    
    The balanced realization makes the controllability and observability
    Gramians equal and diagonal, with the diagonal elements being the
    Hankel singular values.
    
    Args:
        P: Controllability Gramian
        Q: Observability Gramian
        method: "sqrtm" for matrix square root, "chol" for Cholesky decomposition
        eps: Regularization for numerical stability
        
    Returns:
        T: Transformation matrix
        T_inv: Inverse transformation matrix
    """
    n = P.shape[0]
    
    if method == "sqrtm":
        P_sqrt = la.sqrtm(P + eps * np.eye(n))
        U, Sd, V = la.svd(P_sqrt @ Q @ P_sqrt)
        T = P_sqrt @ U @ np.diag(1.0 / np.sqrt(np.sqrt(Sd)))
    elif method == "chol":
        Lo = la.cholesky(Q + eps * np.eye(n), lower=True)
        Lc = la.cholesky(P + eps * np.eye(n), lower=True)
        U, S, VT = np.linalg.svd(Lo.T @ Lc)
        T = Lc @ VT.T @ np.diag(1.0 / np.sqrt(S))
    else:
        raise ValueError(f"Unknown method: {method}. Use 'sqrtm' or 'chol'.")
    
    T_inv = np.linalg.inv(T)
    return T, T_inv


def reduce_discrete_lti(
    lambdas: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    rank: int,
    method: str = "chol",
    selection: SelectionMode = SelectionMode.LARGEST
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reduce a discrete-time LTI system using balanced truncation.
    
    This is the core reduction algorithm. It transforms the system to
    balanced coordinates where states are ordered by their Hankel singular
    values, then truncates to keep the specified number of states.
    
    Args:
        lambdas: Eigenvalues (diagonal of A), shape (N,)
        B: Input matrix, shape (N, H)
        C: Output matrix, shape (H, N)
        P: Controllability Gramian, shape (N, N)
        Q: Observability Gramian, shape (N, N)
        rank: Target dimension (number of states to keep)
        method: "sqrtm" or "chol" for transformation computation
        selection: Which states to keep:
            - LARGEST: Keep states with largest HSVs (recommended)
            - SMALLEST: Keep states with smallest HSVs (ablation)
            - RANDOM: Random selection (ablation)
            
    Returns:
        A_red: Reduced state matrix, shape (rank, rank)
        B_red: Reduced input matrix, shape (rank, H)
        C_red: Reduced output matrix, shape (H, rank)
        
    Raises:
        ValueError: If rank >= current dimension
    """
    n = lambdas.shape[0]
    
    if rank >= n:
        raise ValueError(f"Rank ({rank}) must be smaller than current dimension ({n}).")
    
    if selection == SelectionMode.RANDOM:
        # Random selection: pick random states without transformation
        permuted_indices = np.random.permutation(n)
        selected_indices = permuted_indices[:rank]
        
        lambdas_red = lambdas[selected_indices]
        B_red = B[selected_indices, :]
        C_red = C[:, selected_indices]
        
        A_red = np.matrix(np.diag(lambdas_red))
        B_red = np.matrix(B_red)
        C_red = np.matrix(C_red)
        
    else:
        # Balanced truncation: transform to balanced coordinates
        T, T_inv = balanced_realization_transformation(P, Q, method=method)
        
        # Transform system matrices
        A = np.matrix(np.diag(lambdas))
        B = np.matrix(B)
        C = np.matrix(C)
        
        Ab = T_inv @ A @ T
        Bb = T_inv @ B
        Cb = C @ T
        
        if selection == SelectionMode.LARGEST:
            # Keep first 'rank' states (largest HSVs after balancing)
            A_red = Ab[:rank, :rank]
            B_red = Bb[:rank, :]
            C_red = Cb[:, :rank]
        elif selection == SelectionMode.SMALLEST:
            # Keep last 'rank' states (smallest HSVs)
            A_red = Ab[n-rank:, n-rank:]
            B_red = Bb[n-rank:, :]
            C_red = Cb[:, n-rank:]
        else:
            raise ValueError(f"Unknown selection mode: {selection}")
    
    return A_red, B_red, C_red


def lti_to_lru(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert an LTI system to LRU parameterization.
    
    The LRU uses a specific parameterization:
    - nu_log, theta_log: Log-space representation of eigenvalues
    - B_re, B_im: Real/imaginary parts of input projection
    - C_re, C_im: Real/imaginary parts of output projection
    - gammas: Normalization factors
    
    Args:
        A: State matrix, shape (N, N)
        B: Input matrix, shape (N, H)
        C: Output matrix, shape (H, N)
        
    Returns:
        nu_log: Log of negative log of eigenvalue magnitudes
        theta_log: Log of eigenvalue phases
        B_re: Real part of B (unnormalized by gamma)
        B_im: Imaginary part of B (unnormalized by gamma)
        C_re: Real part of C
        C_im: Imaginary part of C
        gammas: Normalization factors sqrt(1 - |lambda|^2)
    """
    # Diagonalize to get eigenvalues
    lambdas, B, C = diagonalize_lti(A, B, C)
    
    # Convert eigenvalues to log-space parameterization
    nu_log = np.log(-np.log(np.abs(lambdas)))
    theta_log = np.log(np.angle(lambdas) % (2 * np.pi))
    
    # Compute normalization factors
    gammas = np.sqrt(1 - np.abs(lambdas) ** 2)
    
    # Extract real and imaginary parts
    B_re = B.real
    B_im = B.imag
    C_re = C.real
    C_im = C.imag
    
    return nu_log, theta_log, B_re, B_im, C_re, C_im, gammas

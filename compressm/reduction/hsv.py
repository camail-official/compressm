"""
Hankel Singular Value computation and selection modes for SSM reduction.

This module provides functionality to compute Hankel singular values (HSVs)
for diagonal LTI systems and select which states to keep during reduction.
"""

from enum import Enum
from typing import Dict, Any, Tuple, Optional

import numpy as np
import scipy.linalg as la


class SelectionMode(Enum):
    """
    Selection mode for choosing which states to keep during reduction.
    
    Modes:
        LARGEST: Keep states with the largest Hankel singular values (default).
                 These are the most controllable/observable states.
        SMALLEST: Keep states with the smallest Hankel singular values.
                  Used as an ablation baseline to verify HSV importance.
        RANDOM: Randomly select states to keep.
                Used as an ablation baseline for comparison.
    """
    LARGEST = "largest"
    SMALLEST = "smallest"
    RANDOM = "random"


def dlyap_direct_diagonal(lambdas: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    Solve discrete Lyapunov equation for diagonal case.
    
    For diagonal A = diag(lambdas), solves A @ X @ A.H + Q = X
    
    Args:
        lambdas: Diagonal elements of A matrix
        Q: Right-hand side matrix
        
    Returns:
        Solution matrix X
    """
    lhs = np.kron(lambdas, np.conj(lambdas))
    lhs = 1 - lhs
    x = Q.flatten() / lhs
    X = np.reshape(x, Q.shape)
    return X


def hankel_singular_values_diagonal(
    lambdas: np.ndarray,
    B: np.ndarray,
    C: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Hankel singular values for a diagonal LTI system.
    
    For a discrete-time LTI system with diagonal A = diag(lambdas),
    computes the controllability Gramian P, observability Gramian Q,
    and Hankel singular values g = sqrt(eig(P @ Q)).
    
    The Hankel singular values measure the joint controllability and
    observability of each state, indicating its importance for the
    input-output behavior of the system.
    
    Args:
        lambdas: Eigenvalues (diagonal elements of A), shape (N,)
        B: Input matrix, shape (N, H)
        C: Output matrix, shape (H, N)
        
    Returns:
        P: Controllability Gramian, shape (N, N)
        Q: Observability Gramian, shape (N, N)
        g: Hankel singular values, shape (N,)
    """
    B = np.matrix(B)
    C = np.matrix(C)
    P = dlyap_direct_diagonal(lambdas, B @ B.H)
    Q = dlyap_direct_diagonal(np.conjugate(lambdas), C.H @ C)
    PQ = P @ Q
    g = np.sqrt(np.linalg.eigvals(PQ).real)
    return P, Q, g


def diagonalize_lti(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Diagonalize an LTI system by eigendecomposition.
    
    Transforms (A, B, C) to diagonal form (diag(lambdas), Bd, Cd)
    where lambdas are the eigenvalues of A.
    
    Args:
        A: State matrix, shape (N, N)
        B: Input matrix, shape (N, H)
        C: Output matrix, shape (H, N)
        
    Returns:
        lambdas: Eigenvalues of A
        Bd: Transformed input matrix
        Cd: Transformed output matrix
    """
    lambdas, T = la.eig(A)
    T_inv = np.linalg.inv(T)
    Bd = T_inv @ B
    Cd = C @ T
    return lambdas, Bd, Cd


def reduction_analysis(
    g: np.ndarray,
    hankel_tol: float
) -> Dict[str, Any]:
    """
    Analyze reduction potential based on Hankel singular values.
    
    Computes the minimum rank needed to preserve a given fraction of
    the total Hankel energy (sum of squared singular values).
    
    Args:
        g: Hankel singular values (unsorted)
        hankel_tol: Tolerance for energy conservation. A value of 0.99
                   means keep 99% of the total Hankel energy.
                   
    Returns:
        Dictionary containing:
            - hankel_singular_values: Sorted HSVs (descending)
            - cumulative_energy: Cumulative energy fraction
            - total_energy: Total Hankel energy
            - recommended_ranks: Dict with 'threshold' key giving min rank
    """
    # Handle NaNs
    g_np = np.asarray(g)
    g_clean = np.where(np.isnan(g_np), 0.0, g_np)
    
    # Sort in descending order
    g_sorted = np.sort(g_clean)[::-1]
    total_energy = np.sum(g_sorted)
    cumulative_energy = np.cumsum(g_sorted) / total_energy
    
    # Find minimum rank to achieve tolerance (keep hankel_tol fraction of energy)
    # e.g., tol=0.99 means keep states until 99% of energy is captured
    threshold_rank = np.argmax(cumulative_energy >= hankel_tol) + 1
    
    return {
        'hankel_singular_values': g_sorted,
        'cumulative_energy': cumulative_energy,
        'total_energy': total_energy,
        'recommended_ranks': {
            'threshold': int(threshold_rank)
        }
    }

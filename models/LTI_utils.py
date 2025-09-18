# from sympy import Lambda
import equinox as eqx
import jax.numpy as jnp
import jax

import matplotlib.pyplot as plt
import numpy as np

import scipy.linalg as la

# from https://github.com/forgi86/lru-reduction/blob/main/lru/reduction.py
def hankel_singular_values(A, B, C):
    P = la.solve_discrete_lyapunov(A, B @ B.T.conjugate())
    Q = la.solve_discrete_lyapunov(A.T, C.T.conjugate() @ C)
    g = np.sqrt(np.linalg.eigvals(P @ Q))
    return g

# from https://github.com/forgi86/lru-reduction/blob/main/lru/reduction.py
def dlyap_direct_diagonal(lambdas, Q):
    lhs = np.kron(lambdas, np.conj(lambdas))  # Kronecker product
    lhs = 1 - lhs
    x = Q.flatten()/lhs
    X = np.reshape(x, Q.shape)
    return X

# from https://github.com/forgi86/lru-reduction/blob/main/lru/reduction.py
def hankel_singular_values_diagonal(lambdas, B, C):
    B = np.matrix(B)
    C = np.matrix(C)
    P = dlyap_direct_diagonal(lambdas, B @ B.H)
    Q = dlyap_direct_diagonal(np.conjugate(lambdas), C.H @ C)
    PQ = P @ Q
    g = np.sqrt(np.linalg.eigvals(PQ).real)
    return P, Q, g

def diagonalize_LTI(A, B, C):
    Lambdas, T = la.eig(A)
    T_inv = np.linalg.inv(T)
    Bd = T_inv @ B
    Cd = C @ T
    return Lambdas, Bd, Cd

# from https://github.com/forgi86/lru-reduction/blob/main/lru/reduction.py
def _balanced_realization_transformation(P, Q, method="sqrtm", eps=1e-9):
    if method == "sqrtm":
        P_sqrt = la.sqrtm(P + eps*np.eye(P.shape[0]))
        [U, Sd, V] = la.svd(P_sqrt @ Q @ P_sqrt)
        T = P_sqrt @ U @ np.diag(1 / (np.sqrt(np.sqrt(Sd))))
    elif method == "chol":
        Lo = la.cholesky(Q + eps*np.eye(Q.shape[0]), lower=True)
        Lc = la.cholesky(P + eps*np.eye(P.shape[0]), lower=True)
        U, S, VT = np.linalg.svd(Lo.T @ Lc)
        T = Lc @ VT.T @ np.diag(1/np.sqrt(S))
    return T

def reduce_discrete_LTI(Lambdas, B, C, P, Q, rank=None, method="sqrtm", randomize=False):
    """
    Reduce a continuous-time LTI system to a state-space representation with
    a reduced number of states.
    """

    # check that the rank is smaller than the number of states
    if rank is not None and rank >= Lambdas.shape[0]:
        raise ValueError("Rank must be smaller than the number of states.")
    # if rank is not given, return as is
    if rank is None:
        raise ValueError("Rank must be specified for reduction.")
    
    if randomize:
        permuted_indices = np.random.permutation(Lambdas.shape[0])
        selected_indices = np.sort(permuted_indices[:rank])
        Lambdas = Lambdas[selected_indices]
        B = B[selected_indices, :]
        C = C[:, selected_indices]

        Ared = np.matrix(np.diag(Lambdas))
        Bred = np.matrix(B)
        Cred = np.matrix(C)

    else:
        # get the transformation matrix
        T = _balanced_realization_transformation(P, Q, method=method)
        T_inv = la.inv(T)

        # reduce the system matrices
        A = np.matrix(np.diag(Lambdas))
        B = np.matrix(B)
        C = np.matrix(C)

        Ab = T_inv @ A @ T
        Bb = T_inv @ B
        Cb = C @ T

        Ared = Ab[:rank, :rank]
        Bred = Bb[:rank, :]
        Cred = Cb[:, :rank]

    return Ared, Bred, Cred

def LTI_to_LRU(A, B, C):
    """
    Convert an LTI system to an LRU layer.
    """
    # Convert Lambdas to log space
    Lambdas, B, C = diagonalize_LTI(A, B, C)

    nu_log = jnp.log(-jnp.log(jnp.abs(Lambdas)))
    theta_log = jnp.log(jnp.angle(Lambdas) % (2*jnp.pi))

    # Normalize B by gammas
    gammas = jnp.sqrt(1 - jnp.abs(Lambdas) ** 2)
    B_re = B.real
    B_im = B.imag

    # Convert C to real and imaginary parts
    C_re = C.real
    C_im = C.imag

    return nu_log, theta_log, B_re, B_im, C_re, C_im, gammas

@eqx.filter_jit
def _compute_reduction_analysis(g_clean, hankel_tol):
    """Pure computation part - JIT compilable."""
    g_sorted = jnp.sort(g_clean)[::-1]
    total_energy = jnp.sum(g_sorted)
    cumulative_energy = jnp.cumsum(g_sorted) / total_energy
    
    # Compute all thresholds
    threshold_tol = jnp.argmax(cumulative_energy >= (1 - hankel_tol)) + 1 if hankel_tol is not None else 0
    threshold_90 = jnp.argmax(cumulative_energy >= 0.9) + 1
    threshold_95 = jnp.argmax(cumulative_energy >= 0.95) + 1
    threshold_99 = jnp.argmax(cumulative_energy >= 0.99) + 1
    threshold_999 = jnp.argmax(cumulative_energy >= 0.999) + 1
    threshold_9999 = jnp.argmax(cumulative_energy >= 0.9999) + 1
    
    return g_sorted, cumulative_energy, total_energy, threshold_tol, threshold_90, threshold_95, threshold_99, threshold_999, threshold_9999

def reduction_analysis(g, hankel_tol=None):
    """Get analysis of reduction potential using JAX."""
    # TODO: investigate when computations run into nans
    g_clean = jnp.where(jnp.isnan(g), 0.0, g)
    
    # Compute thresholds
    g_sorted, cumulative_energy, total_energy, threshold_tol, threshold_90, threshold_95, threshold_99, threshold_999, threshold_9999 = _compute_reduction_analysis(g_clean, hankel_tol)
    
    if hankel_tol is not None:
        return {
            'hankel_singular_values': g_sorted,
            'cumulative_energy': cumulative_energy,
            'total_energy': total_energy,
            'recommended_ranks': {
                'threshold': threshold_tol
            }
        }
    else:
        return {
            'hankel_singular_values': g_sorted,
            'cumulative_energy': cumulative_energy,
            'total_energy': total_energy,
            'recommended_ranks': {
                '90%': threshold_90,
                '95%': threshold_95,
                '99%': threshold_99,
                '99.9%': threshold_999,
                '99.99%': threshold_9999,
            }
        }
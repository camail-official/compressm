import equinox as eqx
import jax.numpy as jnp
import jax
import jax.scipy.linalg as jla

# Some functions are Jax-adapted from https://github.com/forgi86/lru-reduction/blob/main/lru/reduction.py

@eqx.filter_jit
def diagonalize_LTI(A, B, C):
    """LTI system diagonalization."""
    Lambdas, T = jnp.linalg.eig(A)
    T_inv = jnp.linalg.inv(T)
    Bd = T_inv @ B
    Cd = C @ T
    return Lambdas, Bd, Cd

@eqx.filter_jit
def _balanced_realization_transformation(P, Q, method="sqrtm", eps=1e-9):
    """Balanced realization transformation."""
    def sqrtm_method():
        P_sqrt = jla.sqrtm(P + eps * jnp.eye(P.shape[0]))
        U, Sd, VT = jnp.linalg.svd(P_sqrt @ Q @ P_sqrt)
        return P_sqrt @ U @ jnp.diag(1 / jnp.sqrt(jnp.sqrt(jnp.maximum(Sd, eps))))
    
    def chol_method():
        Lo = jla.cholesky(Q + eps * jnp.eye(Q.shape[0]), lower=True)
        Lc = jla.cholesky(P + eps * jnp.eye(P.shape[0]), lower=True)
        U, S, VT = jnp.linalg.svd(Lo.T @ Lc)
        return Lc @ VT.T @ jnp.diag(1/jnp.sqrt(jnp.maximum(S, eps)))
    
    # Use JAX conditional for JIT compatibility
    return jax.lax.cond(method == "sqrtm", sqrtm_method, chol_method)

@eqx.filter_jit
def reduce_discrete_LTI(Lambdas, B, C, P, Q, rank=None, eps=1e-9):
    """Discrete LTI system reduction with error handling."""
    # Get transformation matrix
    T = _balanced_realization_transformation(P, Q, method="sqrtm", eps=eps)
    T_inv = jnp.linalg.inv(T)
    
    # Create system matrices
    A = jnp.diag(Lambdas)
    
    # Transform to balanced coordinates
    Ab = T_inv @ A @ T
    Bb = T_inv @ B
    Cb = C @ T
    
    # Reduce by truncation
    Ared = Ab[:rank, :rank]
    Bred = Bb[:rank, :]
    Cred = Cb[:, :rank]
    
    return Ared, Bred, Cred

@eqx.filter_jit
def LTI_to_LRU(A, B, C):
    """LTI to LRU conversion."""
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
def dlyap_direct_diagonal(lambdas, Q):
    """Solve discrete Lyapunov equation for diagonal system."""
    lhs = jnp.kron(lambdas, jnp.conj(lambdas))  # Kronecker product
    lhs = 1 - lhs
    x = Q.flatten()/lhs
    X = jnp.reshape(x, Q.shape)
    return X

@eqx.filter_jit
def hankel_singular_values_diagonal(lambdas, B, C, eps=1e-9):
    """Compute Hankel singular values for diagonal system."""
    # TODO: justify the stabilization scheme
    BBT = B @ B.T.conj() + eps * jnp.eye(B.shape[0])
    CTC = C.T.conj() @ C + eps * jnp.eye(C.shape[1])
    
    P = dlyap_direct_diagonal(lambdas, BBT)
    Q = dlyap_direct_diagonal(jnp.conj(lambdas), CTC)
    PQ = P @ Q + eps * jnp.eye(P.shape[0])
    g = jnp.sqrt(jnp.maximum(jnp.linalg.eigvals(PQ).real, eps))
    return P, Q, g

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
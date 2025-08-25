from typing import List

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import nn
from jax.nn.initializers import normal
import math
from jax import random

def simple_uniform_init(rng, shape, std=1.):
    weights = random.uniform(rng, shape)*2.*std - std
    return weights

class GLU(eqx.Module):
    w1: eqx.nn.Linear
    w2: eqx.nn.Linear

    def __init__(self, input_dim, output_dim, key):
        w1_key, w2_key = jr.split(key, 2)
        self.w1 = eqx.nn.Linear(input_dim, output_dim, use_bias=True, key=w1_key)
        self.w2 = eqx.nn.Linear(input_dim, output_dim, use_bias=True, key=w2_key)

    def __call__(self, x):
        return self.w1(x) * jax.nn.sigmoid(self.w2(x))

# Parallel scan operations
@jax.vmap
def binary_operator(q_i, q_j):
    """ Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A.
        Args:
            q_i: tuple containing A_i and Bu_i at position i       (P,), (P,)
            q_j: tuple containing A_j and Bu_j at position j       (P,), (P,)
        Returns:
            new element ( A_out, Bu_out )
    """
    A_i, b_i = q_i
    A_j, b_j = q_j

    N = A_i.size // 4
    iA_ = A_i[0 * N: 1 * N]
    iB_ = A_i[1 * N: 2 * N]
    iC_ = A_i[2 * N: 3 * N]
    iD_ = A_i[3 * N: 4 * N]
    jA_ = A_j[0 * N: 1 * N]
    jB_ = A_j[1 * N: 2 * N]
    jC_ = A_j[2 * N: 3 * N]
    jD_ = A_j[3 * N: 4 * N]
    A_new = jA_ * iA_ + jB_ * iC_
    B_new = jA_ * iB_ + jB_ * iD_
    C_new = jC_ * iA_ + jD_ * iC_
    D_new = jC_ * iB_ + jD_ * iD_
    Anew = jnp.concatenate([A_new, B_new, C_new, D_new])

    b_i1 = b_i[0:N]
    b_i2 = b_i[N:]

    new_b1 = jA_ * b_i1 + jB_ * b_i2
    new_b2 = jC_ * b_i1 + jD_ * b_i2
    new_b = jnp.concatenate([new_b1, new_b2])

    return Anew, new_b + b_j

def apply_linoss_im(A_diag, B, C_tilde, input_sequence, step):
    """Compute the LxH output of LinOSS-IM given an LxH input.
    Args:
        A_diag  (float32):   diagonal state matrix   (P,)
        B       (complex64): input matrix            (P, H)
        C       (complex64): output matrix           (H, P)
        input_sequence (float32): input sequence of features    (L, H)
        step    (float):     discretization time-step $\Delta_t$  (P,)
    Returns:
        outputs (float32): the SSM outputs (LinOSS_IMEX layer preactivations)      (L, H)
    """
    Bu_elements = jax.vmap(lambda u: B @ u)(input_sequence)

    schur_comp = 1. / (1. + step ** 2. * A_diag)
    M_IM_11 = 1. - step ** 2. * A_diag * schur_comp
    M_IM_12 = -1. * step * A_diag * schur_comp
    M_IM_21 = step * schur_comp
    M_IM_22 = schur_comp

    M_IM = jnp.concatenate([M_IM_11, M_IM_12, M_IM_21, M_IM_22])

    M_IM_elements = M_IM * jnp.ones((input_sequence.shape[0],
                                         4 * A_diag.shape[0]))

    F1 = M_IM_11 * Bu_elements * step
    F2 = M_IM_21 * Bu_elements * step
    F = jnp.hstack((F1, F2))

    _, xs = jax.lax.associative_scan(binary_operator, (M_IM_elements, F))
    ys = xs[:, A_diag.shape[0]:]

    return jax.vmap(lambda x: (C_tilde @ x).real)(ys)


def apply_linoss_imex(A_diag, B, C, input_sequence, step):
    """Compute the LxH output of of LinOSS-IMEX given an LxH input.
    Args:
        A_diag  (float32):   diagonal state matrix   (P,)
        B       (complex64): input matrix            (P, H)
        C       (complex64): output matrix           (H, P)
        input_sequence (float32): input sequence of features    (L, H)
        step    (float):     discretization time-step $\Delta_t$  (P,)
    Returns:
        outputs (float32): the SSM outputs (LinOSS_IMEX layer preactivations)      (L, H)
    """
    Bu_elements = jax.vmap(lambda u: B @ u)(input_sequence)

    A_ = jnp.ones_like(A_diag)
    B_ = -1. * step * A_diag
    C_ = step
    D_ = 1. - (step ** 2.) * A_diag

    M_IMEX = jnp.concatenate([A_, B_, C_, D_])

    M_IMEX_elements = M_IMEX * jnp.ones((input_sequence.shape[0],
                                          4 * A_diag.shape[0]))

    F1 = Bu_elements * step
    F2 = Bu_elements * (step ** 2.)
    F = jnp.hstack((F1, F2))

    _, xs = jax.lax.associative_scan(binary_operator, (M_IMEX_elements, F))
    ys = xs[:, A_diag.shape[0]:]

    return jax.vmap(lambda x: (C @ x).real)(ys)

class LinOSSLayer(eqx.Module):
    A_diag: jax.Array
    B: jax.Array
    C: jax.Array
    D: jax.Array
    steps: jax.Array
    discretization: str

    def __init__(
        self,
        ssm_size,
        H,
        discretization,
        *,
        key
    ):

        B_key, C_key, D_key, A_key, step_key, key = jr.split(key, 6)
        self.A_diag = random.uniform(A_key, shape=(ssm_size,))
        self.B = simple_uniform_init(B_key,shape=(ssm_size, H, 2),std=1./math.sqrt(H))
        self.C = simple_uniform_init(C_key,shape=(H, ssm_size, 2),std=1./math.sqrt(ssm_size))
        self.D = normal(stddev=1.0)(D_key, (H,))
        self.steps = random.uniform(step_key,shape=(ssm_size,))
        self.discretization = discretization

    def __call__(self, input_sequence):
        A_diag = nn.relu(self.A_diag)

        B_complex = self.B[..., 0] + 1j * self.B[..., 1]
        C_complex = self.C[..., 0] + 1j * self.C[..., 1]

        steps = nn.sigmoid(self.steps)
        if self.discretization == 'IMEX':
            ys = apply_linoss_imex(A_diag, B_complex, C_complex, input_sequence, steps)
        elif self.discretization == 'IM':
            ys = apply_linoss_im(A_diag, B_complex, C_complex, input_sequence, steps)
        else:
            print('Discretization type not implemented')

        Du = jax.vmap(lambda u: self.D * u)(input_sequence)
        return ys + Du


class LinOSSBlock(eqx.Module):

    norm: eqx.nn.BatchNorm
    ssm: LinOSSLayer
    glu: GLU
    drop: eqx.nn.Dropout

    def __init__(
        self,
        ssm_size,
        H,
        discretization,
        drop_rate=0.05,
        *,
        key
    ):
        ssmkey, glukey = jr.split(key, 2)
        self.norm = eqx.nn.BatchNorm(
            input_size=H, axis_name="batch", channelwise_affine=False
        )
        self.ssm = LinOSSLayer(
            ssm_size,
            H,
            discretization,
            key=ssmkey,
        )
        self.glu = GLU(H, H, key=glukey)
        self.drop = eqx.nn.Dropout(p=drop_rate)

    def __call__(self, x, state, *, key):
        """Compute LinOSS block."""
        dropkey1, dropkey2 = jr.split(key, 2)
        skip = x
        x, state = self.norm(x.T, state)
        x = x.T
        x = self.ssm(x)
        x = self.drop(jax.nn.gelu(x), key=dropkey1)
        x = jax.vmap(self.glu)(x)
        x = self.drop(x, key=dropkey2)
        x = skip + x
        return x, state


class LinOSS(eqx.Module):
    linear_encoder: eqx.nn.Linear
    blocks: List[LinOSSBlock]
    linear_layer: eqx.nn.Linear
    classification: bool
    output_step: int
    stateful: bool = True
    nondeterministic: bool = True
    lip2: bool = False

    def __init__(
        self,
        num_blocks,
        N,
        ssm_size,
        H,
        output_dim,
        classification,
        output_step,
        discretization,
        *,
        key
    ):

        linear_encoder_key, *block_keys, linear_layer_key, weightkey = jr.split(
            key, num_blocks + 3
        )
        self.linear_encoder = eqx.nn.Linear(N, H, key=linear_encoder_key)
        self.blocks = [
            LinOSSBlock(
                ssm_size,
                H,
                discretization,
                key=key,
            )
            for key in block_keys
        ]
        self.linear_layer = eqx.nn.Linear(H, output_dim, key=linear_layer_key)
        self.classification = classification
        self.output_step = output_step

    def __call__(self, x, state, key):
        """Compute LinOSS."""
        dropkeys = jr.split(key, len(self.blocks))
        x = jax.vmap(self.linear_encoder)(x)
        for block, key in zip(self.blocks, dropkeys):
            x, state = block(x, state, key=key)
        if self.classification:
            x = jnp.mean(x, axis=0)
            x = jax.nn.softmax(self.linear_layer(x), axis=0)
        else:
            x = x[self.output_step - 1 :: self.output_step]
            x = jax.nn.tanh(jax.vmap(self.linear_layer)(x))
        return x, state

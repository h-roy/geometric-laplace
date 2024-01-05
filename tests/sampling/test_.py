import time
import argparse

import jax
import matplotlib.pyplot as plt
import optax
import tree_math as tm
from flax import linen as nn
from jax import nn as jnn
from jax import numpy as jnp
from jax import random, jit
import pickle

from src.sampling.invsqrt_vp import inv_sqrt_vp
from src.sampling.low_rank import lanczos_tridiag

key = jax.random.PRNGKey(10)
params = 500
rank = 300
J = jax.random.normal(key, (rank, params))
prior_prec = 5.
GGN_lr = J.T @ J
GGN = J.T @ J + prior_prec * jnp.eye(params)
order = 300
Av = lambda v: GGN_lr @ v
v0 = jnp.ones(params)
eigvals, eigvecs = lanczos_tridiag(Av, v0, order)
recon = eigvecs @ jnp.diag(eigvals) @ eigvecs.T
recon_error = jnp.linalg.norm(GGN_lr - recon)/jnp.linalg.norm(GGN_lr) 
print("Reconstruction Relative Error: {}".format(recon_error))

mvp_fn = inv_sqrt_vp(eigvals, eigvecs, prior_prec)

def test_fn(test_vec):
    mvp_approx = mvp_fn(test_vec)
    ggn_sqrt = jax.scipy.linalg.sqrtm(GGN)
    gt = jnp.linalg.solve(ggn_sqrt, test_vec)
    return jnp.linalg.norm(mvp_approx - gt)/jnp.linalg.norm(gt)

num_tests = 5
test_vecs = [jax.random.normal(k, (params,)) * var for k, var in zip(jax.random.split(key, num_tests), jnp.arange(1, num_tests + 1))]
for i, test_vec in enumerate(test_vecs):
    relative_error = test_fn(test_vec)
    print("Test {} relative error:{}".format(i,relative_error))

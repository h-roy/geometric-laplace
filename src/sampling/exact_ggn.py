import jax
from jax import flatten_util
from jax import numpy as jnp
from src.helper import calculate_exact_ggn
from jax import random
from typing import Literal
from functools import partial

@partial(jax.jit, static_argnames=("loss", "model_fn", "n_params", "n_posterior_samples", "rank", "posterior_type"))
def exact_ggn_laplace(loss,
                       model_fn, 
                       params, 
                       x_train, 
                       y_train, 
                       n_params, 
                       rank,
                       alpha,
                       n_posterior_samples, 
                       key,
                       var=0.1,
                       posterior_type: Literal["low_rank", "full_ggn", "isotropic", "all"] = "low_rank"
                       ):
    ggn = calculate_exact_ggn(loss, model_fn, params, x_train, y_train, n_params)
    eigvals, eigvecs = jnp.linalg.eigh(ggn)
    def ggn_lr_vp(v):
        return eigvecs[:,-rank:] @ jnp.diag(1/jnp.sqrt(eigvals[-rank:]+ alpha)) @ v
    def ggn_vp(v):
        return eigvecs @ jnp.diag(1/jnp.sqrt(eigvals + alpha)) @ v
    eps = jax.random.normal(key, (n_posterior_samples, n_params))
    p0_flat, unravel_func_p = flatten_util.ravel_pytree(params)
    if posterior_type == "low_rank":
        lr_posterior_samples = jax.vmap(lambda single_eps: unravel_func_p(ggn_lr_vp(single_eps[:rank]) + p0_flat))(eps)
        return lr_posterior_samples
    elif posterior_type == "full_ggn":
        posterior_samples = jax.vmap(lambda single_eps: unravel_func_p(ggn_vp(single_eps) + p0_flat))(eps)
        return posterior_samples
    elif posterior_type == "isotropic":
        isotropic_posterior_samples = jax.vmap(lambda single_eps: unravel_func_p(var * single_eps + p0_flat))(eps)
        return isotropic_posterior_samples
    elif posterior_type == "all":
        def get_posteriors(single_eps):
            lr_sample = unravel_func_p(ggn_lr_vp(single_eps[:rank]) + p0_flat)
            posterior_sample = unravel_func_p(ggn_vp(single_eps) + p0_flat)
            isotropic_sample = unravel_func_p(var * single_eps + p0_flat)
            return lr_sample, posterior_sample, isotropic_sample
        lr_posterior_samples, posterior_samples, isotropic_posterior_samples = jax.vmap(get_posteriors)(eps)
        return lr_posterior_samples, posterior_samples, isotropic_posterior_samples

import jax
from jax import flatten_util
from jax import numpy as jnp
from src.helper import calculate_exact_ggn
from jax import random
from typing import Literal
from functools import partial

@partial(jax.jit, static_argnames=("loss", "model_fn", "n_steps", "n_params", "n_samples", "rank", "with_eigvals"))
def exact_nonkernel_diffusion(loss,
                              model_fn,
                              params,
                              n_steps,
                              n_samples,
                              alpha,
                              key,
                              n_params,
                              rank,
                              x_train,
                              y_train,
                              with_eigvals = True          
                              ):
    p0_flat, unravel_func_p = jax.flatten_util.ravel_pytree(params)
    eps = jax.random.normal(key, (n_samples, n_steps, rank))
    def rw_nonker(single_eps_path):
        params_ = p0_flat
        def body_fun(n, res):
            ggn = calculate_exact_ggn(loss, model_fn, unravel_func_p(res), x_train, y_train, n_params)
            eigvals, eigvecs = jnp.linalg.eigh(ggn)
            if with_eigvals:
                lr_sample = eigvecs[:,-rank:] @ jnp.diag(1/jnp.sqrt(alpha + eigvals)) @ single_eps_path[n]
            else:
                lr_sample = 1/jnp.sqrt(alpha) * eigvecs[:,-rank:] @ single_eps_path[n]
            params_ = res + 1/jnp.sqrt(n_steps) * lr_sample
            return params_
        v_ = jax.lax.fori_loop(0, n_steps - 1, body_fun, params_)
        return unravel_func_p(v_)
    nonker_posterior_samples = jax.vmap(rw_nonker)(eps)
    return nonker_posterior_samples

@partial(jax.jit, static_argnames=("loss", "model_fn", "n_steps", "n_params", "n_samples", "rank"))
def exact_kernel_diffusion(loss,
                           model_fn,
                           params,
                           n_steps,
                           n_samples,
                           alpha,
                           key,
                           n_params,
                           rank,
                           x_train,
                           y_train       
                          ):
    p0_flat, unravel_func_p = jax.flatten_util.ravel_pytree(params)
    eps = jax.random.normal(key, (n_samples, n_steps, n_params - rank))
    def rw_ker(single_eps_path):
        params_ = p0_flat
        def body_fun(n, res):
            ggn = calculate_exact_ggn(loss, model_fn, unravel_func_p(res), x_train, y_train, n_params)
            _, eigvecs = jnp.linalg.eigh(ggn)
            lr_sample = 1/jnp.sqrt(alpha) * eigvecs[:,:-rank] @ single_eps_path[n]
            params_ = res + 1/jnp.sqrt(n_steps) * lr_sample
            return params_
        v_ = jax.lax.fori_loop(0, n_steps - 1, body_fun, params_)
        return unravel_func_p(v_)
    ker_posterior_samples = jax.vmap(rw_ker)(eps)
    return ker_posterior_samples

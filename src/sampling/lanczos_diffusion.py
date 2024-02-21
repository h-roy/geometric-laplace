import jax
from jax import numpy as jnp
from src.helper import get_gvp_fun, get_ggn_vector_product
from src.sampling import lanczos_tridiag, unstable_lanczos_tridiag
from typing import Literal
from functools import partial

@partial(jax.jit, static_argnames=("model", "n_steps", "n_params", "n_samples", "rank", "diffusion_type", "likelihood"))
def lanczos_diffusion(model,
                      params,
                      n_steps,
                      n_samples,
                      alpha,
                      key,
                      n_params,
                      rank,
                      x_train,
                      likelihood: Literal["classification", "regression"] = "classification",
                      delta = 1.0,
                      diffusion_type: Literal["kernel", "non-kernel", "non-kernel-eigvals", "full-ggn"] = "kernel"          
                      ):
    p0_flat, unravel_func_p = jax.flatten_util.ravel_pytree(params)
    # if diffusion_type == "non-kernel" or diffusion_type == "non-kernel-eigvals":
    #     eps = jax.random.normal(key, (n_samples, n_steps, rank))
    # else:
    #     eps = jax.random.normal(key, (n_samples, n_steps, n_params))
    key_list = jax.random.split(key, (n_samples,))
    # v0 = jnp.ones(n_params)
    def diffusion(single_key):
        params_ = p0_flat
        key_path = jax.random.split(single_key, (n_steps,))
        def body_fun(n, res):
            # gvp = get_gvp_fun(model_fn, loss, unravel_func_p(res), x_train, y_train)
            gvp = get_gvp_fun(unravel_func_p(res), model, x_train, 10000, likelihood, "running")
            # gvp = get_ggn_vector_product(unravel_func_p(res), model, x_train, None, likelihood)
            if diffusion_type == "kernel":
                eps = jax.random.normal(key_path[n], (n_params,))
                gvp_ = lambda v: gvp(v) + delta * v
                eigvals, eigvecs = lanczos_tridiag(gvp_, eps, rank - 1)
                lr_sample = 1/jnp.sqrt(alpha) * 1/delta * (gvp_(eps) - eigvecs @ jnp.diag(eigvals) @ eigvecs.T @ eps)
            elif diffusion_type == "non-kernel":
                eps = jax.random.normal(key_path[n], (rank,))
                v0 = jnp.concatenate([eps, jnp.ones(n_params - rank)])
                _, eigvecs = lanczos_tridiag(gvp, v0, rank - 1)
                lr_sample = 1/jnp.sqrt(alpha) * eigvecs @ eps
            elif diffusion_type == "non-kernel-eigvals":
                eps = jax.random.normal(key_path[n], (rank,))
                v0 = jnp.concatenate([eps, jnp.ones(n_params - rank)])
                eigvals, eigvecs = lanczos_tridiag(gvp, v0, rank - 1)
                lr_sample = eigvecs @ jnp.diag(1/jnp.sqrt(alpha + eigvals)) @ eps
                #lr_sample = ((1/jnp.sqrt(alpha + eigvals).reshape(-1,1) * eigvecs) @ single_eps_path[n]
            elif diffusion_type == "full-ggn":
                eps = jax.random.normal(key_path[n], (n_params,))
                gvp_ = lambda v: gvp(v) + alpha * v
                eigvals, eigvecs = lanczos_tridiag(gvp_, eps, rank - 1)
                lr_sample = 1/jnp.sqrt(alpha) * eps + eigvecs @ jnp.diag(1/jnp.sqrt(eigvals + alpha) - 1/jnp.sqrt(alpha)) @ eigvecs.T @ eps
            params_ = res + 1/jnp.sqrt(n_steps) * lr_sample
            return params_
        v_ = jax.lax.fori_loop(0, n_steps - 1, body_fun, params_)
        return unravel_func_p(v_)
    # diffusion_posterior_samples = jax.vmap(diffusion)(key_list)
    diffusion_posterior_samples = jax.lax.map(diffusion, key_list)
    return diffusion_posterior_samples


@partial(jax.jit, static_argnames=("model", "n_steps", "n_params", "n_samples", "rank", "basis_dim", "diffusion_type", "likelihood"))
def unstable_lanczos_diffusion(model,
                      params,
                      n_steps,
                      n_samples,
                      alpha,
                      key,
                      n_params,
                      rank,
                      x_train,
                      basis_dim,
                      likelihood: Literal["classification", "regression"] = "classification",
                      delta = 1.0,
                      diffusion_type: Literal["kernel", "non-kernel", "non-kernel-eigvals", "full-ggn"] = "kernel"          
                      ):
    p0_flat, unravel_func_p = jax.flatten_util.ravel_pytree(params)
    # if diffusion_type == "non-kernel" or diffusion_type == "non-kernel-eigvals":
    #     eps = jax.random.normal(key, (n_samples, n_steps, rank))
    # else:
    #     eps = jax.random.normal(key, (n_samples, n_steps, n_params))
    key_list = jax.random.split(key, (n_samples,))
    # v0 = jnp.ones(n_params)
    def diffusion(single_key):
        params_ = p0_flat
        key_path = jax.random.split(single_key, (n_steps,))
        def body_fun(n, res):
            # gvp = get_gvp_fun(model_fn, loss, unravel_func_p(res), x_train, y_train)
            gvp = get_gvp_fun(unravel_func_p(res), model, x_train, 10000, likelihood, "running")
            # gvp = get_ggn_vector_product(unravel_func_p(res), model, x_train, None, likelihood)
            if diffusion_type == "kernel":
                eps = jax.random.normal(key_path[n], (n_params,))
                gvp_ = lambda v: gvp(v) + delta * v
                inv_sqrt_vp = unstable_lanczos_tridiag(gvp_, eps, rank - 1, basis_dim)
                lr_sample = 1/jnp.sqrt(alpha) * 1/delta * (gvp_(eps) - inv_sqrt_vp)
            elif diffusion_type == "non-kernel":
                eps = jax.random.normal(key_path[n], (rank,))
                v0 = jnp.concatenate([eps, jnp.ones(n_params - rank)])
                _, eigvecs = lanczos_tridiag(gvp, v0, rank - 1)
                lr_sample = 1/jnp.sqrt(alpha) * eigvecs @ eps
            elif diffusion_type == "non-kernel-eigvals":
                eps = jax.random.normal(key_path[n], (rank,))
                v0 = jnp.concatenate([eps, jnp.ones(n_params - rank)])
                lr_sample = unstable_lanczos_tridiag(gvp, v0, rank - 1, basis_dim)
                #lr_sample = ((1/jnp.sqrt(alpha + eigvals).reshape(-1,1) * eigvecs) @ single_eps_path[n]
            elif diffusion_type == "full-ggn":
                eps = jax.random.normal(key_path[n], (n_params,))
                gvp_ = lambda v: gvp(v) + alpha * v
                # inv_sqrt_vp = unstable_lanczos_tridiag(gvp_, eps, rank - 1, basis_dim)
                eigvals, eigvecs = lanczos_tridiag(gvp_, eps, rank - 1)
                lr_sample = 1/jnp.sqrt(alpha) * eps + eigvecs @ jnp.diag(1/jnp.sqrt(eigvals + alpha) - 1/jnp.sqrt(alpha)) @ eigvecs.T @ eps
            params_ = res + 1/jnp.sqrt(n_steps) * lr_sample
            return params_
        v_ = jax.lax.fori_loop(0, n_steps - 1, body_fun, params_)
        return unravel_func_p(v_)
    # diffusion_posterior_samples = jax.vmap(diffusion)(key_list)
    diffusion_posterior_samples = jax.lax.map(diffusion, key_list)
    return diffusion_posterior_samples


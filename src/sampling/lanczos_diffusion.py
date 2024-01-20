import jax
from jax import numpy as jnp
from src.helper import get_gvp_fun, get_ggn_vector_product
from src.sampling import lanczos_tridiag
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
    if diffusion_type == "non-kernel" or diffusion_type == "non-kernel-eigvals":
        eps = jax.random.normal(key, (n_samples, n_steps, rank))
    else:
        eps = jax.random.normal(key, (n_samples, n_steps, n_params))
    v0 = jnp.ones(n_params)
    def diffusion(single_eps_path):
        params_ = p0_flat
        def body_fun(n, res):
            # gvp = get_gvp_fun(model_fn, loss, unravel_func_p(res), x_train, y_train)
            gvp = get_ggn_vector_product(unravel_func_p(res), model, x_train, None, likelihood)
            if diffusion_type == "kernel":
                gvp_ = lambda v: gvp(v) + delta * v
                eigvals, eigvecs = lanczos_tridiag(gvp_, v0, rank - 1)
                lr_sample = 1/jnp.sqrt(alpha) * 1/delta * (gvp_(single_eps_path[n]) - eigvecs @ jnp.diag(eigvals) @ eigvecs.T @ single_eps_path[n])
            elif diffusion_type == "non-kernel":
                _, eigvecs = lanczos_tridiag(gvp, v0, rank - 1)
                lr_sample = 1/jnp.sqrt(alpha) * eigvecs @ single_eps_path[n]
            elif diffusion_type == "non-kernel-eigvals":
                eigvals, eigvecs = lanczos_tridiag(gvp, v0, rank - 1)
                lr_sample = eigvecs @ jnp.diag(1/jnp.sqrt(alpha + eigvals)) @ single_eps_path[n]
                #lr_sample = ((1/jnp.sqrt(alpha + eigvals).reshape(-1,1) * eigvecs) @ single_eps_path[n]
            elif diffusion_type == "full-ggn":
                gvp_ = lambda v: gvp(v) + alpha * v
                eigvals, eigvecs = lanczos_tridiag(gvp_, v0, rank - 1)
                lr_sample = 1/jnp.sqrt(alpha) * single_eps_path[n] + eigvecs @ jnp.diag(1/jnp.sqrt(eigvals + alpha) - 1/jnp.sqrt(alpha)) @ eigvecs.T @ single_eps_path[n]
            params_ = res + 1/jnp.sqrt(n_steps) * lr_sample
            return params_
        v_ = jax.lax.fori_loop(0, n_steps - 1, body_fun, params_)
        return unravel_func_p(v_)
    diffusion_posterior_samples = jax.vmap(diffusion)(eps)
    return diffusion_posterior_samples

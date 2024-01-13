from jax.experimental.ode import odeint
import jax
from jax import numpy as jnp
from src.helper import get_gvp_fun
from src.sampling import lanczos_tridiag
from jax import random
from typing import Literal
from functools import partial


partial(jax.jit, static_argnames=("loss", "model_fn", "rank", "integration_time", "n_evals"))
def ode_ggn(loss,
            model_fn,
            params,
            random_init_dir,
            v0,
            rank,
            n_evals,
            integration_time,
            x_train,
            y_train,
            rtol=1e-7,
            atol=1e-7,
            delta=1.0,
            integration_subspace: Literal["kernel", "non-kernel"] = "kernel"):
    p0_flat, unravel_func_p = jax.flatten_util.ravel_pytree(params)
    def ode_func(params_, t):
        gvp = get_gvp_fun(model_fn, loss, unravel_func_p(params_), x_train, y_train)
        if integration_subspace == "kernel":
            gvp_ = lambda v: gvp(v) + delta * v
            eigvals, eigvecs = lanczos_tridiag(gvp_, v0, rank - 1)
            rhs = 1/delta * (gvp_(random_init_dir) - eigvecs @ jnp.diag(eigvals) @ eigvecs.T @ random_init_dir)
        elif integration_subspace == "non-kernel":
            eigvals, eigvecs = lanczos_tridiag(gvp, v0, rank - 1)
            rhs = eigvecs @ eigvecs.T @ random_init_dir
        return rhs
    ode_y0 = p0_flat
    t = jnp.linspace(0., integration_time, n_evals)
    y_sols = odeint(ode_func, ode_y0, t, rtol=rtol, atol=atol)
    sols = jax.vmap(unravel_func_p)(y_sols)
    return sols, y_sols
    

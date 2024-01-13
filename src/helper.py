from functools import partial
import jax
from jax import numpy as jnp
from typing import Optional
from typing import Callable, Literal, Optional
from functools import partial

def get_gvp_fun(
    model_fn: Callable,
    loss_fn: Callable,
    params,
    x,
    y
  ) -> Callable:

  def gvp(eps):
    def scan_fun(carry, batch):
      x_, y_ = batch
      fn = lambda p: model_fn(p,x_[None,:])
      loss_fn_ = lambda preds: loss_fn(preds, y_)
      out, Je = jax.jvp(fn, (params,), (eps,))
      _, HJe = jax.jvp(jax.jacrev(loss_fn_, argnums=0), (out,), (Je,))
      _, vjp_fn = jax.vjp(fn, params)
      value = vjp_fn(HJe)[0]
      return jax.tree_map(lambda c, v: c + v, carry, value), None
    init_value = jax.tree_map(lambda x: jnp.zeros_like(x), params)
    return jax.lax.scan(scan_fun, init_value, (x, y))[0]
  p0_flat, unravel_func_p = jax.flatten_util.ravel_pytree(params)
  def matvec(v_like_params):
    p_unravelled = unravel_func_p(v_like_params)
    ggn_vp = gvp(p_unravelled)
    f_eval, _ = jax.flatten_util.ravel_pytree(ggn_vp)
    return f_eval
  return matvec

def compute_num_params(pytree):
    return sum(x.size if hasattr(x, "size") else 0 for x in jax.tree_util.tree_leaves(pytree))

def calculate_exact_ggn(loss_fn, model_fn, params, X, y, n_params):
    def body_fun(carry, a_tuple):
        x, y = a_tuple
        my_model_fn = partial(model_fn, x=x)  # model_fn wrt parameters
        my_loss_fn = partial(loss_fn, y=y)  # loss_fn wrt model output
        pred = my_model_fn(params)
        jacobian = jax.jacfwd(my_model_fn)(params)
        jacobian = jax.tree_map(lambda x: jnp.reshape(x, (x.shape[0], -1)), jacobian)
        jacobian = jnp.concatenate(jax.tree_util.tree_flatten(jacobian)[0], axis=-1)
        loss_hessian = jax.hessian(my_loss_fn)(pred)
        ggn = jacobian.T @ loss_hessian @ jacobian
        return jax.tree_map(lambda a, b: a + b, carry, ggn), None

    init_value = jnp.zeros((n_params, n_params))  # jacobian.T @ loss_hessian @ jacobian
    return jax.lax.scan(body_fun, init_value, (X, y))[0]

def random_split_like_tree(rng_key, target=None, treedef=None):
    # https://github.com/google/jax/discussions/9508
    if treedef is None:
        treedef = jax.tree_util.tree_structure(target)
    keys = jax.random.split(rng_key, treedef.num_leaves)
    return jax.tree_util.tree_unflatten(treedef, keys)


def tree_random_normal_like(rng_key, target, n_samples: Optional[int] = None):
    # https://github.com/google/jax/discussions/9508
    keys_tree = random_split_like_tree(rng_key, target)
    if n_samples is None:
        return jax.tree_util.tree_map(
            lambda l, k: jax.random.normal(k, l.shape, l.dtype),
            target,
            keys_tree,
        )
    else:
        return jax.tree_util.tree_map(
            lambda l, k: jax.random.normal(k, (n_samples,) + l.shape, l.dtype),
            target,
            keys_tree,
        )


def tree_random_uniform_like(rng_key, target, n_samples: Optional[int] = None, minval: int = 0, maxval: int = 1):
    keys_tree = random_split_like_tree(rng_key, target)
    if n_samples is None:
        return jax.tree_util.tree_map(
            lambda l, k: jax.random.uniform(k, l.shape, l.dtype, minval, maxval),
            target,
            keys_tree,
        )
    else:
        return jax.tree_util.tree_map(
            lambda l, k: jax.random.uniform(k, (n_samples,) + l.shape, l.dtype, minval, maxval),
            target,
            keys_tree,
        )



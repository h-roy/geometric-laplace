from functools import partial
import jax
from jax import numpy as jnp
from typing import Optional
from typing import Callable, Literal, Optional
from functools import partial
from src.losses import cross_entropy_loss, gaussian_log_lik_loss
import flax
import torch

def get_ggn_tree_product(
        params,
        model: flax.linen.Module,
        data_array: jax.Array = None,
        data_loader: torch.utils.data.DataLoader = None,
        likelihood_type: str = "regression"
    ):
    """
    takes as input a parameters pytree, a model and a dataset.
    returns a function v -> GGN * v, where v is a pytree "vector".
    Dataset can be given either ad an array or as a dataloader.
    """
    if data_array is not None:
        @jax.jit
        def ggn_tree_product(tree):
            model_on_data = lambda p: model.apply(p, data_array)
            _, J_tree = jax.jvp(model_on_data, (params,), (tree,))
            pred, model_on_data_vjp = jax.vjp(model_on_data, params)
            if likelihood_type == "regression":
                HJ_tree = J_tree
            elif likelihood_type == "classification":
                pred = jax.nn.softmax(pred, axis=1)
                pred = jax.lax.stop_gradient(pred)
                D = jax.vmap(jnp.diag)(pred)
                H = jnp.einsum("bo, bi->boi", pred, pred)
                H = D - H
                HJ_tree = jnp.einsum("boi, bi->bo", H, J_tree)
            else:
                raise ValueError(f"Likelihood {likelihood_type} not supported. Use either 'regression' or 'classification'.")
            JtHJ_tree = model_on_data_vjp(HJ_tree)[0]
            return JtHJ_tree
    else:
        assert data_loader is not None
        @jax.jit
        def ggn_tree_product_single_batch(tree, data_array):
            model_on_data = lambda p: model.apply(p, data_array)
            _, J_tree = jax.jvp(model_on_data, (params,), (tree,))
            pred, model_on_data_vjp = jax.vjp(model_on_data, params)
            if likelihood_type == "regression":
                HJ_tree = J_tree
            elif likelihood_type == "classification":
                pred = jax.nn.softmax(pred, axis=1)
                pred = jax.lax.stop_gradient(pred)
                D = jax.vmap(jnp.diag)(pred)
                H = jnp.einsum("bo, bi->boi", pred, pred)
                H = D - H
                HJ_tree = jnp.einsum("boi, bi->bo", H, J_tree)
            else:
                raise ValueError(f"Likelihood {likelihood_type} not supported. Use either 'regression' or 'classification'.")
            JtHJ_tree = model_on_data_vjp(HJ_tree)[0]
            return JtHJ_tree
        #@jax.jit
        def ggn_tree_product(tree):
            result = jax.tree_util.tree_map(lambda x : x*0, tree)
            for batch in data_loader:
                data_array = jnp.array(batch[0])
                JtHJ_tree = ggn_tree_product_single_batch(tree, data_array)
                result = jax.tree_util.tree_map(lambda a, b: a+b, JtHJ_tree, result)
            return result
    return ggn_tree_product

def get_ggn_vector_product(
        params,
        model: flax.linen.Module,
        data_array: jax.Array = None,
        data_loader: torch.utils.data.DataLoader = None,
        likelihood_type: str = "regression"
    ):
    """
    takes as input a parameters pytree, a model and a dataset.
    returns a function v -> GGN * v, where v is a jnp.array vector.
    Dataset can be given either ad an array or as a dataloader.
    """
    ggn_tree_product = get_ggn_tree_product(
        params, 
        model, 
        data_array,
        data_loader,
        likelihood_type=likelihood_type)
    devectorize_fun = jax.flatten_util.ravel_pytree(params)[1]
    
    def ggn_vector_product(v):
        tree = devectorize_fun(v)
        ggn_tree = ggn_tree_product(tree)
        ggn_v = jax.flatten_util.ravel_pytree(ggn_tree)[0]
        return jnp.array(ggn_v)
    
    if data_array is not None:
        return jax.jit(ggn_vector_product)
    else:
        return ggn_vector_product

# @partial(jax.jit, static_argnames=("model_fn", "loss_fn"))
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
      # Linearise and use transpose
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
#   return jax.jit(matvec)
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



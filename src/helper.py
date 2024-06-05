from functools import partial
import jax
from jax import numpy as jnp
from typing import Optional
from typing import Callable, Literal, Optional
from functools import partial
from jax.tree_util import Partial as jax_partial
from src.losses import cross_entropy_loss, gaussian_log_lik_loss
import flax
import torch
from jax._src.api import _jacfwd_unravel, _jvp, _std_basis
from jax._src.api_util import argnums_partial
from jax._src import linear_util as lu
from jax._src.util import wraps



def jacfwd_map(fun: Callable, argnums: int = 0) -> Callable:
    @wraps(fun, argnums=argnums)
    def jacfun(*args, **kwargs):
        f = lu.wrap_init(fun, kwargs)
        f_partial, dyn_args = argnums_partial(
            f, argnums, args, require_static_args_hashable=False
        )
        pushfwd: Callable = jax_partial(_jvp, f_partial, dyn_args)
        y, jac = jax.lax.map(pushfwd, _std_basis(dyn_args))
        example_args = dyn_args[0] if isinstance(argnums, int) else dyn_args
        jac_tree = jax.tree_map(
            jax_partial(_jacfwd_unravel, example_args), y, jac
        )
        return jac_tree

    return jacfun

def get_ggn_tree_product(
        params,
        model: flax.linen.Module,
        data_array: jax.Array = None,
        data_loader: torch.utils.data.DataLoader = None,
        likelihood_type: str = "regression",
        is_resnet: bool = False,
        batch_stats = None
    ):
    """
    takes as input a parameters pytree, a model and a dataset.
    returns a function v -> GGN * v, where v is a pytree "vector".
    Dataset can be given either ad an array or as a dataloader.
    """
    if data_array is not None:
        @jax.jit
        def ggn_tree_product(tree):
            if is_resnet:
                model_on_data = lambda p: model.apply({'params': p, 'batch_stats': batch_stats}, data_array, train=False, mutable=False)
            else:
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
        likelihood_type: str = "regression",
        is_resnet: bool = False,
        batch_stats = None
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
        likelihood_type=likelihood_type,
        is_resnet=is_resnet,
        batch_stats=batch_stats)
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
    
# @partial(jax.jit, static_argnames=("model", "likelihood_type"))

def get_gvp_fun(params,
                model: flax.linen.Module,
                data_array: jax.Array,
                batch_size = -1,
                likelihood_type: str = "regression",
                sum_type: Literal["running", "parallel"] = "running",
                is_resnet: bool = False,
                batch_stats = None,

  ) -> Callable:
  if sum_type == "running":
    def gvp(eps):
        def scan_fun(carry, batch):
            x_ = batch
            if batch_size>0:
                model_on_data = lambda p: model.apply(p,x_)
                if is_resnet:
                    model_on_data = lambda p: model.apply({'params': p, 'batch_stats': batch_stats}, x_, train=False, mutable=False)
            else:
                model_on_data = lambda p: model.apply(p,x_[None,:])
                if is_resnet:
                    model_on_data = lambda p: model.apply({'params': p, 'batch_stats': batch_stats}, x_[None,:], train=False, mutable=False)
            # Linearise and use transpose
            _, J_tree = jax.jvp(model_on_data, (params,), (eps,))
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
            return jax.tree_map(lambda c, v: c + v, carry, JtHJ_tree), None
        init_value = jax.tree_map(lambda x: jnp.zeros_like(x), params)
        # def true_fn(data_array):
        #     N = data_array.shape[0]//batch_size
        #     start_indices= (0,)*(len(data_array.shape))
        #     # x = data_array[: N * batch_size].reshape((N, batch_size)+ data_array.shape[1:])
        #     x = jax.lax.dynamic_slice(data_array, start_indices= (0,)*(len(data_array.shape)), slice_sizes=(N * batch_size,) + data_array.shape)
        #     return x
        # def false_fn(data_array):
        #     x = data_array
        #     return x
        # x = jax.lax.cond(batch_size>0, true_fn, false_fn, data_array)

        # Make this jitable
        # return jax.lax.scan(scan_fun, init_value, x)[0]
        return jax.lax.scan(scan_fun, init_value, data_array)[0]
    _, unravel_func_p = jax.flatten_util.ravel_pytree(params)

    def matvec(v_like_params):
        p_unravelled = unravel_func_p(v_like_params)
        ggn_vp = gvp(p_unravelled)
        f_eval, _ = jax.flatten_util.ravel_pytree(ggn_vp)
        return f_eval
    return matvec
    # return jax.jit(matvec)
  elif sum_type == "parallel":
    def gvp(eps):
        @jax.vmap
        def body_fn(batch):  
            x_ = batch
            if batch_size>0:
                model_on_data = lambda p: model.apply(p,x_)
            else:
                model_on_data = lambda p: model.apply(p,x_[None,:])
            # Linearise and use transpose
            _, J_tree = jax.jvp(model_on_data, (params,), (eps,))
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
        if batch_size>0:
            N = data_array.shape[0]//batch_size
            x = data_array[: N * batch_size].reshape((N, batch_size)+ data_array.shape[1:])
        else:
            x = data_array
        return jax.tree_map(lambda x: x.sum(axis=0), body_fn(x))
    _, unravel_func_p = jax.flatten_util.ravel_pytree(params)
    def matvec(v_like_params):
        p_unravelled = unravel_func_p(v_like_params)
        ggn_vp = gvp(p_unravelled)
        f_eval, _ = jax.flatten_util.ravel_pytree(ggn_vp)
        return f_eval
    # return jax.jit(matvec)
    return matvec


def get_gvp_new_fun(params,
                model_fn,
                data_array: jax.Array,
                batch_size = -1,
                likelihood_type: str = "regression",
                sum_type: Literal["running", "parallel"] = "running",
                v_in_type: Literal["vector", "tree"] = "vector"
  ) -> Callable:
  if sum_type == "running":
    def gvp(eps):
        def scan_fun(carry, batch):
            x_ = batch
            if batch.shape[0]>1:
                model_on_data = lambda p: model_fn(p,x_)
            else:
                model_on_data = lambda p: model_fn(p,x_[None,:])
            # Linearise and use transpose
            _, J_tree = jax.jvp(model_on_data, (params,), (eps,))
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
            return jax.tree_map(lambda c, v: c + v, carry, JtHJ_tree), None
        init_value = jax.tree_map(lambda x: jnp.zeros_like(x), params)
        return jax.lax.scan(scan_fun, init_value, data_array)[0]
    _, unravel_func_p = jax.flatten_util.ravel_pytree(params)
    def matvec(v_like_params):
        p_unravelled = unravel_func_p(v_like_params)
        ggn_vp = gvp(p_unravelled)
        f_eval, _ = jax.flatten_util.ravel_pytree(ggn_vp)
        return f_eval
    if v_in_type == "vector":
        return matvec
    elif v_in_type == "tree":
        return gvp
    # return jax.jit(matvec)
  elif sum_type == "parallel":
    def gvp(eps):
        @jax.vmap
        def body_fn(batch):  
            x_ = batch
            if batch_size>0:
                model_on_data = lambda p: model_fn(p,x_)
            else:
                model_on_data = lambda p: model_fn(p,x_[None,:])
            # Linearise and use transpose
            _, J_tree = jax.jvp(model_on_data, (params,), (eps,))
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
        return jax.tree_map(lambda x: x.sum(axis=0), body_fn(data_array))
    _, unravel_func_p = jax.flatten_util.ravel_pytree(params)
    def matvec(v_like_params):
        p_unravelled = unravel_func_p(v_like_params)
        ggn_vp = gvp(p_unravelled)
        f_eval, _ = jax.flatten_util.ravel_pytree(ggn_vp)
        return f_eval
    # return jax.jit(matvec)
    if v_in_type == "vector":
        return matvec
    elif v_in_type == "tree":
        return gvp



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



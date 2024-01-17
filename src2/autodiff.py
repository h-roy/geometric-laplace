import jax
import jax.numpy as jnp
import flax
import torch
from typing import Tuple
from training.loss import cross_entropy_loss, gaussian_log_lik_loss


#####################################
# Generalize Gauss Newtown products #

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



####################
# Hessian products #

def get_hessian_tree_product(
        params,
        model: flax.linen.Module,
        data_array: Tuple[jax.Array, jax.Array] = None,
        data_loader: torch.utils.data.DataLoader = None,
        likelihood_type: str = "regression"
    ):
    """
    takes as input a parameters pytree, a model and a dataset.
    returns a function v -> H * v, where v is a pytree "vector".
    Dataset can be given either ad an array or as a dataloader.
    """
    if data_array is not None:
        @jax.jit
        def hessian_tree_product(tree):
            if likelihood_type == "regression":
                negative_log_likelihood = gaussian_log_lik_loss
            elif likelihood_type == "classification":
                negative_log_likelihood = cross_entropy_loss
            else:
                raise ValueError(f"Likelihood {likelihood_type} not supported. Use either 'regression' or 'classification'.")
            X, Y = data_array
            loss_on_data = lambda p: negative_log_likelihood(model.apply(p, X), Y)
            return jax.jvp(jax.jacrev(loss_on_data), (params,), (tree,))[1]
    else:
        assert data_loader is not None
        @jax.jit
        def hessian_tree_product_single_batch(tree, data_array):
            if likelihood_type == "regression":
                negative_log_likelihood = gaussian_log_lik_loss
            elif likelihood_type == "classification":
                negative_log_likelihood = cross_entropy_loss
            else:
                raise ValueError(f"Likelihood {likelihood_type} not supported. Use either 'regression' or 'classification'.")
            X, Y = data_array
            loss_on_data = lambda p: negative_log_likelihood(model.apply(p, X), Y)
            return jax.jvp(jax.jacrev(loss_on_data), (params,), (tree,))[1]
        #@jax.jit
        def hessian_tree_product(tree):
            result = jax.tree_util.tree_map(lambda x : x*0, tree)
            for batch in data_loader:
                data_array = (jnp.array(batch[0]), jnp.array(batch[1])) 
                hessian_tree = hessian_tree_product_single_batch(tree, data_array)
                result = jax.tree_util.tree_map(lambda a, b: a+b, hessian_tree, result)
            return result
    return hessian_tree_product

def get_hessian_vector_product(
        params,
        model: flax.linen.Module,
        data_array: jax.Array = None,
        data_loader: torch.utils.data.DataLoader = None,
        likelihood_type: str = "regression"
    ):
    """
    takes as input a parameters pytree, a model and a dataset.
    returns a function v -> H * v, where v is a jnp.array vector.
    Dataset can be given either ad an array or as a dataloader.
    """
    hessian_tree_product = get_hessian_tree_product(
        params, 
        model, 
        data_array,
        data_loader,
        likelihood_type=likelihood_type)
    devectorize_fun = jax.flatten_util.ravel_pytree(params)[1]
    
    def ggn_vector_product(v):
        tree = devectorize_fun(v)
        hessian_tree = hessian_tree_product(tree)
        hessian_v = jax.flatten_util.ravel_pytree(hessian_tree)[0]
        return jnp.array(hessian_v)
    
    if data_array is not None:
        return jax.jit(ggn_vector_product)
    else:
        return ggn_vector_product
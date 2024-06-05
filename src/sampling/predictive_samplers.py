import jax
import jax.numpy as jnp
import tree_math as tm
from typing import Callable, Literal


def sample_predictive(
    posterior_samples,
    params,
    model_fn,
    x_test,
    linearised_laplace: bool = True,
    posterior_sample_type: Literal["Pytree", "List"] = "List",
):
    # Sample from the predictive
    if linearised_laplace:
        predictive_samples = sample_linearised_predictive(
            posterior_samples=posterior_samples,
            params_map=params,
            model_fn=model_fn,
            x_test=x_test,
            posterior_sample_type=posterior_sample_type,
        )
    else:
        predictive_samples = sample_laplace(
            posterior_samples=posterior_samples,
            model_fn=model_fn,
            x_test=x_test,
            posterior_sample_type=posterior_sample_type,
        )

    return predictive_samples


def linearized_predictive_posterior(x_test, params_sample, param_map, pred_map, model_fn):
    f_test = lambda p: model_fn(p, x_test)
    centered_sample = (tm.Vector(params_sample) - tm.Vector(param_map)).tree
    centered_pred = jax.jvp(f_test, (param_map,), (centered_sample,))[1]
    posterior_pred = centered_pred + pred_map
    return posterior_pred


def sample_linearised_predictive(
    posterior_samples,
    params_map,
    model_fn: Callable,
    x_test: jnp.ndarray,
    posterior_sample_type: Literal["Pytree", "List"],
):
    pred_map = model_fn(params_map, x_test)

    linearize = lambda p: linearized_predictive_posterior(x_test, p, params_map, pred_map, model_fn)
    if posterior_sample_type == "Pytree":
        posterior_predictive_samples = jax.vmap(linearize)(posterior_samples)
    elif posterior_sample_type == "List":
        posterior_predictive_list = []
        for sample in posterior_samples:
            posterior_predictive_list.append(linearize(sample))
        posterior_predictive_samples = jnp.stack(posterior_predictive_list)
    else:
        raise ValueError("posterior_sample_type must be either Pytree or List")

    return posterior_predictive_samples


def sample_laplace(
    posterior_samples, model_fn: Callable, x_test: jnp.ndarray, posterior_sample_type: Literal["Pytree", "List"]
):
    if posterior_sample_type == "List":
        posterior_predictive_list = []
        for sample in posterior_samples:
            posterior_predictive_list.append(model_fn(sample, x_test))
        posterior_predictive_samples = jnp.stack(posterior_predictive_list)
        posterior_predictive_samples = posterior_predictive_samples.squeeze()
    elif posterior_sample_type == "Pytree":
        pushforward = lambda p: model_fn(p, x_test)
        posterior_predictive_samples = jax.vmap(pushforward)(posterior_samples)
    return posterior_predictive_samples

def sample_hessian_predictive(posterior_samples,
                              model_fn,
                              params,
                              x_val
):
    def hessian_predictive(single_sample):
        f_test = lambda p: model_fn(p, x_val)
        delta = (tm.Vector(single_sample) - tm.Vector(params)).tree
        pred_map, lin_pred = jax.jvp(f_test, (params,), (delta,))
        _, hvp = jax.jvp(jax.jacrev(f_test), (params,), (delta,))
        hes_pred = jax.tree_map(lambda x: jnp.squeeze(x, 1), hvp)
        hes_pred = 0.5 * jax.vmap(lambda x: tm.Vector(delta) @ tm.Vector(x))(hes_pred).reshape(-1,1)
        return pred_map + lin_pred + hes_pred
    predictive = jax.vmap(hessian_predictive)(posterior_samples)
    return predictive


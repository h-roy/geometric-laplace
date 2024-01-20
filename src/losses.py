import jax
from jax import numpy as jnp
from functools import partial
from typing import Literal
import tree_math as tm

def sse_loss(preds, y):
    residual = preds - y
    return jnp.sum(residual**2)


def cross_entropy_loss(preds, y, rho=1.0):
    """
    preds: (n_samples, n_classes) (logits)
    y: (n_samples, n_classes) (one-hot labels)
    """
    preds = preds * rho
    preds = jax.nn.log_softmax(preds, axis=-1)
    return -jnp.sum(jnp.sum(preds * y, axis=-1))

def accuracy(params, model, batch_x, batch_y):
    preds = model.apply(params, batch_x)
    return jnp.sum(preds.argmax(axis=-1) == batch_y.argmax(axis=-1))

def accuracy_preds(preds, batch_y):
    return jnp.sum(preds.argmax(axis=-1) == batch_y.argmax(axis=-1))

def nll(preds, y):
    preds = jax.nn.log_softmax(preds, axis=-1)
    return (-jnp.sum(jnp.sum(preds * y, axis=-1), axis=-1)).mean()

def sse_loss(preds, y):
    residual = preds - y
    return jnp.sum(residual**2)

def gaussian_log_lik_loss(preds, y, rho=1.0):
    O = y.shape[-1]
    return 0.5 * O * jnp.log(2 * jnp.pi) - 0.5 * O * jnp.log(rho) + 0.5 * rho * sse_loss(preds, y)

@partial(jax.jit, static_argnames=['alpha', 'rho', 'model', 'D', 'N', 'likelihood', 'extra_stats'])
def log_posterior_loss(
    params,
    alpha,
    rho,
    model,
    x_batch,
    y_batch,
    D: int,
    N: int,
    likelihood: Literal["classification", "regression"] = "classification",
    extra_stats: bool = False,
):
    # define dict for logging purposes
    loss_dict = {}
    B = x_batch.shape[0]
    O = y_batch.shape[-1]
    vparams = tm.Vector(params)

    if likelihood == "regression":
        negative_log_likelihood = gaussian_log_lik_loss
    elif likelihood == "classification":
        negative_log_likelihood = cross_entropy_loss
    else:
        raise ValueError(f"Likelihood {likelihood} not supported. Use either 'regression' or 'classification'.")

    y_pred = model.apply(params, x_batch)

    neg_loglikelihood = negative_log_likelihood(y_pred, y_batch, rho)
    logprior = -D / 2 * jnp.log(2 * jnp.pi) - (1 / 2) * alpha * (vparams @ vparams) + D / 2 * jnp.log(alpha)
    logposterior = - neg_loglikelihood + logprior
    scaled_neg_logposterior = (1/B) * neg_loglikelihood - logprior # this is the loss

    loss_dict = {
        "log_likelihood": -neg_loglikelihood, 
        "log_prior": logprior, 
        "log_posterior": logposterior,
    }
    if extra_stats:
        loss_dict["sum_squared_error"] = sse_loss(y_pred, y_batch)
    return scaled_neg_logposterior, loss_dict
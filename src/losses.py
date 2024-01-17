import jax
from jax import numpy as jnp

def mse_loss(preds, y):
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


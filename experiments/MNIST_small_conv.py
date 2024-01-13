import pickle
import os
import argparse
import torch
from jax import random
import json
import datetime
from src.losses import cross_entropy_loss
from src.helper import calculate_exact_ggn, compute_num_params
from src.sampling.predictive_samplers import sample_predictive, sample_hessian_predictive
from jax import numpy as jnp
import jax
from jax import flatten_util
import matplotlib.pyplot as plt
from jax.lib import xla_bridge
from src.data.torch_datasets import MNIST, numpy_collate_fn
from src.models.convnet import ConvNet


if __name__ == "__main__":
    # Load Params Dict
    param_dict = pickle.load(open("./checkpoints/small_conv.pickle", "rb"))
    params = param_dict['params']
    alpha = param_dict['alpha']
    rho = param_dict['rho']
    n_params, train_samples, classes_train, n_classes, model = param_dict['train_stats']['n_params'], param_dict['train_stats']['train_samples'], param_dict['train_stats']['classes_train'], param_dict['train_stats']['n_classes'], param_dict['train_stats']['model']
    data_train = MNIST(path_root= "/work3/hroy/data/",
            train=True, n_samples=train_samples if train_samples > 0 else None, cls=classes_train
        )
    data_test = MNIST(path_root = "/work3/hroy/data/", train=False, cls=classes_train)

    if train_samples > 0:
        N = train_samples * n_classes
    else:
        N = len(data_train)
    N_test = len(data_test)

    #Posterior Data Processing

    sampling_train_loader = torch.utils.data.DataLoader(
    data_train, batch_size=N, shuffle=True, collate_fn=numpy_collate_fn, drop_last=True,
)
    data = next(iter(sampling_train_loader))
    x_train = jnp.array(data["image"])
    y_train = jnp.array(data["label"])
    sampling_val_loader = torch.utils.data.DataLoader(
        data_test, batch_size=N_test, shuffle=True, collate_fn=numpy_collate_fn, drop_last=True,
    )
    data = next(iter(sampling_val_loader))
    x_val = jnp.array(data["image"])
    y_val = jnp.array(data["label"])

    sample_key = jax.random.PRNGKey(0)
    n_posterior_samples = 200
    n_sample_batch_size = 1
    n_sample_batches = N // n_sample_batch_size


    # Create GGN
    _model_fn = lambda params, x: model.apply(params, x[None, ...])[0]
    ggn = calculate_exact_ggn(cross_entropy_loss, _model_fn, params, x_train, y_train, n_params)
    eigvals, eigvecs = jnp.linalg.eigh(ggn)
    alpha = 1.0
    rank = 100

    def ggn_lr_vp(v):
        return eigvecs[:,-rank:] @ jnp.diag(1/jnp.sqrt(eigvals[-rank:]+ alpha)) @ v

    def ggn_vp(v):
        return eigvecs @ jnp.diag(1/jnp.sqrt(eigvals + alpha)) @ v
    n_posterior_samples = 20
    D = compute_num_params(params)
    sample_key = jax.random.PRNGKey(0)
    eps = jax.random.normal(sample_key, (n_posterior_samples, D))
    p0_flat, unravel_func_p = flatten_util.ravel_pytree(params)
    var = 0.1
    def get_posteriors(single_eps):
        lr_sample = unravel_func_p(ggn_lr_vp(single_eps[:rank]) + p0_flat)
        posterior_sample = unravel_func_p(ggn_vp(single_eps) + p0_flat)
        isotropic_sample = unravel_func_p(var * single_eps + p0_flat)
        return lr_sample, posterior_sample, isotropic_sample
    lr_posterior_samples, posterior_samples, isotropic_posterior_samples = jax.vmap(get_posteriors)(eps)

    breakpoint()

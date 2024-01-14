import pickle
import os
import argparse
import torch
from jax import random
import json
import datetime
from src.losses import cross_entropy_loss, accuracy, accuracy_preds
from src.helper import calculate_exact_ggn, compute_num_params
from src.sampling.predictive_samplers import sample_predictive, sample_hessian_predictive
from jax import numpy as jnp
import jax
from jax import flatten_util
import matplotlib.pyplot as plt
from jax.lib import xla_bridge
from src.data.torch_datasets import MNIST, numpy_collate_fn
from src.sampling.exact_ggn import exact_ggn_laplace
from src.sampling.lanczos_diffusion import lanczos_diffusion



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
    n_posterior_samples = 500
    n_sample_batch_size = 1
    n_sample_batches = N // n_sample_batch_size


    # laplace
    # model_fn = lambda params, x: model.apply(params, x[None, ...])[0]
    # rank = 10
    # var = 0.5
    # jit_exact_ggn_laplace = jax.jit(exact_ggn_laplace, static_argnames=("loss", "model_fn", "n_params", "n_posterior_samples", "rank", "posterior_type"), backend='cpu')
    # lr_posterior_samples, posterior_samples, isotropic_posterior_samples = jit_exact_ggn_laplace(cross_entropy_loss, 
    #                                                                                          model_fn,
    #                                                                                          params,
    #                                                                                          x_train,
    #                                                                                          y_train,
    #                                                                                          n_params,
    #                                                                                          rank,
    #                                                                                          alpha,
    #                                                                                          n_posterior_samples,
    #                                                                                          sample_key,
    #                                                                                          var,
    #                                                                                          "all"
    #                                                                                          )
    # predictive_lr = sample_predictive(lr_posterior_samples, params, model, x_val, False, "Pytree")
    # predictive = sample_predictive(posterior_samples, params, model, x_val, False, "Pytree")
    # predictive_isotropic = sample_predictive(isotropic_posterior_samples, params, model, x_val, False, "Pytree")
    # print("MAP accuracy:", accuracy(params, model, x_val, y_val)/x_val.shape[0])
    # accuracies = jax.vmap(accuracy_preds, in_axes=(0,None))(predictive_lr, y_val)
    # accuracies /= x_val.shape[0]
    # print("lr:",jnp.mean(accuracies))
    # accuracies = jax.vmap(accuracy_preds, in_axes=(0,None))(predictive, y_val)
    # accuracies /= x_val.shape[0]
    # print("full:", jnp.mean(accuracies))
    # accuracies = jax.vmap(accuracy_preds, in_axes=(0,None))(predictive_isotropic, y_val)
    # accuracies /= x_val.shape[0]
    # print("isotropic:", jnp.mean(accuracies))

    n_steps = 20
    n_samples = 50
    alpha = 10.0
    rank = 50
    nonker_posterior_samples = lanczos_diffusion(cross_entropy_loss, model.apply, params,n_steps,n_samples,alpha,sample_key,n_params,rank,x_train,y_train,1.0,"non-kernel-eigvals")
    predictive_lr = sample_predictive(nonker_posterior_samples, params, model, x_val, False, "Pytree")
    print("MAP accuracy:", accuracy(params, model, x_val, y_val)/x_val.shape[0])
    accuracies = jax.vmap(accuracy_preds, in_axes=(0,None))(predictive_lr, y_val)
    accuracies /= x_val.shape[0]
    print("lr:",jnp.mean(accuracies))

    breakpoint()

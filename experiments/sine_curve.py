import pickle
import os
import argparse
import torch
from jax import random
import json
import datetime
from src.losses import mse_loss
from src.helper import calculate_exact_ggn
from src.sampling.predictive_samplers import sample_predictive, sample_hessian_predictive
from jax import numpy as jnp
import jax
from jax import flatten_util
import matplotlib.pyplot as plt
from jax.lib import xla_bridge



def f(x):
    return jnp.sin(5 * x + 1)

if __name__ == "__main__":
    param_dict = pickle.load(open("./checkpoints/syntetic_regression.pickle", "rb"))
    params = param_dict['params']
    alpha = param_dict['alpha']
    rho = param_dict['rho']
    x_train, y_train, x_val, y_val, model, D = param_dict["train_stats"]['x_train'],param_dict["train_stats"]['y_train'],param_dict["train_stats"]['x_val'],param_dict["train_stats"]['y_val'],param_dict["train_stats"]['model'], param_dict["train_stats"]['n_params']
    _model_fn = lambda params, x: model.apply(params, x[None, ...])[0]
    # ggn = calculate_exact_ggn(mse_loss, _model_fn, params, x_train, y_train, D)
    sample_key = jax.random.PRNGKey(0)
    n_steps = 10
    n_samples = 50
    alpha = 10.0
    rank = 7
    eps = jax.random.normal(sample_key, (n_samples, n_steps, rank))
    p0_flat, unravel_func_p = jax.flatten_util.ravel_pytree(params)
    # rank = 100
    # alpha = 0.1
    @jax.jit
    def rw_nonker(single_eps_path):
        params_ = p0_flat
        posterior_list = [params]
        for i in range(n_steps):
            ggn = calculate_exact_ggn(mse_loss, _model_fn, unravel_func_p(params_), x_train, y_train, D)
            _, eigvecs = jnp.linalg.eigh(ggn)
            # lr_sample = eigvecs[:,-rank:] @ jnp.diag(1/jnp.sqrt(eigvals[-rank:]+ alpha)) @ single_eps_path[i]
            lr_sample = 1/jnp.sqrt(alpha) * eigvecs[:,-rank:] @ single_eps_path[i]
            params_ = params_ + 1/jnp.sqrt(n_steps) * lr_sample
            posterior_list.append(unravel_func_p(params_))
            print("NonKernel Path finished")
        return posterior_list
        # return unravel_func_p(params_)
    nonker_posterior_samples = jax.vmap(rw_nonker)(eps)[-1]

    n_steps = 10
    n_samples = 50
    rank = 100#200
    alpha = 10.0
    eps = jax.random.normal(sample_key, (n_samples, n_steps, D - rank))
    p0_flat, unravel_func_p = jax.flatten_util.ravel_pytree(params)
    @jax.jit
    def rw_ker(single_eps_path):
        params_ = p0_flat
        posterior_list = [params]
        for i in range(n_steps):
            ggn = calculate_exact_ggn(mse_loss, _model_fn, unravel_func_p(params_), x_train, y_train, D)
            _, eigvecs = jnp.linalg.eigh(ggn)
            lr_sample = 1/jnp.sqrt(alpha) * eigvecs[:,:-rank] @ single_eps_path[i]
            params_ = params_ + 1/jnp.sqrt(n_steps) * lr_sample
            posterior_list.append(unravel_func_p(params_))
            print("Kernel Path finished")
        return posterior_list
    ker_posterior_samples = jax.vmap(rw_ker)(eps)[-1]

    x_val = x_train
    ker_predictive = sample_predictive(ker_posterior_samples, params, model, x_val, False, "Pytree")
    nonker_predictive = sample_predictive(nonker_posterior_samples, params, model, x_val, False, "Pytree")
    ker_posterior_predictive_mean = jnp.mean(ker_predictive, axis=0).squeeze()
    ker_posterior_predictive_std = jnp.std(ker_predictive, axis=0).squeeze()
    nonker_posterior_predictive_mean = jnp.mean(nonker_predictive, axis=0).squeeze()
    nonker_posterior_predictive_std = jnp.std(nonker_predictive, axis=0).squeeze()
    preds = model.apply(params, x_val)


    x_sorted, idx = jnp.sort(x_val[:, 0]), jnp.argsort(x_val[:, 0])
    preds_sorted = preds[idx][:, 0]
    ker_means_sorted = ker_posterior_predictive_mean[idx]
    nonker_means_sorted = nonker_posterior_predictive_mean[idx]
    ker_samples_sorted = ker_predictive[:, idx, :].squeeze()
    nonker_samples_sorted = nonker_predictive[:, idx, :].squeeze()
    ker_std_sorted = ker_posterior_predictive_std[idx]
    nonker_std_sorted = nonker_posterior_predictive_std[idx]

    print(xla_bridge.get_backend().platform)


    #Plots
    # Plot 1: Mean and Std of Sampled Predictive
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(x_sorted, preds_sorted, label="Predictions", marker="None")
    line = ax.plot(x_sorted, ker_means_sorted, label="Ker Samples", marker="None")
    ax.fill_between(
        x_sorted, ker_means_sorted - ker_std_sorted, ker_means_sorted + ker_std_sorted, alpha=0.1, color=line[0].get_color()
    )
    line = ax.plot(x_sorted, nonker_means_sorted, label="Nonker Samples", marker="None")
    ax.fill_between(
        x_sorted, nonker_means_sorted - nonker_std_sorted, nonker_means_sorted + nonker_std_sorted, alpha=0.1, color=line[0].get_color()
    )

    line = ax.plot(x_sorted, f(x_sorted), label="Targets", marker="None")
    # plt.plot(
    #     jnp.array(x_train[:, 0]), jnp.array(y_train[:, 0]), color=line[0].get_color(), linestyle="None", marker="o"
    # )
    ax.legend(fontsize="x-small", loc="upper right")
    fig.savefig("figures/predictive_posterior.png")
    #Plot 2
    fig, ax = plt.subplots(ncols=2, figsize=(15, 10))
    ax[0].plot(x_sorted, ker_samples_sorted.transpose(), marker="None", linestyle="-", alpha=0.2, color="slategrey")
    ax[0].plot(jnp.array(x_train[:, 0]), jnp.array(y_train[:, 0]), label="Targets", linestyle="None", marker="o")
    ax[0].set_title("Sampled Functions in Kernel Directions")
    ax[1].plot(x_sorted, nonker_samples_sorted.transpose(), marker="None", linestyle="-", alpha=0.2, color="slategrey")
    ax[1].plot(jnp.array(x_train[:, 0]), jnp.array(y_train[:, 0]), label="Targets", linestyle="None", marker="o")
    ax[1].set_title("Sampled Functions in Non-Kernel Directions")
    fig.savefig("figures/sampled_fn.png")





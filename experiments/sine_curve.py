import pickle
import os
import argparse
import torch
from jax import random
import json
import datetime
from src.losses import sse_loss
from src.helper import calculate_exact_ggn
from src.sampling.predictive_samplers import sample_predictive, sample_hessian_predictive
from jax import numpy as jnp
import jax
from jax import flatten_util
import matplotlib.pyplot as plt
from jax.lib import xla_bridge
from src.sampling.lanczos_diffusion import lanczos_diffusion



def f(x):
    return jnp.sin(5 * x + 1)

if __name__ == "__main__":
    param_dict = pickle.load(open("./checkpoints/syntetic_regression.pickle", "rb"))
    params = param_dict['params']
    alpha = param_dict['alpha']
    rho = param_dict['rho']
    x_train, y_train, x_val, y_val, model, D = param_dict["train_stats"]['x_train'],param_dict["train_stats"]['y_train'],param_dict["train_stats"]['x_val'],param_dict["train_stats"]['y_val'],param_dict["train_stats"]['model'], param_dict["train_stats"]['n_params']
    sample_key = jax.random.PRNGKey(0)
    n_steps = 1000
    n_samples = 50
    alpha = 10.0
    rank = 7
    # nonker_posterior_samples = lanczos_diffusion(sse_loss, model.apply, params, n_steps, n_samples, alpha, sample_key, D, rank, x_train, y_train, 1.0, "non-kernel-eigvals")
    nonker_posterior_samples = lanczos_diffusion(model, params, n_steps, n_samples, alpha, sample_key, D, rank, x_train, "regression", 1.0, "non-kernel-eigvals")
    

    n_steps = 1000
    n_samples = 50
    rank = 10#200
    alpha = 10.0
    # ker_posterior_samples = lanczos_diffusion(sse_loss, model.apply, params, n_steps, n_samples, alpha, sample_key, D, rank, x_train, y_train, 1.0, "kernel")
    ker_posterior_samples = lanczos_diffusion(model, params, n_steps, n_samples, alpha, sample_key, D, rank, x_train, "regression", 1.0, "kernel")

    # x_val = x_train
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





import time
import argparse
import jax
import matplotlib.pyplot as plt
import optax
import matfree
import tree_math as tm
from flax import linen as nn
from jax import nn as jnn
from jax import numpy as jnp
from jax import random, jit
import pickle
from src.models import LeNet
from src.losses import cross_entropy_loss, accuracy_preds, nll
from src.helper import calculate_exact_ggn, tree_random_normal_like, compute_num_params
from src.sampling.predictive_samplers import sample_predictive, sample_hessian_predictive
from src.sampling.lanczos_diffusion import lanczos_diffusion
from jax import flatten_util
import matplotlib.pyplot as plt
from src.data.datasets import get_rotated_mnist_loaders
import torch
from src.data import MNIST, n_classes
from src.ood_functions.evaluate import evaluate
from src.ood_functions.metrics import compute_metrics


if __name__ == "__main__":
    param_dict = pickle.load(open("./checkpoints/MNIST/LeNet/OOD_MNIST_seed420_params.pickle", "rb"))
    params, alpha, rho, model_id = param_dict['params'], param_dict['prior_precision'], param_dict['likelihood_precision'], param_dict['model']
    if model_id == 'LeNet':
        model = LeNet(output_dim=10, activation="tanh")
    cls = list(range(n_classes("MNIST")))
    dataset = MNIST(path_root="/work3/hroy/data/", train=True, n_samples_per_class=None, download=True, cls=cls, seed=420)
    x_train = jnp.array([data[0] for data in dataset])
    # y_train = jnp.array([data[1] for data in dataset])
    eval_args = {}
    eval_args["linearised_laplace"] = False
    eval_args["posterior_sample_type"] = "Pytree"
    eval_args["likelihood"] = "classification"
    n_steps = 2
    n_samples = 50
    alpha = 1.0
    rank = 1000
    n_params = compute_num_params(params)
    sample_key = jax.random.PRNGKey(420)
    start_time = time.time()
    nonker_posterior_samples = lanczos_diffusion(model, 
                                                 params,
                                                 n_steps,
                                                 n_samples,
                                                 alpha,
                                                 sample_key,
                                                 n_params,
                                                 rank,
                                                 x_train,
                                                 "classification",
                                                 1.0,
                                                 "non-kernel-eigvals")
    print(f"Lanczos diffusion (for a {n_params} parameter model with {n_steps - 1} steps, {n_samples} samples and {rank} iterations) took {time.time()-start_time:.5f} seconds")
    breakpoint()
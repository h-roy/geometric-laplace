import time
import argparse
import jax
import matplotlib.pyplot as plt
import datetime
import tree_math as tm
from flax import linen as nn
from jax import nn as jnn
from jax import numpy as jnp
import json
from jax import random, jit
import pickle
from src.models import LeNet, MLP
from src.losses import cross_entropy_loss, accuracy_preds, nll
from src.helper import calculate_exact_ggn, tree_random_normal_like, compute_num_params
from src.sampling.predictive_samplers import sample_predictive, sample_hessian_predictive
from src.sampling.lanczos_diffusion import lanczos_diffusion
from jax import flatten_util
import matplotlib.pyplot as plt
from src.data.datasets import get_rotated_mnist_loaders
from src.data import MNIST, FashionMNIST, CIFAR10, CIFAR100, SVHN, n_classes
from src.ood_functions.evaluate import evaluate
from src.ood_functions.metrics import compute_metrics

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", type=str, default="./checkpoints/MNIST/LeNet/OOD_MNIST_seed420", help="path of model")
parser.add_argument("--run_name", default=None, help="Fix the save file name.")
parser.add_argument("--diffusion_steps", type=int, default=20)
parser.add_argument("--num_samples", type=int, default=5)
parser.add_argument("--lanczos_iters", type=int, default=1000)
parser.add_argument("--sample_seed",  type=int, default=0)
parser.add_argument("--posthoc_precision",  type=float, default=1.0)


if __name__ == "__main__":
    now = datetime.datetime.now()
    now_string = now.strftime("%Y-%m-%d-%H-%M-%S")

    args = parser.parse_args()

    param_dict = pickle.load(open(f"{args.checkpoint_path}_params.pickle", "rb"))
    with open(f"{args.checkpoint_path}_args.json", 'r') as JSON: 
        train_args_dict = json.load(JSON)

    ###############
    ### dataset ###
    n_samples_per_class = None if train_args_dict['n_samples'] is None else int(train_args_dict['n_samples']/n_classes(train_args_dict['dataset']))
    cls=list(range(n_classes(train_args_dict['dataset'])))
    if train_args_dict['dataset'] == "MNIST":
        dataset = MNIST(path_root=train_args_dict['data_path'], train=True, n_samples_per_class=n_samples_per_class, download=True, cls=cls, seed=train_args_dict['seed'])
    elif train_args_dict['dataset'] == "FMNIST":
        dataset = FashionMNIST(path_root=train_args_dict['data_path'], train=True, n_samples_per_class=n_samples_per_class, download=True, cls=cls, seed=train_args_dict['seed'])
    elif train_args_dict['dataset'] == "CIFAR-10":
        dataset = CIFAR10(path_root=train_args_dict['data_path'], train=True, n_samples_per_class=n_samples_per_class, download=True, cls=cls, seed=train_args_dict['seed'])
    elif train_args_dict['dataset'] == "CIFAR-100":
        dataset = CIFAR100(path_root=train_args_dict['data_path'], train=True, n_samples_per_class=n_samples_per_class, download=True, cls=cls, seed=train_args_dict['seed'])
    elif train_args_dict['dataset'] == "SVHN":
        dataset = SVHN(path_root=train_args_dict['data_path'], train=True, n_samples_per_class=n_samples_per_class, download=True, cls=cls, seed=train_args_dict['seed'])
    dataset_size = len(dataset)

    #############
    ### model ###
    output_dim = n_classes(train_args_dict['dataset'])
    if train_args_dict['model'] == "MLP":
        model = MLP(output_dim=output_dim, num_layers=train_args_dict['mlp_num_layers'], hidden_dim=train_args_dict['mlp_hidden_dim'], activation=train_args_dict['activation_fun'])
    elif train_args_dict['model'] == "LeNet":
        model = LeNet(output_dim=output_dim, activation=train_args_dict['activation_fun'])
    else:
        raise ValueError(f"Model {train_args_dict['model']} is not implemented")

    params, alpha, rho, model_id = param_dict['params'], param_dict['prior_precision'], param_dict['likelihood_precision'], param_dict['model']
    x_train = jnp.array([data[0] for data in dataset])
    # y_train = jnp.array([data[1] for data in dataset])
    n_steps = args.diffusion_steps
    n_samples = args.num_samples
    rank = args.lanczos_iters
    n_params = compute_num_params(params)
    alpha = args.posthoc_precision
    sample_key = jax.random.PRNGKey(args.sample_seed)
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

    posterior_samples = lanczos_diffusion(model, 
                                          params,
                                          2,
                                          n_samples,
                                          alpha,
                                          sample_key,
                                          n_params,
                                          rank,
                                          x_train,
                                          "classification",
                                          1.0,
                                          "full-ggn")
    
    posterior_dict = {
        "Non-ker-eigvals": nonker_posterior_samples,
        "full-samples": posterior_samples
    }
    if args.run_name is not None:
        save_name = f"{args.run_name}_seed{args.sample_seed}"
    else:
        save_name = f"started_{now_string}"

    save_path = f"./checkpoints/{train_args_dict['dataset']}/posterior_samples_{save_name}"
    pickle.dump(posterior_dict, open(f"{save_path}_params.pickle", "wb"))

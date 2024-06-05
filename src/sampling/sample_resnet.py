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
from src.models import ResNet, ResNetBlock
from src.losses import cross_entropy_loss, accuracy_preds, nll
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
parser.add_argument("--checkpoint_path", type=str, default="./checkpoints/CIFAR-10/ResNet/epoch200_seed0_params.pickle", help="path of model")
parser.add_argument("--run_name", default=None, help="Fix the save file name.")
parser.add_argument("--diffusion_steps", type=int, default=20)
parser.add_argument("--num_samples", type=int, default=5)
parser.add_argument("--lanczos_iters", type=int, default=1000)
parser.add_argument("--sample_seed",  type=int, default=0)
parser.add_argument("--posthoc_precision",  type=float, default=1.0)
parser.add_argument("--gvp_batch_size",  type=int, default=5000)
parser.add_argument("--posterior_type",  type=str, choices=["non-kernel-eigvals", "full-ggn"], default="non-kernel-eigvals")



if __name__ == "__main__":
    now = datetime.datetime.now()
    now_string = now.strftime("%Y-%m-%d-%H-%M-%S")

    args = parser.parse_args()

    param_dict = pickle.load(open(f"{args.checkpoint_path}", "rb"))
    params = param_dict['params']
    batch_stats = param_dict['batch_stats']


    ###############
    ### dataset ###
    n_samples_per_class = None 
    cls=list(range(n_classes("CIFAR-10")))
    dataset = CIFAR10(path_root='/dtu/p1/hroy/data', train=True, n_samples_per_class=n_samples_per_class, download=True, cls=cls, seed=0)
    dataset_size = len(dataset)

    #############
    ### model ###
    output_dim = 10
    model = ResNet(
        num_classes = output_dim,
        c_hidden =(16, 32, 64),
        num_blocks = (3, 3, 3),
        act_fn = nn.relu,
        block_class = ResNetBlock #PreActResNetBlock #
    )
    x_train = jnp.array([data[0] for data in dataset])
    # Only for rebuttal experiments:
    x_train = jnp.moveaxis(x_train, 2, 3)
    ###############
    n_steps = args.diffusion_steps
    n_samples = args.num_samples
    rank = args.lanczos_iters
    n_params = compute_num_params(params)
    alpha = args.posthoc_precision
    gvp_type = "batch-sum"
    gvp_bs = args.gvp_batch_size
    posterior_type = args.posterior_type
    N = x_train.shape[0]//gvp_bs
    data_array = x_train[: N * gvp_bs].reshape((N, gvp_bs)+ x_train.shape[1:])
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
                                                 data_array,
                                                 "classification",
                                                 1.0,
                                                 posterior_type,
                                                 gvp_type,
                                                 gvp_bs,
                                                 is_resnet=True,
                                                 batch_stats=batch_stats
                                                 )
    print(f"Lanczos diffusion (for a {n_params} parameter model with {n_steps - 1} steps, {n_samples} samples and {rank} iterations) took {time.time()-start_time:.5f} seconds")    
    posterior_dict = {
        "posterior_samples": nonker_posterior_samples,
    }
    model_on_data = lambda p: model.apply({'params': p, 'batch_stats': batch_stats}, x_train[:500], train=False, mutable=False)
    sample_preds = jax.vmap(model_on_data)(nonker_posterior_samples)
    sample_preds = sample_preds.squeeze()
    y_train = jnp.array([data[1] for data in dataset])[:500]
    accuracy = accuracy_preds(sample_preds, y_train)/500 * 100.
    preds = model_on_data(params)
    map_accuracy = accuracy_preds(preds, y_train)/500 * 100.
    print("map Accuracy: ", map_accuracy)
    print("sample Accuracy: ", accuracy)
    if args.run_name is not None:
        save_name = f"{args.run_name}_seed{args.sample_seed}_type_{args.posterior_type}_iter{args.lanczos_iters}_steps{args.diffusion_steps}_samples{args.num_samples}_prec{args.posthoc_precision}"
    else:
        save_name = f"started_{now_string}"

    save_path = f"./checkpoints/CIFAR-10/posterior_samples_{save_name}"
    pickle.dump(posterior_dict, open(f"{save_path}_params.pickle", "wb"))

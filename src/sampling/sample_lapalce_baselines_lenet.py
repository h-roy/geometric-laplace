import json
import time
import argparse
import jax
import datetime
from flax import linen as nn
from jax import numpy as jnp
import pickle
from src.models import LeNet, MLP
from src.losses import cross_entropy_loss, accuracy_preds, nll
from src.helper import compute_num_params
from src.sampling.laplace_baselines import hutchinson_diagonal_laplace, exact_diagonal_laplace, last_layer_lapalce
from src.sampling.swag import swag_score_fun
import matplotlib.pyplot as plt
from src.data import MNIST, FashionMNIST
from src.data.utils import numpy_collate_fn
from src.data import CIFAR10, n_classes
from torch.utils import data

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", type=str, default="./checkpoints/MNIST/LeNet/OOD_MNIST_seed420", help="path of model")
parser.add_argument("--method", type=str, choices=["Subnetwork", "Hutchinson_Diag", "Exact_Diag", "SWAG"], default="Subnetwork", help="Method to use for sampling")
parser.add_argument("--run_name", default=None, help="Fix the save file name.")
parser.add_argument("--diffusion_steps", type=int, default=20)
parser.add_argument("--num_samples", type=int, default=5)
parser.add_argument("--sample_seed",  type=int, default=0)
parser.add_argument("--posthoc_precision",  type=float, default=1.0)
parser.add_argument("--batch_size",  type=int, default=100)

parser.add_argument("--num_ll_params",  type=int, default=1000)
parser.add_argument("--hutchinson_samples",  type=int, default=100)
parser.add_argument("--hutchinson_levels",  type=int, default=3)
parser.add_argument("--gvp_batch_size",  type=int, default=50)



if __name__ == "__main__":
    now = datetime.datetime.now()
    now_string = now.strftime("%Y-%m-%d-%H-%M-%S")

    args = parser.parse_args()

    param_dict = pickle.load(open(f"{args.checkpoint_path}_params.pickle", "rb"))
    with open(f"{args.checkpoint_path}_args.json", 'r') as JSON: 
        train_args_dict = json.load(JSON)

    params = param_dict['params']
    method = args.method
    ###############
    ### dataset ###
    n_samples_per_class = None 
    cls=list(range(n_classes("CIFAR-10")))
    if train_args_dict['dataset'] == "MNIST":
        dataset = MNIST(path_root='/dtu/p1/hroy/data', train=True, n_samples_per_class=n_samples_per_class, download=True, cls=cls, seed=train_args_dict['seed'])
    elif train_args_dict['dataset'] == "FMNIST":
        dataset = FashionMNIST(path_root='/dtu/p1/hroy/data', train=True, n_samples_per_class=n_samples_per_class, download=True, cls=cls, seed=train_args_dict['seed'])
    dataset_size = len(dataset)

    #############
    ### model ###
    output_dim = 10
    model = LeNet(output_dim=output_dim, activation=train_args_dict['activation_fun'])    
    x_train = jnp.array([data[0] for data in dataset])
    ###############
    n_steps = args.diffusion_steps
    n_samples = args.num_samples
    n_params = compute_num_params(params)
    alpha = args.posthoc_precision
    batch_size = args.batch_size

    train_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=4, collate_fn=numpy_collate_fn)
    sample_key = jax.random.PRNGKey(args.sample_seed)
    start_time = time.time()
    model_fn = model.apply

    leafs, _ = jax.tree_util.tree_flatten(params)
    num_ll_params = len(leafs[-1].flatten()) + len(leafs[-2])
    likelihood = "classification"

    hutchinson_samples = args.hutchinson_samples
    num_levels = args.hutchinson_levels
    gvp_batch_size = args.gvp_batch_size

    if method == "Exact_Diag":
        nonker_posterior_samples, metrics = exact_diagonal_laplace(model_fn,
                                                            params,
                                                            n_samples,
                                                            alpha,
                                                            train_loader,
                                                            sample_key,
                                                            output_dim,
                                                            likelihood,)
    elif method == "Hutchinson_Diag":
        nonker_posterior_samples, metrics = hutchinson_diagonal_laplace(model_fn, 
                                                        params, 
                                                        n_samples,
                                                        alpha,
                                                        gvp_batch_size,
                                                        train_loader,
                                                        sample_key,
                                                        num_levels,
                                                        hutchinson_samples,
                                                        likelihood,
                                                        "parallel")
    elif method == "SWAG":
        nonker_posterior_samples = swag_score_fun(model, param_dict, sample_key, n_samples, train_loader, likelihood)
        metrics = None
    elif method == "Subnetwork":
        nonker_posterior_samples, metrics = last_layer_lapalce(
                                        model_fn,
                                        params,
                                        alpha,
                                        sample_key,
                                        num_ll_params,
                                        n_samples,
                                        train_loader,
                                        "classification",
                                        )
    print(f"{method} for a {n_params} parameter model {n_samples} samples took {time.time()-start_time:.5f} seconds") 
    breakpoint()   
    posterior_dict = {
        "posterior_samples": nonker_posterior_samples,
    }
    if args.run_name is not None:
        save_name = f"{args.run_name}_seed_{args.sample_seed}_prec_{args.posthoc_precision}"
    else:
        save_name = f"started_{now_string}"

    save_path = f"./checkpoints/{train_args_dict['dataset']}/baselines/{method}_{save_name}"
    pickle.dump(posterior_dict, open(f"{save_path}_params.pickle", "wb"))

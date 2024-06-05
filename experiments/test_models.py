import pickle
from src.losses import cross_entropy_loss, accuracy_preds, nll
from src.helper import calculate_exact_ggn, tree_random_normal_like, compute_num_params
from src.sampling.predictive_samplers import sample_predictive, sample_hessian_predictive
from src.models import ResNet, ResNetBlock, PreActResNetBlock
from jax import flatten_util
import matplotlib.pyplot as plt
from flax import linen as nn

from src.data import CIFAR10, n_classes
import torch
from src.data.torch_datasets import MNIST, numpy_collate_fn
import jax.numpy as jnp
from src.data.datasets import get_rotated_cifar_loaders, get_cifar10_ood_loaders, load_corrupted_cifar10_per_type, load_corrupted_cifar10
from src.ood_functions.evaluate import evaluate, evaluate_map, evaluate_samples
from src.ood_functions.metrics import compute_metrics


if __name__=="__main__":
    param_list = []
    param_list.append(pickle.load(open("./checkpoints/CIFAR-10/ResNet/good_params_seed0.pickle", "rb")))
    param_list.append(pickle.load(open("./checkpoints/CIFAR-10/ResNet/good_params_seed1.pickle", "rb")))
    param_list.append(pickle.load(open("./checkpoints/CIFAR-10/ResNet/good_params_seed2.pickle", "rb")))
    param_list.append(pickle.load(open("./checkpoints/CIFAR-10/ResNet/good_params_seed3.pickle", "rb")))
    param_list.append(pickle.load(open("./checkpoints/CIFAR-10/ResNet/good_params_seed4.pickle", "rb")))

    output_dim = 10
    model = ResNet(
            num_classes = output_dim,
            c_hidden =(16, 32, 64),
            num_blocks = (3, 3, 3),
            act_fn = nn.relu,
            block_class = ResNetBlock #PreActResNetBlock #
        )
    n_samples_per_class = None
    cls=list(range(n_classes("CIFAR-10")))
    dataset = CIFAR10(path_root='/dtu/p1/hroy/data', train=True, n_samples_per_class=n_samples_per_class, download=True, cls=cls, seed=0)
    data_array = jnp.array([data[0] for data in dataset])

    x_train = data_array[:500]
    y_train = jnp.array([data[1] for data in dataset])[:500]
    for seed, param_dict in enumerate(param_list):
        params = param_dict['params']
        batch_stats = param_dict['batch_stats']
        map_preds = model.apply({'params': params, 'batch_stats': batch_stats},
                                        x_train,
                                        train=False,
                                        mutable=False)
        print(f"Accuracy for seed {seed}:",accuracy_preds(map_preds, y_train)/y_train.shape[0] * 100)

    

 
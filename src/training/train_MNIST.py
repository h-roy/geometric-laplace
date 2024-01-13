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
from src.losses import cross_entropy_loss, accuracy
from src.helper import compute_num_params
from src.sampling.predictive_samplers import sample_predictive, sample_hessian_predictive
from jax import flatten_util
import matplotlib.pyplot as plt
import torch
from src.data.torch_datasets import MNIST, numpy_collate_fn
from src.models.convnet import ConvNet

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_samples", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--n_classes", type=int, default=10)
    args = parser.parse_args()
    print(args)

    train_samples = 10#1000
    classes_train = [0,1,2,3,4,5,6,7,8,9]
    n_classes = 10
    batch_size = 20#256
    test_batch_size = 256

    data_train = MNIST(path_root= "/work3/hroy/data/",
                train=True, n_samples=train_samples if train_samples > 0 else None, cls=classes_train
            )
    data_test = MNIST(path_root = "/work3/hroy/data/", train=False, cls=classes_train)

    if train_samples > 0:
        N = train_samples * n_classes
    else:
        N = len(data_train)
    N_test = len(data_test)
    if test_batch_size > 0:
        test_batch_size = test_batch_size
    else:
        test_batch_size = len(data_test)

    n_test_batches = int(N_test / test_batch_size)
    n_batches = int(N / batch_size)

    train_loader = torch.utils.data.DataLoader(
        data_train, batch_size=batch_size, shuffle=True, collate_fn=numpy_collate_fn, drop_last=True,
    )

    valid_loader = torch.utils.data.DataLoader(
        data_test, batch_size=test_batch_size, shuffle=True, collate_fn=numpy_collate_fn, drop_last=True,
    )

    model = ConvNet()
    batch = next(iter(train_loader))
    x_init, y_init = batch["image"], batch["label"]
    output_dim = y_init.shape[-1]
    key, split_key = random.split(jax.random.PRNGKey(0))
    params = model.init(key, x_init)
    alpha = 1.
    optim = optax.chain(
            optax.clip(1.),
            getattr(optax, "adam")(1e-2),
        )
    opt_state = optim.init(params)
    n_params = compute_num_params(params)
    n_epochs = 100
    rho = 1.


    def map_loss(
        params,
        model,
        x_batch,
        y_batch,
        alpha,
        n_params: int,
        N_datapoints_max: int,
    ):
        # define dict for logging purposes
        B = x_batch.shape[0]
        O = y_batch.shape[-1]
        D = n_params
        N = N_datapoints_max

        # hessian_scaler = 1

        vparams = tm.Vector(params)

        rho = 1.
        nll = lambda x, y, rho: 1/B * cross_entropy_loss(x, y, rho)

        y_pred = model.apply(params, x_batch)

        loglike_loss = nll(y_pred, y_batch, rho) #* hessian_scaler

        log_prior_term = -D / 2 * jnp.log(2 * jnp.pi) - (1 / 2) * alpha * (vparams @ vparams) + D / 2 * jnp.log(alpha)
        # log_det_term = 0
        loss = loglike_loss - 0. * log_prior_term

        return loss

    def make_step(params, alpha, opt_state, x, y):
        grad_fn = jax.value_and_grad(map_loss, argnums=0, has_aux=False)
        loss, grads = grad_fn(params, model, x, y, alpha, n_params, N)
        param_updates, opt_state = optim.update(grads, opt_state)
        params = optax.apply_updates(params, param_updates)
        return loss, params, opt_state

    jit_make_step = jit(make_step)

    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0
        epoch_accuracy = 0
        start_time = time.time()
        for _, batch in zip(range(n_batches), train_loader):
            X = batch["image"]
            y = batch["label"]
            B = X.shape[0]
            train_key, split_key = random.split(split_key)

            loss, params, opt_state = jit_make_step(params, alpha, opt_state, X, y)
            loss = loss
            epoch_loss += loss.item()

            epoch_accuracy += accuracy(params, model, X, y).item()

        epoch_accuracy /= (n_batches * B)
        epoch_time = time.time() - start_time
        print(
            f"epoch={epoch}, loss={epoch_loss:.3f}, , accuracy={epoch_accuracy:.2f}, alpha={alpha:.2f}, time={epoch_time:.3f}s"
        )
    # Save learned parameters
    train_stats_dict = {}
    train_stats_dict['n_params'] = n_params
    train_stats_dict['train_samples'] = train_samples
    train_stats_dict['classes_train'] = classes_train
    train_stats_dict['n_classes'] = n_classes
    train_stats_dict['model'] = model
    with open(f"./checkpoints/small_conv.pickle", "wb") as file:
        pickle.dump(
            {"params": params, "alpha": alpha, "rho": rho, "train_stats": train_stats_dict}, file
        )
        



import time
import argparse

import jax
import matplotlib.pyplot as plt
import optax
import tree_math as tm
from flax import linen as nn
from jax import nn as jnn
from jax import numpy as jnp
from jax import random, jit
import pickle
from src.models.fc import FC_NN
from src.helper import compute_num_params
from src.losses import mse_loss

def f(x):
    return jnp.sin(5 * x + 1) #+ jnp.cos(25 * x + 1) + jnp.exp(0.1 * x) + 5


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dims", type=int, default=1)
    parser.add_argument("--output_dim", type=int, default=1)
    parser.add_argument("--n_batches", type=int, default=20)
    parser.add_argument("--N", type=int, default=50)
    args = parser.parse_args()
    print(args)
    model_key, data_key, noise_key, split_key = random.split(random.PRNGKey(42), 4)
    input_dim = args.input_dims
    output_dim = args.output_dim
    n_batches = args.n_batches
    N = args.N
    # Initialize prior and likelihood precision

    x_train = random.uniform(data_key, (N, input_dim), minval=0, maxval=1)
    x_val = jnp.linspace(-1, 2, 100).reshape(-1, 1)
    B = int(N / n_batches)
    noise_std = 0.05
    rho = 1 / noise_std**2
    alpha = 10.
    log_alpha, log_rho = jnp.log(alpha), jnp.log(rho)
    y_train = f(x_train) + random.normal(noise_key, (N, input_dim)) * noise_std
    y_train = y_train[:, :output_dim]

    model = FC_NN(output_dim, 10, 2)
    params = model.init(model_key, x_train[:B])
    D = compute_num_params(params)
    print(f"Number of parameters: {D}")

    def map_loss(params, log_alpha, log_rho, x, y):
        B = x.shape[0]
        O = y.shape[-1]
        out = model.apply(params, x)
        vparams = tm.Vector(params)
        log_likelihood = (
            -N * O / 2 * jnp.log(2 * jnp.pi)
            + N * O / 2 * log_rho
            - (N / B) * 0.5 * rho * jnp.sum(jax.vmap(mse_loss)(out, y))  # Sum over the observations
        )
        log_prior = -D / 2 * jnp.log(2 * jnp.pi) + D / 2 * log_alpha - 0.5 * alpha * vparams @ vparams
        loss = log_likelihood + log_prior
        return -loss, (log_likelihood, log_prior)

    lr = 1e-3
    n_epochs = 2000
    optim = optax.adam(lr)
    opt_state = optim.init(params)
    estimator = "Normal"

    def make_step(params, opt_state, x, y):
        grad_fn = jax.value_and_grad(map_loss, argnums=0, has_aux=True)
        loss, grads = grad_fn(params, log_alpha, log_rho, x, y)
        param_updates, opt_state = optim.update(grads, opt_state)
        params = optax.apply_updates(params, param_updates)
        return loss, params, opt_state
    jit_make_step = jit(make_step)

    losses = []
    log_likelihoods = []
    log_priors = []
    mean_loss = []
    mse_preds = []
    # Training with Marginal Likelihood loss
    print("Starting training...")
    for epoch in range(n_epochs):
        start_time = time.time()
        train_key, split_key = random.split(split_key)
        batch_indices_shuffled = random.permutation(train_key, x_train.shape[0])
        for i in range(n_batches):
            train_key, split_key = random.split(split_key)
            x_batch = x_train[batch_indices_shuffled[i * B : (i + 1) * B]]
            y_batch = y_train[batch_indices_shuffled[i * B : (i + 1) * B]]
            loss, params, opt_state = jit_make_step(
                params, opt_state, x_batch, y_batch
            )
            loss, (log_likelihood, log_prior) = loss
            losses.append(loss)
            log_likelihoods.append(log_likelihood.item())
            log_priors.append(log_prior.item())
        epoch_time = time.time() - start_time
        log_likelihood_epoch_loss = jnp.mean(jnp.array(log_likelihoods[-B:]))
        epoch_loss = jnp.mean(jnp.array(losses[-B:]))
        epoch_prior = jnp.mean(jnp.array(log_priors[-B:]))
        print(
            f"epoch={epoch}, log likelihood loss={log_likelihood_epoch_loss:.2f}, loss ={epoch_loss:.2f}, prior loss={epoch_prior:.2f}, time={epoch_time:.3f}s"
        )

    # Save Learned parameters
    train_stats_dict = {}
    train_stats_dict['x_train'] = x_train
    train_stats_dict['y_train'] = y_train
    train_stats_dict['x_val'] = x_val
    train_stats_dict['y_val'] = f(x_val)
    train_stats_dict['model'] = model
    train_stats_dict['n_params'] = D

    with open(f"./checkpoints/syntetic_regression.pickle", "wb") as file:
        pickle.dump(
            {"args": args, "params": params, "alpha": alpha, "rho": rho, "train_stats": train_stats_dict}, file
        )

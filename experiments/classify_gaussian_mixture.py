"""Classify a Gaussian mixture model and calibrated different GGNs."""


import argparse
import os
import pickle

import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint

import matplotlib.pyplot as plt
import optax
from src.models.fc import FC_NN
from src.losses import cross_entropy_loss, accuracy_preds
from tueplots import bundles
config_plt = bundles.neurips2023()
config_plt['text.usetex'] = False
plt.rcParams.update(config_plt)


# Make directories

# Parse arguments
# todo: add seed-argument to argparse and
#  average results over seeds in dataframe script?

# A bunch of hyperparameters
seed = 1
num_data_in = 100
num_data_out = 100  # OOD
train_num_epochs = 100
train_batch_size = num_data_in
train_lrate = 1e-2
train_print_frequency = 10
calibrate_num_epochs = 100
calibrate_batch_size = num_data_in
calibrate_lrate = 1e-1
calibrate_print_frequency = 10
calibrate_log_alpha_min = 1e-3
numerics_lanczos_rank = 10
numerics_slq_num_samples = 100
numerics_slq_num_batches = 1
evaluate_num_samples = 100
plot_num_linspace = 250
plot_xmin, plot_xmax = -7, 7
plot_figsize = (8, 3)

# Create data
key = jax.random.PRNGKey(seed)
key, key_1, key_2 = jax.random.split(key, num=3)
m = 3.15
mu_1, mu_2 = jnp.array((-m, m)), jnp.array((m, -m))
x_1 = 0.6 * jax.random.normal(key_1, (num_data_in, 2)) + mu_1[None, :]
y_1 = jnp.asarray(num_data_in * [[1, 0]])
x_2 = 0.6 * jax.random.normal(key_2, (num_data_in, 2)) + mu_2[None, :]
y_2 = jnp.asarray(num_data_in * [[0, 1]])
x_train = jnp.concatenate([x_1, x_2], axis=0)
y_train = jnp.concatenate([y_1, y_2], axis=0)

# Create model
hidden = 16
num_layers = 2
model = FC_NN(2, 16, 2)
model_apply = model.apply
key, subkey = jax.random.split(key, num=2)
variables_dict = model.init(subkey, x_train)
variables, unflatten = jax.flatten_util.ravel_pytree(variables_dict)

# Train the model

optimizer = optax.adam(train_lrate)
optimizer_state = optimizer.init(variables)


def loss_p(v, x, y):
    logits = model_apply(unflatten(v), x)
    return cross_entropy_loss(preds=logits, y=y)


loss_value_and_grad = jax.jit(jax.value_and_grad(loss_p, argnums=0))

for epoch in range(train_num_epochs):
    # Subsample data
    key, subkey = jax.random.split(key)
    idx = jax.random.choice(
        subkey, x_train.shape[0], (train_batch_size,), replace=False
    )

    # Apply an optimizer-step
    loss, grad = loss_value_and_grad(variables, x_train[idx], y_train[idx])
    updates, optimizer_state = optimizer.update(grad, optimizer_state)
    variables = optax.apply_updates(variables, updates)

    # Look at intermediate results
    if epoch % train_print_frequency == 0:
        y_pred = model_apply(unflatten(variables), x_train[idx])
        y_probs = jax.nn.softmax(y_pred, axis=-1)
        acc = accuracy_preds(preds=y_probs, batch_y=y_train[idx])
        print(f"Epoch {epoch}, loss {loss:.3f}, accuracy {acc:.3f}")

print()

# Calibrate the linearised Laplace

# Linearise the model around the calibrated alpha
model_fn = lambda v, x: model_apply(unflatten(v), x)

def model_linear(sample, v, x):
    """Evaluate the model after linearising around the optimal parameters."""
    fx = model_fn(v, x)
    _, jvp = jax.jvp(lambda p: model_fn(p, x), (v,), (sample - v,))
    return fx + jvp

# Samples from laplace Diffusion, sampled laplace and linearised laplace
J = jax.jacfwd(model_fn)(variables, x_train)
pred = model_fn(variables, x_train)
pred = jax.nn.softmax(pred, axis=1)
pred = jax.lax.stop_gradient(pred)
# D = jax.vmap(jnp.diag)(pred)
# H = jnp.einsum("bo, bi->boi", pred, pred)
# H = D - H
GGN = 0
for i in range(J.shape[0]):
    H = jax.hessian(cross_entropy_loss)(pred[i], y_train[i])
    GGN += J[i].T @ H @ J[i]

alpha = 1.0
Cov_sqrt = jnp.linalg.cholesky(jnp.linalg.inv(GGN + alpha * jnp.eye(GGN.shape[0])))

eigvals, eigvecs = jnp.linalg.eigh(GGN)
threshold = 1e-3
idx = eigvals < threshold
idx_ = eigvals >= threshold

diag_ggn = 1/jnp.sqrt(eigvals + alpha)

diag_nonker = 1/jnp.sqrt(eigvals + alpha)
diag_nonker = jnp.where(idx, 0., diag_nonker)

diag_ker = 1/jnp.sqrt(eigvals + alpha)
diag_ker = jnp.where(idx_, 0., diag_ker)

# eigvecs = eigvecs[:, idx]
Cov_nonker = eigvecs @ (diag_nonker * eigvecs.T)
Cov_ker = eigvecs @ (diag_ker * eigvecs.T)



# Sample from the posterior
key, subkey = jax.random.split(key)
num_samples = 20
eps = jax.random.normal(subkey, (num_samples, len(variables)))
posterior_samples = jax.vmap(lambda e: variables + (eigvecs @ (diag_ggn * e)))(eps)
ker_samples = jax.vmap(lambda e: variables + (eigvecs @ (diag_ker * e)))(eps)
nonker_samples = jax.vmap(lambda e: variables + (eigvecs @ (diag_nonker * e)))(eps)

# Saple from lapalce Diffusion
diffusion_samples = jnp.stack([variables] * num_samples)
diffusion_ker_samples = jnp.stack([variables] * num_samples)
diffusion_nonker_samples = jnp.stack([variables] * num_samples)


# partial(jax.jit, static_argnames=("model", "rank", "integration_time", "n_evals"))
# def ode_ggn(model,
#             params,
#             random_init_dir,
#             v0,
#             rank,
#             n_evals,
#             integration_time,
#             x_train,
#             likelihood: Literal["classification", "regression"] = "classification",
#             rtol=1e-7,
#             atol=1e-7,
#             delta=1.0,
#             integration_subspace: Literal["kernel", "non-kernel"] = "kernel"):
#     p0_flat, unravel_func_p = jax.flatten_util.ravel_pytree(params)
#     def ode_func(params_, t):
#         # gvp = get_gvp_fun(model_fn, loss, unravel_func_p(params_), x_train, y_train)
#         gvp = get_ggn_vector_product(unravel_func_p(params_), model, x_train, None, likelihood)
#         if integration_subspace == "kernel":
#             gvp_ = lambda v: gvp(v) + delta * v
#             eigvals, eigvecs = lanczos_tridiag(gvp_, v0, rank - 1)
#             rhs = 1/delta * (gvp_(random_init_dir) - eigvecs @ jnp.diag(eigvals) @ eigvecs.T @ random_init_dir)
#         elif integration_subspace == "non-kernel":
#             eigvals, eigvecs = lanczos_tridiag(gvp, v0, rank - 1)
#             rhs = eigvecs @ eigvecs.T @ random_init_dir
#         return rhs
#     ode_y0 = p0_flat
#     t = jnp.linspace(0., integration_time, n_evals)
#     y_sols = odeint(ode_func, ode_y0, t, rtol=rtol, atol=atol)
#     sols = jax.vmap(unravel_func_p)(y_sols)
#     return sols, y_sols


# def non_ker_ode_solve_multiple_dir(single_dir):
#     n_evals = 2
#     integration_time = 1.0
#     rank = 7
#     v0 = jnp.ones(D)*5
#     nonker_posterior, y_sols = ode_ggn(model,params,single_dir,v0,rank,n_evals,integration_time,x_train,"regression",1e-7, 1e-7,1.0,"non-kernel")
#     nonker_posterior = jax.tree_map(lambda x: x[-1], nonker_posterior)
#     return nonker_posterior
    
# # nonker_posterior = jax.vmap(non_ker_ode_solve_multiple_dir)(random_init_dir)
# nonker_posterior = jax.lax.map(non_ker_ode_solve_multiple_dir, random_init_dir)



def get_GGN_spectrum(v):
    J = jax.jacfwd(model_fn)(v, x_train)
    pred = model_fn(v, x_train)
    pred = jax.nn.softmax(pred, axis=1)
    pred = jax.lax.stop_gradient(pred)
    D = jax.vmap(jnp.diag)(pred)
    H = jnp.einsum("bo, bi->boi", pred, pred)
    H = D - H
    GGN = 0
    for i in range(J.shape[0]):
        GGN += J[i].T @ H[i] @ J[i]

    eigvals, eigvecs = jnp.linalg.eigh(GGN)
    threshold = 1e-3
    threshold_ = 1e-7
    idx = eigvals < threshold
    idx_ = eigvals >= threshold_

    return eigvals, eigvecs, idx, idx_

# def sample_diffusions(random_dir):
#     def kernel_vector_field(v, t):
#         eigvals, eigvecs, idx, idx_ = get_GGN_spectrum(v)
#         # diag_ker = 1/jnp.sqrt(eigvals + alpha)
#         # diag_ker = jnp.where(idx_, 0., diag_ker)
#         # proj_ker = eigvecs @ (diag_ker * eigvecs.T)

#         eigvecs = eigvecs[:, :10]
#         proj_ker = 1/jnp.sqrt(alpha) * eigvecs @ eigvecs.T
#         rhs = proj_ker @ random_dir
#         return rhs
    
#     def non_kernel_vector_field(v, t):
#         eigvals, eigvecs, idx, idx_ = get_GGN_spectrum(v)
#         # diag_ker = 1/jnp.sqrt(eigvals + alpha)
#         # diag_ker = jnp.where(idx, 0., diag_ker)
#         # proj_nonker = eigvecs @ (diag_ker * eigvecs.T)
#         diag_ker = 1/jnp.sqrt(eigvals[-10:] + alpha)
#         eigvecs = eigvecs[:, -10:]
#         proj_nonker = eigvecs @ (jnp.diag(diag_ker) @ eigvecs.T)
#         rhs = proj_nonker @ random_dir
#         return rhs
    
#     n_evals = 2
#     integration_time = 0.5
#     t = jnp.linspace(0., integration_time, n_evals)
#     kernel_posterior = odeint(kernel_vector_field, random_dir, t)
#     non_kernel_posterior = odeint(non_kernel_vector_field, random_dir, t)
#     return kernel_posterior[-1], non_kernel_posterior[-1]

# kernel_posterior, non_kernel_posterior = sample_diffusions(jnp.ones_like(variables))
# ker_logits = model_fn(kernel_posterior, x_train)
# nonker_logits = model_fn(non_kernel_posterior, x_train)
# ker_probs = jax.nn.softmax(ker_logits, axis=-1)
# nonker_probs = jax.nn.softmax(nonker_logits, axis=-1)
# acc_ker = accuracy_preds(preds=ker_probs, batch_y=y_train)/y_train.shape[0]
# acc_nonker = accuracy_preds(preds=nonker_probs, batch_y=y_train)/y_train.shape[0]
# breakpoint()


        

n_steps = 100
def diffusion_step(v_full , v_nonker, v_ker, key):

    eps = jax.random.normal(key, (len(variables),))

    eigvals_nonker, eigvecs_nonker, idx, _ = get_GGN_spectrum(v_nonker)
    diag_nonker = 1/jnp.sqrt(eigvals_nonker + alpha)
    diag_nonker = jnp.where(idx, 0., diag_nonker)
    nonker_samples = v_nonker + 1/jnp.sqrt(n_steps) * (eigvecs_nonker @ (diag_nonker * eps))

    eigvals_ker, eigvecs_ker, _, idx_ = get_GGN_spectrum(v_ker)
    diag_ker = 1/jnp.sqrt(eigvals_ker + alpha)
    diag_ker = jnp.where(idx_, 0., diag_ker)
    ker_samples = v_ker + 1/jnp.sqrt(n_steps) * (eigvecs_ker @ (diag_ker * eps))
    
    eigvals_full, eigvecs_full, _, idx_ = get_GGN_spectrum(v_full)
    diag_full = 1/jnp.sqrt(eigvals_full + alpha)
    diag_full = jnp.where(idx_, 0., diag_full)
    full_samples = v_full + 1/jnp.sqrt(n_steps) * (eigvecs_full @ (diag_full * eps))
    eigvals_full, eigvecs_full, idx, _ = get_GGN_spectrum(full_samples)
    diag_full = 1/jnp.sqrt(eigvals_full + alpha)
    diag_full = jnp.where(idx, 0., diag_full)
    diffusion_sample = full_samples + 1/jnp.sqrt(n_steps) * (eigvecs_full @ (diag_full * eps))

    return diffusion_sample, nonker_samples, ker_samples
    
for i in range(n_steps):
    key, subkey = jax.random.split(key)
    key_list = jax.random.split(subkey, num=num_samples)
    diffusion_samples, diffusion_nonker_samples, diffusion_ker_samples = jax.vmap(lambda v_full, v_nonker, v_ker, k: diffusion_step(v_full, v_nonker, v_ker, k))(diffusion_samples, diffusion_nonker_samples, diffusion_ker_samples, key_list)

# Predict (in-distribution)
linearized_logits_fn = lambda sample_pytree, x: jax.vmap(lambda s: model_linear(s, variables, x))(sample_pytree)
sampled_logits_fn = lambda sample_pytree, x: jax.vmap(lambda s: model_fn(s, x))(sample_pytree)


# Create plotting grid
x_1d = jnp.linspace(plot_xmin, plot_xmax, num=plot_num_linspace)
x_plot_x, x_plot_y = jnp.meshgrid(x_1d, x_1d)
x_plot = jnp.stack((x_plot_x, x_plot_y)).reshape((2, -1)).T

# Compute marginal standard deviations for plotting inputs
linearized_logits = linearized_logits_fn(posterior_samples, x_plot)
linearized_ker_logits = linearized_logits_fn(ker_samples, x_plot)
linearized_nonker_logits = linearized_logits_fn(nonker_samples, x_plot)

sampled_logits = sampled_logits_fn(posterior_samples, x_plot)
sampled_ker_logits = sampled_logits_fn(ker_samples, x_plot)
sampled_nonker_logits = sampled_logits_fn(nonker_samples, x_plot)

diffusion_logits = sampled_logits_fn(diffusion_samples, x_plot)
diffusion_ker_logits = sampled_logits_fn(diffusion_ker_samples, x_plot)
diffusion_nonker_logits = sampled_logits_fn(diffusion_nonker_samples, x_plot)


linearized_probs = jax.nn.softmax(linearized_logits, axis=-1)
linearized_ker_probs = jax.nn.softmax(linearized_ker_logits, axis=-1)
linearized_nonker_probs = jax.nn.softmax(linearized_nonker_logits, axis=-1)

sampled_probs = jax.nn.softmax(sampled_logits, axis=-1)
sampled_ker_probs = jax.nn.softmax(sampled_ker_logits, axis=-1)
sampled_nonker_probs = jax.nn.softmax(sampled_nonker_logits, axis=-1)

diffusion_probs = jax.nn.softmax(diffusion_logits, axis=-1)
diffusion_ker_probs = jax.nn.softmax(diffusion_ker_logits, axis=-1)
diffusion_nonker_probs = jax.nn.softmax(diffusion_nonker_logits, axis=-1)

stdvs_linearized = jnp.sum(jnp.std(linearized_probs, axis=0), axis=-1)
stdvs_ker_linearized = jnp.sum(jnp.std(linearized_ker_probs, axis=0), axis=-1)
stdvs_nonker_linearized = jnp.sum(jnp.std(linearized_nonker_probs, axis=0), axis=-1)

stdvs_sampled = jnp.sum(jnp.std(sampled_probs, axis=0), axis=-1)
stdvs_ker_sampled = jnp.sum(jnp.std(sampled_ker_probs, axis=0), axis=-1)
stdvs_nonker_sampled = jnp.sum(jnp.std(sampled_nonker_probs, axis=0), axis=-1)

stdvs_diffusion = jnp.sum(jnp.std(diffusion_probs, axis=0), axis=-1)
stdvs_ker_diffusion = jnp.sum(jnp.std(diffusion_ker_probs, axis=0), axis=-1)
stdvs_nonker_diffusion = jnp.sum(jnp.std(diffusion_nonker_probs, axis=0), axis=-1)

stdev_linearized_plot = stdvs_linearized.T.reshape((plot_num_linspace, plot_num_linspace))
stdev_ker_linearized_plot = stdvs_ker_linearized.T.reshape((plot_num_linspace, plot_num_linspace))
stdev_nonker_linearized_plot = stdvs_nonker_linearized.T.reshape((plot_num_linspace, plot_num_linspace))

stdev_sampled_plot = stdvs_sampled.T.reshape((plot_num_linspace, plot_num_linspace))
stdev_ker_sampled_plot = stdvs_ker_sampled.T.reshape((plot_num_linspace, plot_num_linspace))
stdev_nonker_sampled_plot = stdvs_nonker_sampled.T.reshape((plot_num_linspace, plot_num_linspace))

stdev_diffusion_plot = stdvs_diffusion.T.reshape((plot_num_linspace, plot_num_linspace))
stdev_ker_diffusion_plot = stdvs_ker_diffusion.T.reshape((plot_num_linspace, plot_num_linspace))
stdev_nonker_diffusion_plot = stdvs_nonker_diffusion.T.reshape((plot_num_linspace, plot_num_linspace))

# Compute labels for plotting inputs
logits_plot = model_apply(unflatten(variables), x_plot)
labels_plot = jax.nn.log_softmax(logits_plot).argmax(axis=-1)
labels_plot = labels_plot.T.reshape((plot_num_linspace, plot_num_linspace))
# Choose a plotting style
style_data = {
    "in": {
        "color": "black",
        "zorder": 1,
        "linestyle": "None",
        "marker": "o",
        "markeredgecolor": "grey",
        "alpha": 0.75,
    },
    "out": {
        "color": "white",
        "zorder": 1,
        "linestyle": "None",
        "marker": "P",
        "markeredgecolor": "black",
        "alpha": 0.75,
    },
}
style_contour = {
    "uq": {"cmap": "viridis", "zorder": 0}, #"vmin": 0, "vmax": 1},
    # "bdry": {"vmin": 0, "vmax": 1, "cmap": "seismic", "zorder": 0, "alpha": 0.5},
}


# Plot the results
layout = [["uq_sam", "uq_sam_ker", "uq_sam_nonker"], 
           ["uq_lin", "uq_lin_ker", "uq_lin_nonker"], 
           ["uq_dif", "uq_dif_ker", "uq_dif_nonker"]] #"bdry"]]
_fig, axes = plt.subplot_mosaic(layout, sharex=True, sharey=True, figsize=plot_figsize, constrained_layout=True)

# axes["bdry"].set_title("Decision boundary")
# axes["bdry"].contourf(x_plot_x, x_plot_y, labels_plot, 3, **style_contour["bdry"])
# axes["bdry"].plot(x_train[:, 0], x_train[:, 1], **style_data["in"])


axes["uq_sam"].set_title("Sampled Full Distribution")
axes["uq_sam"].plot(x_train[:, 0], x_train[:, 1], **style_data["in"])
cbar = axes["uq_sam"].contourf(x_plot_x, x_plot_y, stdev_sampled_plot, **style_contour["uq"])

axes["uq_sam_ker"].set_title("Sampled Kernel Distribution")
axes["uq_sam_ker"].plot(x_train[:, 0], x_train[:, 1], **style_data["in"])
cbar = axes["uq_sam_ker"].contourf(x_plot_x, x_plot_y, stdev_ker_sampled_plot, **style_contour["uq"])

axes["uq_sam_nonker"].set_title("Sampled Non-Kernel Distribution")
axes["uq_sam_nonker"].plot(x_train[:, 0], x_train[:, 1], **style_data["in"])
cbar = axes["uq_sam_nonker"].contourf(x_plot_x, x_plot_y, stdev_nonker_sampled_plot, **style_contour["uq"])
plt.colorbar(cbar)


axes["uq_lin"].set_title("Linearized Laplace uncertainty")
axes["uq_lin"].plot(x_train[:, 0], x_train[:, 1], **style_data["in"])
cbar = axes["uq_lin"].contourf(x_plot_x, x_plot_y, stdev_linearized_plot, **style_contour["uq"])

axes["uq_lin_ker"].set_title("Linearized Kernel uncertainty")
axes["uq_lin_ker"].plot(x_train[:, 0], x_train[:, 1], **style_data["in"])
cbar = axes["uq_lin_ker"].contourf(x_plot_x, x_plot_y, stdev_ker_linearized_plot, **style_contour["uq"])

axes["uq_lin_nonker"].set_title("Linearized Non-Kernel uncertainty")
axes["uq_lin_nonker"].plot(x_train[:, 0], x_train[:, 1], **style_data["in"])
cbar = axes["uq_lin_nonker"].contourf(x_plot_x, x_plot_y, stdev_nonker_linearized_plot, **style_contour["uq"])
plt.colorbar(cbar)


axes["uq_dif"].set_title("Laplace Diffusion uncertainty")
axes["uq_dif"].plot(x_train[:, 0], x_train[:, 1], **style_data["in"])
cbar = axes["uq_dif"].contourf(x_plot_x, x_plot_y, stdev_diffusion_plot, **style_contour["uq"])

axes["uq_dif_ker"].set_title("Diffusion Kenrel uncertainty")
axes["uq_dif_ker"].plot(x_train[:, 0], x_train[:, 1], **style_data["in"])
cbar = axes["uq_dif_ker"].contourf(x_plot_x, x_plot_y, stdev_ker_diffusion_plot, **style_contour["uq"])

axes["uq_dif_nonker"].set_title("Diffusion Non-Kernel uncertainty")
axes["uq_dif_nonker"].plot(x_train[:, 0], x_train[:, 1], **style_data["in"])
cbar = axes["uq_dif_nonker"].contourf(x_plot_x, x_plot_y, stdev_nonker_diffusion_plot, **style_contour["uq"])
plt.colorbar(cbar)



# Save the plot to a file
plt.savefig("./figures/classify_gaussian_mixture.pdf")

import jax
import jax.numpy as jnp
import flax
import torch
import optax
import time

from src.losses import log_posterior_loss, accuracy
from src.helper import compute_num_params

def maximum_a_posteriori(
    model: flax.linen.Module,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
    key: jax.random.PRNGKey,
    args_dict: dict,
):
    """
    Maximize the posterior for a given model and dataset.
    :param model: initialized model to use for training
    :param train_loader: train dataloader (torch.utils.data.DataLoader)
    :param valid_loader: test dataloader (torch.utils.data.DataLoader)
    :param key: random.PRNGKey for jax modules
    :param args_dict: dictionary of arguments for training passed from the command line
    :return: params
    """

    alpha, rho = args_dict["prior_precision"], args_dict["likelihood_precision"]
    likelihood_type = args_dict["likelihood"]
    N = len(train_loader.dataset) 
    batch = next(iter(train_loader))
    x_init, y_init = jnp.array(batch[0].numpy()), jnp.array(batch[1].numpy()),

    params = model.init(key, x_init)
    D = compute_num_params(params)

    optim = getattr(optax, args_dict["optimizer"])(args_dict["learning_rate"])
    opt_state = optim.init(params)

    @jax.jit
    def make_step(params, opt_state, x, y):
        grad_fn = jax.value_and_grad(log_posterior_loss, argnums=0, has_aux=True)
        loss, grads = grad_fn(
            params,
            alpha,
            rho,
            model,
            x,
            y,
            D,
            N,
            likelihood_type,
        )
        param_updates, opt_state = optim.update(grads, opt_state)
        params = optax.apply_updates(params, param_updates)
        return loss, params, opt_state

    print("Starting training...")
    epoch_log_posterior = []
    epoch_log_likelihood, epoch_log_prior = [], []
    epoch_accuracy = []
    for epoch in range(1, args_dict["n_epochs"] + 1):
        log_posterior = 0.
        log_likelihood, log_prior = 0., 0.
        accuracy_stat = 0.
        n_batches = 0
        start_time = time.time()
        for batch in train_loader:
            n_batches += 1
            X = jnp.array(batch[0].numpy())
            Y = jnp.array(batch[1].numpy())
            B = X.shape[0]

            (loss, loss_dict), params, opt_state = make_step(
                params,
                opt_state,
                X,
                Y,
            )
            log_posterior -= loss.item()
            log_likelihood += loss_dict["log_likelihood"].item()
            log_prior += loss_dict["log_prior"].item()
            if likelihood_type=="classification":
                accuracy_stat += accuracy(params, model, X, Y).item()
    
        accuracy_stat /= N
        log_likelihood /= N
        log_prior /= n_batches
        log_posterior /= n_batches
        if likelihood_type=="classification":
            print(f"epoch={epoch} averages - log likelihood={log_likelihood:.3f}, log prior={log_prior:.2f}, loss={-log_posterior:.2f}, accuracy={accuracy_stat:.2f}, time={time.time() - start_time:.3f}s")
        elif likelihood_type=="regression":
            print(f"epoch={epoch} averages - log likelihood={log_likelihood:.3f}, log prior={log_prior:.2f}, loss={-log_posterior:.2f}, time={time.time() - start_time:.3f}s")
        epoch_log_posterior.append(log_posterior)
        epoch_log_likelihood.append(log_likelihood)
        epoch_log_prior.append(log_prior)
        epoch_accuracy.append(accuracy_stat)

        if epoch % args_dict["test_every_n_epoch"] != 0 and epoch != args_dict["n_epochs"]:
            continue

        def get_precise_stats(loader):
            accuracy_stat, mse = 0., 0.
            log_likelihood, log_prior = 0., 0.
            start_time = time.time()
            for batch in loader:
                X = jnp.array(batch[0].numpy())
                Y = jnp.array(batch[1].numpy())
                B = X.shape[0]
                if likelihood_type=="classification":
                    accuracy_stat += accuracy(params, model, X, Y).item()
                _, loss_dict = log_posterior_loss(
                    params,
                    alpha,
                    rho,
                    model,
                    X,
                    Y,
                    D,
                    len(loader.dataset),
                    likelihood_type,
                    extra_stats = True
                )
                log_likelihood += loss_dict["log_likelihood"].item()
                log_prior = loss_dict["log_prior"].item()
                mse += loss_dict["sum_squared_error"].item()
            accuracy_stat /= len(loader.dataset)
            mse /= len(loader.dataset)
            log_posterior = log_likelihood + log_prior  # posterior per dataset
            log_likelihood /= len(loader.dataset)       # likelihood per datapoint
            stats_dict = {
                "log_likelihood": log_likelihood, 
                "log_prior": log_prior, 
                "log_posterior": log_posterior, 
                "accuracy": accuracy_stat, 
                "mse": mse,
                "time": time.time() - start_time}
            return stats_dict
        
        epoch_stats_dict = {
            "epoch_log_likelihood": epoch_log_likelihood,
            "epoch_log_prior": epoch_log_prior,
            "epoch_accuracy": epoch_accuracy,
            "epoch_log_posterior": epoch_log_posterior
        }

        stats_dict = get_precise_stats(train_loader)
        if likelihood_type=="classification":
            print(f"Train stats\t - log likelihood={stats_dict['log_likelihood']:.3f}, log prior={stats_dict['log_prior']:.2f}, loss={-stats_dict['log_posterior']:.2f}, accuracy={stats_dict['accuracy']:.2f}, mse={stats_dict['mse']:.2f}, time={stats_dict['time']:.3f}s")
        elif likelihood_type=="regression":
            print(f"Train stats\t - log likelihood={stats_dict['log_likelihood']:.3f}, log prior={stats_dict['log_prior']:.2f}, loss={-stats_dict['log_posterior']:.2f}, mse={stats_dict['mse']:.2f}, time={stats_dict['time']:.3f}s")
        train_stats_dict = {'train '+k : v for k,v in stats_dict.items()}    

        stats_dict = get_precise_stats(valid_loader)
        if likelihood_type=="classification":
            print(f"Validation stats - log likelihood={stats_dict['log_likelihood']:.3f}, log prior={stats_dict['log_prior']:.2f}, loss={-stats_dict['log_posterior']:.2f}, accuracy={stats_dict['accuracy']:.2f}, mse={stats_dict['mse']:.2f}, time={stats_dict['time']:.3f}s")
        elif likelihood_type=="regression":
            print(f"Validation stats - log likelihood={stats_dict['log_likelihood']:.3f}, log prior={stats_dict['log_prior']:.2f}, loss={-stats_dict['log_posterior']:.2f}, mse={stats_dict['mse']:.2f}, time={stats_dict['time']:.3f}s")
        test_stats_dict = {'test '+k : v for k,v in stats_dict.items()}   

        stats_dict = {**train_stats_dict, **test_stats_dict, **epoch_stats_dict}

    return params, stats_dict
import numpy as onp
import jax.numpy as jnp
import jax
from src.losses import nll
from src.sampling.predictive_samplers import sample_predictive

def evaluate(test_loader, posterior_samples, params, model_fn, eval_args, is_cifar=True):
    all_y_prob = []
    all_y_log_prob = []
    all_y_true = []
    all_y_var = []
    for x_test, y_test in test_loader:
        x_test = jnp.array(x_test.numpy())
        if is_cifar:
            # Only for rebuttal:
            x_test = jnp.moveaxis(x_test, 2, 3)
            ############################
        y_test = jnp.array(y_test.numpy())
        if len(y_test.shape) == 1:
            y_test = jax.nn.one_hot(y_test, num_classes=10)

        predictive_samples = sample_predictive(
            posterior_samples=posterior_samples,
            params=params,
            model_fn=model_fn,
            x_test=x_test,
            linearised_laplace=eval_args["linearised_laplace"],
            posterior_sample_type=eval_args["posterior_sample_type"],
        )
        predictive_samples_mean = jnp.mean(predictive_samples, axis=0)
        # if eval_args["likelihood"] == "regression":
        predictive_samples_std = jnp.std(predictive_samples, axis=0)
        all_y_var.append(predictive_samples_std**2)

        y_prob = jax.nn.softmax(predictive_samples_mean, axis=-1)
        y_log_prob = jax.nn.log_softmax(predictive_samples_mean, axis=-1)

        # import pdb; pdb.set_trace()
        all_y_prob.append(y_prob)
        all_y_log_prob.append(y_log_prob)
        all_y_true.append(y_test)

    all_y_prob = jnp.concatenate(all_y_prob, axis=0)
    all_y_log_prob = jnp.concatenate(all_y_log_prob, axis=0)
    all_y_true = jnp.concatenate(all_y_true, axis=0)

    # compute some metrics: mean confidence, accuracy and negative log-likelihood
    metrics = {}
    if eval_args["likelihood"] == "classification":
        all_y_var = jnp.concatenate(all_y_var, axis=0)
        metrics["conf"] = (jnp.max(all_y_prob, axis=1)).mean().item()
        metrics["nll"] = (-jnp.mean(jnp.sum(all_y_log_prob * all_y_true, axis=-1), axis=-1)).mean()
        metrics["acc"] =  (jnp.argmax(all_y_prob, axis=1)==jnp.argmax(all_y_true, axis=1)).mean().item()
    elif eval_args["likelihood"] == "regression":
        sigma_noise = 1  # TODO: define sigma_noise!
        all_y_var = jnp.concatenate(all_y_var, axis=0) + sigma_noise**2
        metrics["nll"] = nll(all_y_prob, all_y_true, all_y_var)
    else:
        raise ValueError("Unknown likelihood.")

    # cast to numpy
    all_y_prob = onp.array(all_y_prob)
    

    return metrics, all_y_prob, all_y_true, all_y_var



def evaluate_map(test_loader, params, model_fn, eval_args, is_cifar=True):
    all_y_prob = []
    all_y_log_prob = []
    all_y_true = []
    all_y_var = []
    for x_test, y_test in test_loader:
        x_test = jnp.array(x_test.numpy())
        if is_cifar:
            # Only for rebuttal:
            x_test = jnp.moveaxis(x_test, 2, 3)
            ############################
        y_test = jnp.array(y_test.numpy())
        if len(y_test.shape) == 1:
            y_test = jax.nn.one_hot(y_test, num_classes=10)

        predictive_samples = model_fn(params, x_test)
        # predictive_samples_mean = jnp.mean(predictive_samples, axis=0)
        if eval_args["likelihood"] == "regression":
            predictive_samples_std = jnp.std(predictive_samples, axis=0)
            all_y_var.append(predictive_samples_std**2)

        y_prob = jax.nn.softmax(predictive_samples, axis=-1)
        y_log_prob = jax.nn.log_softmax(predictive_samples, axis=-1)

        # import pdb; pdb.set_trace()
        all_y_prob.append(y_prob)
        all_y_log_prob.append(y_log_prob)
        all_y_true.append(y_test)

    all_y_prob = jnp.concatenate(all_y_prob, axis=0)
    all_y_log_prob = jnp.concatenate(all_y_log_prob, axis=0)
    all_y_true = jnp.concatenate(all_y_true, axis=0)

    # compute some metrics: mean confidence, accuracy and negative log-likelihood
    metrics = {}
    if eval_args["likelihood"] == "classification":
        all_y_var = None
        metrics["conf"] = (jnp.max(all_y_prob, axis=1)).mean().item()
        metrics["nll"] = (-jnp.mean(jnp.sum(all_y_log_prob * all_y_true, axis=-1), axis=-1)).mean()
        metrics["acc"] =  (jnp.argmax(all_y_prob, axis=1)==jnp.argmax(all_y_true, axis=1)).mean().item()
    elif eval_args["likelihood"] == "regression":
        sigma_noise = 1  # TODO: define sigma_noise!
        all_y_var = jnp.concatenate(all_y_var, axis=0) + sigma_noise**2
        metrics["nll"] = nll(all_y_prob, all_y_true, all_y_var)
    else:
        raise ValueError("Unknown likelihood.")

    # cast to numpy
    all_y_prob = onp.array(all_y_prob)
    

    return metrics, all_y_prob, all_y_true, all_y_var

def evaluate_samples(test_loader, posterior_samples, params, model_fn, eval_args):
    all_y_prob = []
    all_y_log_prob = []
    all_y_true = []
    all_y_var = []
    for x_test, y_test in test_loader:
        x_test = jnp.array(x_test.numpy())
        y_test = jnp.array(y_test.numpy())
        if len(y_test.shape) == 1:
            y_test = jax.nn.one_hot(y_test, num_classes=10)

        predictive_samples = sample_predictive(
            posterior_samples=posterior_samples,
            params=params,
            model_fn=model_fn,
            x_test=x_test,
            linearised_laplace=eval_args["linearised_laplace"],
            posterior_sample_type=eval_args["posterior_sample_type"],
        )
        # predictive_samples_mean = jnp.mean(predictive_samples, axis=0)
        if eval_args["likelihood"] == "regression":
            predictive_samples_std = jnp.std(predictive_samples, axis=0)
            all_y_var.append(predictive_samples_std**2)

        y_prob = jax.nn.softmax(predictive_samples, axis=-1)
        y_log_prob = jax.nn.log_softmax(predictive_samples, axis=-1)

        # import pdb; pdb.set_trace()
        all_y_prob.append(y_prob)
        all_y_log_prob.append(y_log_prob)
        all_y_true.append(y_test)
    all_y_prob = jnp.concatenate(all_y_prob, axis=1)
    all_y_log_prob = jnp.concatenate(all_y_log_prob, axis=1)
    all_y_true = jnp.concatenate(all_y_true, axis=0)


    # compute some metrics: mean confidence, accuracy and negative log-likelihood
    metrics = {}
    if eval_args["likelihood"] == "classification":
        all_y_var = None
        metrics["conf"] = list(map(lambda x: (jnp.max(x, axis=-1)).mean().item(), all_y_prob))
        metrics["nll"] = list(map(lambda x: (-jnp.mean(jnp.sum(x * all_y_true, axis=-1), axis=-1)).mean(), all_y_log_prob))
        metrics["acc"] =  list(map(lambda x: (jnp.argmax(x, axis=1)==jnp.argmax(all_y_true, axis=1)).mean().item(), all_y_prob))
    elif eval_args["likelihood"] == "regression":
        sigma_noise = 1  # TODO: define sigma_noise!
        all_y_var = jnp.concatenate(all_y_var, axis=0) + sigma_noise**2
        metrics["nll"] = nll(all_y_prob, all_y_true, all_y_var)
    else:
        raise ValueError("Unknown likelihood.")

    # cast to numpy
    all_y_prob = onp.array(all_y_prob)
    

    return metrics, all_y_prob, all_y_true, all_y_var

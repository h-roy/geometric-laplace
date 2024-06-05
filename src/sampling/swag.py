import flax
import jax
import jax.numpy as jnp
from functools import partial
from typing import Literal
from jax import flatten_util
import optax
import tqdm
import time
from src.helper import compute_num_params
from src.models.resnet import ResNet

def has_batchstats(model: flax.linen.Module):
    return isinstance(model, ResNet)


def mse_loss(preds, y):
    residual = preds - y
    return jnp.mean(residual**2)

@partial(jax.jit, static_argnames=['rho'])
def log_gaussian_log_loss(preds, y, rho=1.0):
    """
    preds: (batch_size, output_dim) (predictions)
    y: (batch_size, output_dim) (targets)
    """
    O = y.shape[-1]
    return 0.5 * O * jnp.log(2 * jnp.pi) - 0.5 * O * jnp.log(rho) + 0.5 * rho * mse_loss(preds, y)

@partial(jax.jit, static_argnames=['rho'])
def cross_entropy_loss(preds, y, rho=1.0):
    """
    preds: (batch_size, n_classes) (logits)
    y: (batch_size, n_classes) (one-hot labels)
    """
    preds = preds * rho
    preds = jax.nn.log_softmax(preds, axis=-1)
    return -jnp.mean(jnp.sum(preds * y, axis=-1))


@partial(jax.jit, static_argnames=['rho'])
def multiclass_binary_cross_entropy_loss(preds, y, rho=1.0):
    """
    preds: (batch_size, n_classes) (logits) Each element is the unnormalized log probability of a binary
      prediction.
    y: (batch_size, n_classes) Binary labels whose values are {0,1} or multi-class target
      probabilities. See note about compatibility with `logits` above.
    """
    preds = preds * rho
    y = y.astype(preds.dtype)
    log_p = jax.nn.log_sigmoid(preds)
    log_not_p = jax.nn.log_sigmoid(-preds)
    return jnp.mean(y * log_p + (1. - y) * log_not_p)

#@partial(jax.jit, static_argnames=['model', 'train', 'likelihood'])
def calculate_loss_with_batchstats(
    model, 
    params, 
    batch_stats, 
    x, 
    y, 
    train: bool = False,
    likelihood: Literal["classification", "regression", "binary_multiclassification"] = "classification"
):
    if likelihood == "regression":
        negative_log_likelihood = log_gaussian_log_loss
    elif likelihood == "classification":
        negative_log_likelihood = cross_entropy_loss
    elif likelihood == "binary_multiclassification":
        negative_log_likelihood = multiclass_binary_cross_entropy_loss
    else:
        raise ValueError(f"Likelihood {likelihood} not supported. Use either 'regression', 'classification' or 'binary_multiclassification'.")
    outs = model.apply({'params': params, 'batch_stats': batch_stats},
                x,
                train=train,
                mutable=['batch_stats'] if train else False)
    preds, new_model_state = outs if train else (outs, None)
    loss = negative_log_likelihood(preds, y)
    if likelihood == "regression":
        sse = jnp.sum((preds-y)**2)
        return loss, (sse, new_model_state)
    elif likelihood == "classification":
        acc = jnp.sum(preds.argmax(axis=-1) == y.argmax(axis=-1))
        return loss, (acc, new_model_state)
    elif likelihood == "binary_multiclassification":
        num_classes = preds.shape[1]
        #acc = jnp.sum((preds>0.) == (y==1)) / num_classes
        acc = jnp.sum((preds>0.) == (y==1), axis=0)
        return loss, (acc, new_model_state)



@partial(jax.jit, static_argnames=['model', 'likelihood'])
def calculate_loss_without_batchstats(
    model, 
    params, 
    x, 
    y,
    likelihood: Literal["classification", "regression", "binary_multiclassification"] = "classification",
):
    if likelihood == "regression":
        negative_log_likelihood = log_gaussian_log_loss
    elif likelihood == "classification":
        negative_log_likelihood = cross_entropy_loss
    elif likelihood == "binary_multiclassification":
        negative_log_likelihood = multiclass_binary_cross_entropy_loss
    else:
        raise ValueError(f"Likelihood {likelihood} not supported. Use either 'regression', 'classification' or 'binary_multiclassification'.")
    preds = model.apply(params, x)
    loss = negative_log_likelihood(preds, y)

    if likelihood == "regression":
        sse = jnp.sum((preds-y)**2)
        return loss, (sse, )
    elif likelihood == "classification":
        acc = jnp.sum(preds.argmax(axis=-1) == y.argmax(axis=-1))
        return loss, (acc, )
    elif likelihood == "binary_multiclassification":
        num_classes = preds.shape[1]
        #acc = jnp.sum((preds>0.) == (y==1)) / num_classes
        acc = jnp.sum((preds>0.) == (y==1), axis=0)
        return loss, (acc, )


def swag_score_fun(
        model, 
        params_dict,
        key,
        num_samples,
        train_loader, 
        likelihood: Literal["classification", "regression"]="classification", 
        diag_only=True, max_num_models=20, swa_c_epochs=1, swa_c_batches=None,
        swa_lr=0.01, momentum=0.9, wd=3e-4
    ):

    ########################
    # initialize optimizer #
    optimizer = optax.chain(
        optax.add_decayed_weights(wd),
        optax.sgd(swa_lr, momentum=momentum)
    )
    opt_state = optimizer.init(params_dict['params'])

    ########################
    # define training step #
    if not (has_batchstats(model)):
        #@jax.jit
        def train_step(opt_state, params_dict, x, y):
            loss_fn = lambda p: calculate_loss_without_batchstats(
                model, 
                p, 
                x, 
                y, 
                likelihood=likelihood
            )
            # Get loss, gradients for loss, and other outputs of loss function
            ret, grads = jax.value_and_grad(loss_fn, has_aux=True)(params_dict['params'])
            loss, (acc_or_sse, ) = ret
            # Update parameters
            param_updates, opt_state = optimizer.update(grads, opt_state, params_dict['params'])
            params_dict['params'] = optax.apply_updates(params_dict['params'], param_updates)
            return opt_state, params_dict, loss, acc_or_sse
    else:
        #@jax.jit
        def train_step(opt_state, params_dict, x, y):
            loss_fn = lambda p: calculate_loss_with_batchstats(
                model, 
                p, 
                params_dict['batch_stats'], 
                x, 
                y, 
                train=True, 
                likelihood=likelihood
            )
            # Get loss, gradients for loss, and other outputs of loss function
            ret, grads = jax.value_and_grad(loss_fn, has_aux=True)(params_dict['params'])
            loss, (acc_or_sse, new_model_state) = ret
            # Update parameters and batch statistics
            param_updates, opt_state = optimizer.update(grads, opt_state, params_dict['params'])
            params_dict = {
                'params' : optax.apply_updates(params_dict['params'], param_updates),
                'batch_stats' : new_model_state['batch_stats']
            }
            return opt_state, params_dict, loss, acc_or_sse
    
    def fit_batch_stats(train_loader, params_dict):
        for batch in train_loader:
            X = jnp.array(batch['image'])
            _, new_model_state = model.apply(
                params_dict,
                X,
                train = True,
                mutable = ['batch_stats'])
            params_dict = {
                'params' : params_dict['params'],
                'batch_stats' : new_model_state['batch_stats']
            }
        return params_dict['batch_stats']

        
    ######################
    # define SWAG update #
    #@jax.jit
    def collect_model(params_dict, mean, sq_mean, n_models, cov_mat_sqrt):

        params = flatten_util.ravel_pytree(params_dict['params'])[0]

        # first moment
        mean = mean * n_models / (
            n_models + 1.0
        ) + params / (n_models + 1.0)

        # second moment
        sq_mean = sq_mean * n_models / (
            n_models + 1.0
        ) + params ** 2 / (n_models + 1.0)

        # square root of covariance matrix
        if diag_only is False:

            # block covariance matrices, store deviation from current mean
            dev = (params - mean)

            cov_mat_sqrt.append(dev)

            # remove first column if we have stored too many models
            if n_models + 1 > max_num_models:
                cov_mat_sqrt = cov_mat_sqrt[1:]

        n_models += 1
        return mean, sq_mean, cov_mat_sqrt, n_models
        
    ###################################
    # start stochastic gradient steps #
    num_params = compute_num_params(params_dict['params'])
    mean = jnp.zeros(num_params)
    sq_mean = jnp.zeros(num_params)
    cov_mat_sqrt = []
    n_models = 0

    if swa_c_epochs is not None and swa_c_batches is not None:
        raise RuntimeError("One of swa_c_epochs or swa_c_batches must be None!")
    if swa_c_epochs is not None:
        n_epochs = swa_c_epochs * max_num_models
    else:
        n_epochs = 1 + (max_num_models * swa_c_batches) // len(train_loader)


    start = time.time()
    #epochs_bar = tqdm.tqdm(range(n_epochs))
    #for epoch in epochs_bar:
    for epoch in range(n_epochs):
        loss_avg = 0.
        batches_bar = tqdm.tqdm(enumerate(train_loader))
        for batch_idx, batch in batches_bar:
        #for batch_idx, batch in enumerate(train_loader):
            X = jnp.array(batch['image'])
            Y = jnp.array(batch['label'])

            opt_state, params_dict, batch_loss, batch_acc_or_sse = train_step(opt_state, params_dict, X, Y)

            loss_avg += batch_loss.item()

            if swa_c_batches is not None and (epoch*len(train_loader) + batch_idx + 1) % swa_c_batches == 0:
                mean, sq_mean, cov_mat_sqrt, n_models = collect_model(params_dict, mean, sq_mean, n_models, cov_mat_sqrt)
                if n_models == max_num_models:
                    break
            batches_bar.set_description(f"Epoch {epoch}/{n_epochs}, batch {batch_idx}/{len(train_loader)}, loss = {loss_avg/len(train_loader):.4f}")

        if swa_c_epochs is not None and epoch % swa_c_epochs == 0:
            mean, sq_mean, cov_mat_sqrt, n_models = collect_model(params_dict, mean, sq_mean, n_models, cov_mat_sqrt)
        
        #epochs_bar.set_description(f"Epoch {epoch}/{n_epochs}, loss = {loss_avg/len(train_loader):.4f}")
    
    print(f"Models collection took {time.time()-start} seconds")

    cov_mat_sqrt = jnp.array(cov_mat_sqrt)
    cov_mat_sqrt = cov_mat_sqrt.transpose()
    print("Covariance matrix done, shape", cov_mat_sqrt.shape)

    ##########################
    # define sample function #
    @jax.jit
    def sample(key, scale=0.5):
        key1, key2 = jax.random.split(key, 2)

        # draw diagonal variance sample
        var = jnp.clip(sq_mean - mean ** 2, 1e-30, None)
        rand_sample = (var ** 0.5) * jax.random.normal(key1, shape=(num_params,))

        # if covariance draw low rank sample
        if diag_only is False:
            cov_sample = cov_mat_sqrt @ jax.random.normal(key2, shape=(max_num_models,))
            cov_sample /= (max_num_models - 1) ** 0.5
            rand_sample += cov_sample

        # update sample with mean and scale
        sample = mean + scale**0.5 * rand_sample

        return sample
    


    devectorize_fun = flatten_util.ravel_pytree(params_dict['params'])[1]

    @jax.jit
    def sample_swag_fun(key):
        preds = []
        for s in range(50):
            key, key_s = jax.random.split(key, 2)
            sample_params = devectorize_fun(sample(key_s, scale=0.5))
        return sample_params    
    
    key_list = jax.random.split(key, num_samples)
    samples = jax.vmap(sample_swag_fun)(key_list)
    return samples
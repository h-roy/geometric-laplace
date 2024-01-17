import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.jit, static_argnames=['model'])
def get_accuracy(params, model, batch_x, batch_y):
    preds = model.apply(params, batch_x)
    return jnp.sum(preds.argmax(axis=-1) == batch_y.argmax(axis=-1))

def compute_num_params(pytree):
    return sum(x.size if hasattr(x, "size") else 0 for x in jax.tree_util.tree_leaves(pytree))
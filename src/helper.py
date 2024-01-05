import jax

def compute_num_params(pytree):
    return sum(x.size if hasattr(x, "size") else 0 for x in jax.tree_util.tree_leaves(pytree))

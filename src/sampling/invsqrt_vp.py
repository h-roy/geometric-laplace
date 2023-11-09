import jax
import jax.numpy as jnp


def inv_sqrt_vp(
        lr_eigvals: jax.Array,
        lr_eigvecs: jax.Array,
        prior_precision: float
):
    def matvec(v):
        diag_mat = 1/jnp.sqrt(lr_eigvals + prior_precision) - 1/jnp.sqrt(prior_precision)
        v_ = (diag_mat.reshape(-1,1) * lr_eigvecs.T) @ v
        v_ = lr_eigvecs @ v_
        return 1/jnp.sqrt(prior_precision) * v + v_
    
    return matvec

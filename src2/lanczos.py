import jax
import jax.numpy as jnp
import scipy
import numpy as np
from matfree import decomp


def full_reorth_lanczos(key, mv_prod, dim, n_iter):
    lanczos_alg = decomp.lanczos_tridiag_full_reortho(n_iter - 1)
    v0 = jax.random.normal(key, shape=(dim, ))
    v0 /= jnp.sqrt(dim)
    basis, (diag, offdiag) = decomp.decompose_fori_loop(v0, mv_prod, algorithm=lanczos_alg)
    eig_val, trid_eig_vec = scipy.linalg.eigh_tridiagonal(diag, offdiag, lapack_driver='stebz')
    # flip eigenvalues and eigenvectors so that they are in decreasing order
    trid_eig_vec = np.stack(list(trid_eig_vec.T)[::-1], axis=1)
    eig_val = np.array(list(eig_val)[::-1])
    # multiply eigenvector matrices
    eig_vec = basis.T @ trid_eig_vec
    
    return eig_vec, eig_val 
import jax
import jax.numpy as jnp
import matfree
from matfree import decomp, lanczos, montecarlo
from matfree.backend import func, linalg, np
from typing import Callable, Literal, Optional


def lanczos_tridiag(
        Av: Callable,
        v0: jax.Array,
        order: int
):
    ncols = v0.shape[0]
    if order >= ncols or order < 1:
        raise ValueError
    algorithm = matfree.lanczos.tridiagonal_full_reortho(order)
    u0 = v0/jnp.linalg.norm(v0)
    basis, tridiag = decomp.decompose_fori_loop(u0, Av, algorithm=algorithm)
    (diag, off_diag) = tridiag
    diag = linalg.diagonal_matrix(diag)
    offdiag1 = linalg.diagonal_matrix(off_diag, -1)
    offdiag2 = linalg.diagonal_matrix(off_diag, 1)
    dense_matrix = diag + offdiag1 + offdiag2
    eigvals, tri_eigvecs = linalg.eigh(dense_matrix)
    eigvecs = basis.T @ tri_eigvecs
    return eigvals, eigvecs

def lanczos_bidiag(
        Av: Callable,
        v0: jax.Array,
        order: int
):
    raise NotImplementedError


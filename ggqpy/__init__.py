import numpy as np
import scipy as sp

from ggqpy.discretize import Discretizer
from ggqpy.compress import compress_sequence_of_functions
from ggqpy.optimize import QuadOptimizer


def construct_Chevyshev_quadratures(eval_points: tuple, w, U):
    r = U.T @ w
    k = len(r)

    B = np.sqrt(w) * U.T
    Q, R, perm = sp.linalg.qr(B, pivoting=True)
    z = np.linalg.solve(R[:k, :k], Q.conj().T @ r)

    idx_cheb = perm[:k]
    
    eval_points = tuple(map(lambda x: x[idx_cheb], eval_points))
    
    w_cheb = z * np.sqrt(w[idx_cheb])

    return eval_points, w_cheb

def generalized_gaussian_quadrature(
    function_family,
    discretizer=Discretizer(),
    optimizer=None,
    eps_comp=1e-3,
    eps_quad=1e-3,
):

    x_disc, w_disc, endpoints, intervals = discretizer.adaptive_discretization(
        function_family
    )

    U_disc, A, rank, u_list = compress_sequence_of_functions(
        function_family,
        x_disc,
        w_disc,
        eps_comp,
        discretizer.interpolation_degree,
        intervals,
    )

    x_cheb, w_cheb, idx_cheb = construct_Chevyshev_quadratures(x_disc, w_disc, U_disc)

    r = U_disc.T @ w_disc
    if optimizer is None:
        optimizer = QuadOptimizer(u_list, r)

    x, w = optimizer.point_reduction(x_cheb, w_cheb, eps_quad)
    
    return x, w

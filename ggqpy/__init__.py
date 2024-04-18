import numpy as np
import scipy as sp
from ggqpy.discretize import *
from ggqpy.optimize import *
from ggqpy.quad import Quadrature
from ggqpy.utils import *
from ggqpy.nystrom import *


verbose = True
if verbose:

    def vprint(*messages) -> None:
        for message in messages:
            print(message)
        print()
        return

else:

    def vprint(*messages) -> None:
        return


def construct_Chebyshev_quadratures(eval_points: tuple, w, U):
    """

    Parameters
    ----------
    :
    Returns
    -------
    :
    """
    r = U.T @ w
    k = len(r)

    B = np.sqrt(w)[np.newaxis, :] * U.T
    Q, R, perm = sp.linalg.qr(B, pivoting=True)
    z = np.linalg.solve(R[:k, :k], Q.T.conj() @ r)

    idx_cheb = perm[:k]

    eval_points = tuple(map(lambda x: x[idx_cheb], eval_points))

    w_cheb = z * np.sqrt(w[idx_cheb])

    return eval_points, w_cheb


def generalized_gaussian_quadrature(
    function_family: FunctionFamily,
    min_length: float = 1e-6,
    eps_disc: float = 1e-13,
    eps_comp: float = 1e-13,
    eps_quad: float = 1e-7,
    interpolation_degree: int = 30,
    full: bool = None,
):
    """

    Parameters
    ----------
    :
    Returns
    -------
    :
    """

    vprint(
        f"Function family consists of {len(function_family.functions_lambdas)} functions"
    )

    ## Discretize
    discretizer = Discretizer(eps_disc, min_length, interpolation_degree)
    x_disc, w_disc = discretizer.adaptive_discretization(function_family)
    vprint(
        f"Adaptive discretization divided the domain into {len(discretizer.intervals)} subintervals"
    )

    ## Compress functions
    U_disc, rank = compress_sequence_of_functions(
        function_family.functions_lambdas, x_disc, w_disc, eps_comp
    )
    vprint(f"Determined numerical rank to be {rank}")

    ## Construct Chebyshev quadrature
    (x_cheb,), w_cheb = construct_Chebyshev_quadratures((x_disc,), w_disc, U_disc)
    r = U_disc.T @ w_disc

    ## Point by point reduction.
    U_family = discretizer.interpolate_piecewise_legendre(U_disc)
    optimizer = QuadOptimizer(U_family, r)
    x, w = optimizer.reduce_quadrature(x_cheb, w_cheb, eps_quad)

    if full:
        return Quadrature(x, w), Quadrature(x_disc, w_disc), Quadrature(x_cheb, w_cheb)

    return x, w

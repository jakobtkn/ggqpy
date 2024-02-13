import numpy as np
import scipy as sp
from numpy.typing import ArrayLike
import numpy.polynomial.legendre as legendre
import matplotlib.pyplot as plt

from ggqpy.functionfamiliy import PiecewiseLegendre


def construct_A_matrix(eval_points, weights, functions):
    if type(eval_points) is not tuple:
        eval_points = (eval_points,)
    
    A = np.column_stack([phi(*eval_points) * np.sqrt(weights) for phi in functions])

    return A

def compress_sequence_of_functions(functions, eval_points, weights, precision):
    
    ## Construct rank revealing QR s.t. sp.linalg.norm(A[:,perm] - Q[:,:k]@R[:k,:]) <= precision]
    A = construct_A_matrix(eval_points, weights, functions)
    Q, R, _ = sp.linalg.qr(A, pivoting=True)
    rank = np.sum(np.abs(np.diag(R)) > precision)

    U = Q[:, :rank] * (np.sqrt(weights)[:, np.newaxis]) ** (-1)
    return U, rank

def interp_legendre(U, k, intervals):
    x, _ = legendre.leggauss(2 * k)
    u_list = list()
    for u_global in U.T:
        u_local = np.split(u_global, len(intervals))
        P = list()

        for u, interval in zip(u_local, intervals):
            x, _ = legendre.leggauss(2 * k)
            coef = legendre.legfit(x, u, deg=2 * k - 1)
            p = legendre.Legendre(coef, tuple(interval))
            P.append(p)

        u_list.append(PiecewiseLegendre(P, intervals))

    return u_list


def visualise_diagonal_dropoff(A, eps_comp):
    _, R, _ = sp.linalg.qr(A, mode="economic", pivoting=True)
    plt.xlabel(r"$i$")
    plt.semilogy(np.abs(np.diag(R)), "-xr", label=r"$|R_{ii}|$")
    plt.axhline(eps_comp, linestyle="--", label=r"$\varepsilon_{comp}$")
    plt.legend()
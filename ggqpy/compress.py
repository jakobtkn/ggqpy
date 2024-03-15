import numpy as np
import scipy as sp
from numpy.typing import ArrayLike
import numpy.polynomial.legendre as legendre
import matplotlib.pyplot as plt

from ggqpy.functionfamiliy import PiecewiseLegendre, PiecewiseLegendreFamily


def construct_A_matrix(eval_points, weights, functions):
    if type(eval_points) is not tuple:
        eval_points = (eval_points,)
    
    A = np.column_stack([phi(*eval_points) * np.sqrt(weights) for phi in functions])

    return A

def compress_sequence_of_functions(functions, eval_points, weights, precision):
    
    ## Construct rank revealing QR s.t. sp.linalg.norm(A[:,perm] - Q[:,:k]@R[:k,:]) <= precision]
    A = construct_A_matrix(eval_points, weights, functions)
    Q, R, _ = sp.linalg.qr(A, pivoting=True, mode="economic")
    rank = np.sum(np.abs(np.diag(R)) > precision)

    U = Q[:, :rank] * (np.sqrt(weights)[:, np.newaxis]) ** (-1)
    return U, rank

def interp_legendre(U, endpoints):
    points_total, number_of_polynomials = U.shape
    number_of_intervals = len(endpoints) - 1
    points_per_interval = points_total//number_of_intervals
    u_list = list()
    
    x, _ = legendre.leggauss(points_per_interval)
    for n in range(number_of_polynomials):
        piecewise_poly = list()
        for i in range(number_of_intervals):
            u_local = U[points_per_interval*i:points_per_interval*(i+1), n]
            coef = np.polynomial.legendre.legfit(x, u_local, deg=points_per_interval - 1)
            p_local = legendre.Legendre(coef, (endpoints[i], endpoints[i+1]))
            piecewise_poly.append(p_local)
            
        u_list.append(PiecewiseLegendre(piecewise_poly, endpoints))
        
    return PiecewiseLegendreFamily(u_list,endpoints)


def visualise_diagonal_dropoff(A, eps_comp):
    _, R, _ = sp.linalg.qr(A, mode="economic", pivoting=True)
    plt.xlabel(r"$i$")
    plt.semilogy(np.abs(np.diag(R)), "-xr", label=r"$|R_{ii}|$")
    plt.axhline(eps_comp, linestyle="--", label=r"$\varepsilon_{comp}$")
    plt.legend()
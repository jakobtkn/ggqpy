import sys
import pandas as pd
import os
import sympy
import argparse
from numpy.polynomial.legendre import leggauss, legvander2d
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath("."))
from ggqpy import *
from ggqpy.nystrom import *
from ggqpy.parametrization import Parametrization

param = Parametrization.droplet()
rho, drho, jacobian, normal = param.get_lambdas()
verbose = False

def Phi(s, t, k=1.0, p0=np.array([10, 0, 0])):
    p = rho(s, t)
    dist = np.linalg.norm(p - p0[:, np.newaxis], axis=0)
    return np.exp(1j * k * dist) / dist


def kernel(x0, y0, s, t, k=1.0):
    q = rho(x0, y0)[:, np.newaxis]
    p = rho(s, t)
    n = normal(s, t)
    dist = np.linalg.norm(q - p, axis=0)
    return (
        (np.sum(n * (q - p), axis=0))
        / dist**3
        * np.exp(1j * k * dist)
        * (1.0 - 1j * k * dist)
    )


def main(M, N, order, k, dh, h):
    system = QuadratureLoader(order)

    A, ss, tt, ww = construct_discretization_matrix(
        Interval(0, 2 * np.pi),
        Interval(0, np.pi),
        M,
        N,
        rho,
        drho,
        kernel,
        jacobian,
        system
    )
    
    f = dh(ss, tt) * np.sqrt(ww)
    A = -0.5 * np.identity(M * N) + (4.0 * np.pi) ** (-1) * A
    q = np.linalg.solve(A, f)

    p0 = np.array([-10, 0, 0])
    target = np.array(h(*p0))
    result = (4.0 * np.pi) ** (-1) * np.sum(
        Phi(ss, tt, k, p0) * q * np.sqrt(ww)
    )
    relative_error = abs(result - target) / abs(target)
    condition_number = np.linalg.cond(A)
    if verbose:
        print("Relative error:", relative_error)
        print("Result:", result)
        print("Target:", target)
        print(condition_number)
    return relative_error, condition_number

def compute_droplet(order, wavenumber, M, N):
    order, k = int(order), float(wavenumber)
    h, h_grad = param.h_and_hgrad(k)

    def dh(s, t):
        x, y, z = rho(s, t)
        return np.sum(normal(s, t) * h_grad(x, y, z), axis=0)
    
   
    error = list()
    condition = list()
    for m, n in zip(M, N):
        err, cond = main(m, n, order, k, dh, h)
        error.append(err)
        condition.append(cond)
    return error, condition

if __name__ == "__main__":
    N = [2,3,5,8]
    M = [2*x for x in N]
    error4, condition4 = compute_droplet(4, 1, M, N)
    error16, condition16 = compute_droplet(16, 1, M, N)
    
    columns_ = [
    [
        f"\\makecell{{Relative error \\\\ $n={n}$}}",
        f"\\makecell{{Condition Number \\\\ $n={n}$}}",
    ]
    for n in [4, 16]
    ]
    columns = ["$n_x$", "$n_y$"]
    for col in columns_:
        columns = columns + col

    df = pd.DataFrame(np.column_stack([M, N, error4, condition4, error16, condition16]), columns = columns)
    styler = df.style
    styler.format_index(escape="latex")
    format = dict(
    zip(
        columns,
        ["{:.0f}", "{:.0f}", "{:.2e}", "{:.2e}", "{:.2e}", "{:.2e}"],
    )
    )
    styler.format(format)
    styler.hide(axis="index")
    latex_table = styler.to_latex(
        position_float="centering",
        position="ht",
        caption=f"\\capdroplet",
        label=f"tab:droplet-test",
        column_format="cccccc",
        hrules = True,
    )
    print(latex_table)
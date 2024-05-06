import sys
import pandas as pd
import os
import sympy
import argparse
from numpy.polynomial.legendre import leggauss, legvander2d

sys.path.append(os.path.abspath("."))
from ggqpy import *
from ggqpy.nystrom import *
from ggqpy.parametrization import Parametrization
from itertools import product
import matplotlib.pyplot as plt

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


def main(M, N, order, k):
    A, ss, tt, ww = construct_discretization_matrix(
        Interval(0, 2 * np.pi),
        Interval(0, np.pi),
        M,
        N,
        rho,
        drho,
        kernel,
        jacobian,
        order=order,
    )

    h, h_grad = param.h_and_hgrad(k)

    def dh(s, t):
        x, y, z = rho(s, t)
        return np.sum(normal(s, t) * h_grad(x, y, z), axis=0)

    f = dh(ss, tt) * np.sqrt(ww)
    A = -0.5 * np.identity(M * N) + (4.0 * np.pi) ** (-1) * A
    q = np.linalg.solve(A, f)

    p0 = np.array([-10, 5, 0])
    target = np.array(h(*p0))
    result = (4.0 * np.pi) ** (-1) * np.sum(
        Phi(ss, tt, k, p0) * q * jacobian(ss, tt) * np.sqrt(ww)
    )
    relative_error = abs(result - target) / abs(target)
    if verbose:
        print("Relative error:", relative_error)
        print("Result:", result)
        print("Target:", target)
        print(np.linalg.cond(A))
    return relative_error


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("order", default=4)
    parser.add_argument("wavenumber", default=1.0)
    args = parser.parse_args()
    order, k = int(args.order), float(args.wavenumber)

    N = [2, 3, 5, 7, 10, 15]
    M = [2 * n for n in N]
    error = list()
    for m, n in zip(M, N):
        error.append(main(m, n, order, k))

    df = pd.DataFrame(np.column_stack([M, N, error]), columns = ["$m$", "$n$", "Relative error"])
    styler = df.style
    styler.format_index(escape="latex")
    styler.format({"$m$": '{:.0f}', "$n$": '{:.0f}', "Relative error": '{:.2e}'}, na_rep='MISS')
    styler.hide(axis = "index")
    latex_table = styler.to_latex(
        position_float="centering",
        position="ht",
        caption=f"Results of droplet test with order {order} and $k={k}$",
        label=f"tab:droplet-test.{order}.{args.wavenumber}",
        column_format="ccc",
        hrules = True,
    )
    print(latex_table)
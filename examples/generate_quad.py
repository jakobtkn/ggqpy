import click
import os
import sys
import scipy as sp

sys.path.append(os.path.abspath("."))

from ggqpy import *


def generate_function_family(number_of_parameters = 16, n = 2):
    gamma = (
        lambda r0, theta0, u: r0
        * np.sin(theta0)
        / (r0 * np.sin(theta0 - theta0 * u) + np.sin(theta0 * u))
    )

    x_gl, _ = np.polynomial.legendre.leggauss(number_of_parameters)

    (amin, amax) = (1e-7, 1)
    alphas = (amax - amin) * (x_gl + 1) / 2 + amin

    (bmin, bmax) = (1e-7, np.pi)
    betas = (bmax - bmin) * (x_gl + 1) / 2 + bmin

    functions_cos = [
        lambda u, alpha=alpha, beta=beta: beta
        * gamma(alpha, beta, u) ** (i + 2)
        / (i + 2)
        * np.cos(j * beta * u)
        for i in range(-1, n + 1)
        for j in range(0, 3 * (i + 1) + 2 + 1)
        for alpha in alphas
        for beta in betas
    ]
    functions_sin = [
        lambda u, alpha=alpha, beta=beta: beta
        * gamma(alpha, beta, u) ** (i + 2)
        / (i + 2)
        * np.sin(j * beta * u)
        for i in range(-1, n + 1)
        for j in range(0, 3 * (i + 1) + 2 + 1)
        for alpha in alphas
        for beta in betas
    ]

    functions = functions_cos + functions_sin

    F = FunctionFamily(Interval(0, 1), functions)
    return F


# @click.option()
# @click.option()
def main():
    eps_disc = 1e-9
    eps_comp = eps_disc * 1e2
    eps_quad = 1e-7
    min_length = 1e-7

    disc = Discretizer(
        precision=eps_disc, min_length=min_length, interpolation_degree=10
    )
    F = generate_function_family(4,2)
    x, w = generalized_gaussian_quadrature(
        F, disc, eps_comp=eps_comp, eps_quad=eps_quad
    )
    quad = Quadrature(x, w)
    quad.save_to_file("quads/out.quad")


if __name__ == "__main__":
    main()

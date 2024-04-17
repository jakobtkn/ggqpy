import sys
import pandas as pd
import os
import sympy
import argparse

sys.path.append(os.path.abspath("."))
from ggqpy import *


def analytic_integral(alpha):
    integral = (
        -alpha
        * 2
        * (
            np.arctanh(1 / np.sqrt(alpha**2 + 1)) * alpha**4
            + np.arctanh((alpha - 1) / np.sqrt(alpha**2 + 1)) * alpha**4
            - alpha * (alpha**2 + 1) ** (3 / 2)
            + (alpha**2 + 1) ** (3 / 2)
            - np.arctanh(1 / np.sqrt(alpha**2 + 1))
            - np.arctanh((alpha - 1) / np.sqrt(alpha**2 + 1))
        )
        / (alpha**2 + 1) ** (5 / 2)
    )

    return integral


def main(alpha, discretization_level=16, order=8):
    r0 = alpha
    theta0 = np.pi / 2
    gamma = (
        lambda u: r0
        * np.sin(theta0)
        / (r0 * np.sin(theta0 - theta0 * u) + np.sin(theta0 * u))
    )
    f = lambda r, theta: np.full_like(r, np.cos(2 * theta))

    def evaluate_integrand(u):
        gammau = gamma(u)
        x, w = np.polynomial.legendre.leggauss(order)
        gl = Quadrature(gammau * (x + 1) / 2, w * gammau / 2)
        return f(gl.x, theta0 * u) @ gl.w

    quad = Quadrature.load_from_file(
        f"quads/nystrom.{discretization_level}.{order}.quad"
    )
    quad.save_to_file("quads/test.quad")

    sum = theta0 * np.array([evaluate_integrand(x) for x in quad.x]).flatten() @ quad.w

    return abs(sum.item() - analytic_integral(alpha))


def main(alpha, discretization_level=16, order=8):
    r0 = alpha
    theta0 = np.pi / 2
    f = lambda r, theta: np.cos(2 * theta)
    r, theta, w = quad_on_standard_triangle(r0, theta0)

    return abs(f(r, theta) @ w - analytic_integral(alpha))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("count", default=16)
    parser.add_argument("order", default=4)
    args = parser.parse_args()

    alpha = [0.5, 1e-4, 1e-7]
    error = list()
    for a in alpha:
        error.append(main(a, int(args.count), int(args.order)))
    df = pd.DataFrame(dict(alpha=alpha, error=error))

    latex_table = df.to_latex(
        index=False,
        header=["$\\alpha$", "Absolute error"],
        caption="Results",
        label="tab:triangle-test",
        float_format="{:.2e}".format,
        position="centering",
    )
    print(latex_table)

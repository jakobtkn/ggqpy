import sys
import pandas as pd
import os
import sympy
import argparse

sys.path.append(os.path.abspath("."))
from ggqpy import *
from ggqpy.quad import Quadrature


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

def main(alpha, order):
    r0 = alpha
    theta0 = np.pi / 2
    f = lambda r, theta: np.cos(2 * theta)
    r, theta, w = quad_on_standard_triangle(r0, theta0)

    return abs(f(r, theta) @ w - analytic_integral(alpha))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("order", default=4)
    args = parser.parse_args()

    alpha = [0.5, 1e-4, 1e-5]
    error = list()
    for a in alpha:
        error.append(main(a, int(args.order)))
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

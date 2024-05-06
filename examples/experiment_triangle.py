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
    f = lambda r, theta: np.cos(2 * theta) / r
    r, theta, w = quad_on_standard_triangle(order, r0, theta0)

    return abs(f(r, theta) @ w - analytic_integral(alpha))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("order", default=4)
    args = parser.parse_args()

    alpha = [0.5, 1e-4, 1e-5]
    error = list()
    order = int(args.order)
    for a in alpha:
        error.append(main(a, order))
    
    df = pd.DataFrame(np.column_stack([alpha, error]), columns = ["$\\alpha$", "Absolute error"])
    styler = df.style
    styler.format_index(escape="latex")
    styler.format('{:.2e}', na_rep='MISS')
    styler.hide(axis = "index")
    latex_table = styler.to_latex(
        position_float="centering",
        position="ht",
        caption=f"Results Triangle experiment of order {order}",
        label=f"tab:triangle-test.{order}",
        column_format="cc",
        hrules = True,
    )
    print(latex_table)

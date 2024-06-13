import sys
import pandas as pd
import os
import sympy
import argparse

sys.path.append(os.path.abspath("."))
from ggqpy import *
from ggqpy.nystrom import QuadratureLoader


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

    quad_loader = QuadratureLoader(order)
    r, theta, w = quad_loader.quad_on_standard_triangle(r0, theta0)
    N = len(r)
    sol = analytic_integral(alpha)

    error = abs(f(r, theta) @ w - analytic_integral(alpha))
    rel_error = error / abs(sol)

    return error, rel_error, N


if __name__ == "__main__":
    alpha = [0.9, 0.5, 0.5e-4, 0.5e-5]
    error = list()
    rel_error_4 = list()
    length_4 = list()
    rel_error_8 = list()
    length_8 = list()
    rel_error_16 = list()
    length_16 = list()

    for a in alpha:
        err, rel_err, N = main(a, 4)
        error.append(err)
        rel_error_4.append(rel_err)
        length_4.append(N)

    for a in alpha:
        err, rel_err, N = main(a, 8)
        error.append(err)
        rel_error_8.append(rel_err)
        length_8.append(N)

    for a in alpha:
        err, rel_err, N = main(a, 16)
        error.append(err)
        rel_error_16.append(rel_err)
        length_16.append(N)

    columns_ = [
        [
            f"\\makecell{{$N$ \\\\ $n={n}$}}",
            f"\\makecell{{Relative error \\\\ $n={n}$}}",
        ]
        for n in [4, 8, 16]
    ]
    columns = ["$\\alpha$"]
    for col in columns_:
        columns = columns + col

    df = pd.DataFrame(
        np.column_stack(
            [
                alpha,
                length_4,
                rel_error_4,
                length_8,
                rel_error_8,
                length_16,
                rel_error_16,
            ]
        ),
        columns=columns,
    )
    styler = df.style
    styler.format_index(escape="latex")
    format = dict(
        zip(
            columns,
            ["{:.2e}", "{:.0f}", "{:.2e}", "{:.0f}", "{:.2e}", "{:.0f}", "{:.2e}"],
        )
    )
    styler.format(format)
    styler.hide(axis="index")
    latex_table = styler.to_latex(
        position_float="centering",
        position="ht",
        label=f"tab:triangle-test",
        column_format="ccccccc",
        hrules=True,
        caption=f"\\captri",
    )
    print(latex_table)

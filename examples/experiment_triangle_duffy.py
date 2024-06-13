import sys
import pandas as pd
import os
import sympy
import argparse

sys.path.append(os.path.abspath("."))
from ggqpy import *
from ggqpy.quad import Quadrature
from ggqpy.nystrom import QuadratureLoader
from ggqpy.duffy import duffy_on_standard_triangle, make_2d_qaud_unit_square


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

def duffy_test(alpha, N):
    f = lambda r, theta: np.cos(2 * theta) / r
    sol = analytic_integral(alpha)

    uu,vv,ww = make_2d_qaud_unit_square(N)
    x,y,w = duffy_on_standard_triangle(alpha, np.pi/2.0, uu, vv, ww)
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan(y/x)
    error = abs(f(r, theta) @ w - analytic_integral(alpha))
    rel_error = error/abs(sol)

    return error, rel_error, len(uu)
    

if __name__ == "__main__":
    alpha = [0.9, 0.5, 0.03, 0.2e-2, 0.2e-5]
    error = list()
    rel_error_4 = list()
    length_4 = list()
    rel_error_8 = list()
    length_8 = list()
    rel_error_16 = list()
    length_16 = list()

    rel_error_duffy = list()
    length_duffy = list()
    for a in alpha:
        err, rel_err, N = duffy_test(a, 25)
        error.append(err)
        rel_error_duffy.append(rel_err)
        length_duffy.append(N)

    columns =  ["$\\alpha$"] + [f"$N$", f"Relative error"]

    df = pd.DataFrame(np.column_stack([alpha, length_duffy, rel_error_duffy]), columns = columns)
    styler = df.style
    styler.format_index(escape="latex")
    format = dict(zip(columns, ['{:.2e}','{:.0f}','{:.2e}']))
    styler.format(format)
    styler.hide(axis = "index")
    latex_table = styler.to_latex(
        position_float="centering",
        position="ht",
        label=f"tab:triangle-test-duffy",
        column_format="ccc",
        hrules = True,
        caption=f"\\capduffy",
    )
    print(latex_table)

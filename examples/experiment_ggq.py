from tester import Tester
import sys
import os
import numpy.polynomial.legendre as legendre
import numpy as np
import sympy as sp
import time as time
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(os.path.abspath("."))
from ggqpy import generalized_gaussian_quadrature, Discretizer, compress_sequence_of_functions, construct_Chebyshev_quadratures, QuadOptimizer, Quadrature
from ggqpy.duffy import duffy_on_standard_triangle, duffy_quad
from ggqpy.secret import secret_trick
from ggqpy.utils import Interval, FunctionFamily, FunctionFamilySymbolic
from ggqpy.nystrom import SingularTriangleQuadFinder, QuadratureLoader
from ggqpy.parametrization import Parametrization

class Timer():
    def __init__(self):
        pass
    
    def start(self):
        self.time = time.time()

    def end(self, text = None):
        print(text, time.time() - self.time)

def main():
    T = Timer()
    function_family, f, fsym = FunctionFamilySymbolic.polynomials_and_singularity(Interval(0,1), order=9, number_of_polynomials=100, perturbed=True)

    eps_disc = 1e-7
    eps_comp = 1e-6
    eps_quad = 1e-6
    min_length = 1e-16
    interpolation_degree = 7

    T.start()
    discretizer = Discretizer(eps_disc, min_length, interpolation_degree)
    x_disc, w_disc = discretizer.adaptive_discretization(function_family, priority=True)
    T.end("Discretizer:")
    Q_disc = Quadrature(x_disc,w_disc)

    ## Compress functions
    T.start()
    U_disc, rank = compress_sequence_of_functions(
        function_family.functions_lambdas, x_disc, w_disc, eps_comp
    )
    T.end("Compression:")

    ## Construct Chebyshev quadrature
    T.start()
    (x_cheb,), w_cheb = construct_Chebyshev_quadratures((x_disc,), w_disc, U_disc)
    r = U_disc.T @ w_disc
    Q_cheb = Quadrature(x_cheb,w_cheb)
    T.end("Construct Chebyshev quad:")

    ## Point by point reduction.
    T.start()
    U_family = discretizer.interpolate_piecewise_legendre(U_disc)
    optimizer = QuadOptimizer(U_family, r)
    x, w = optimizer.reduce_quadrature(x_cheb, w_cheb, eps_quad)
    Q_ggq = Quadrature(x,w)
    T.end("Point by point")
    x_gl,w_gl = legendre.leggauss(5)
    Q_gl = Quadrature((x_gl+1)/2, w_gl/2)

    print(len(Q_gl.x),len(Q_ggq.x))
    target = function_family.integral(fsym)
    rel = lambda x: abs((x-target)/target)

    nodes = [Q_disc.size, Q_cheb.size, Q_ggq.size, Q_gl.size]
    rel_error = [rel(Q_disc.eval(f)), rel(Q_cheb.eval(f)), rel(Q_ggq.eval(f)), rel(Q_gl.eval(f))]
    print(f"DISC: {rel(Q_disc.eval(f))}")
    print(f"CHEB: {rel(Q_cheb.eval(f))}")
    print(f"GGQ: {rel(Q_ggq.eval(f))}")
    print(f"GL: {rel(Q_gl.eval(f))}")

    columns = ["Quadrature", "Relative error", "Number of nodes"]
    names = ["Discretization", "\\chebyshev", "GGQ", "Gauss-Legendre"]
    df = pd.DataFrame(
        np.column_stack(
            [
                names,
                rel_error,
                nodes
            ]
        ),
        columns=columns,
    )
    styler = df.style
    styler.format_index(escape="latex")
    format = dict(
        zip(
            ["Relative error", "Number of nodes"],
            ["{:.2e}", "{:.0f}"],
        )
    )
    styler.format(format)
    styler.hide(axis="index")
    latex_table = styler.to_latex(
        position_float="centering",
        position="ht",
        label=f"tab:ggq-test",
        column_format="ccc",
        hrules=True,
        caption=f"\\capggq",
    )
    print(latex_table)







    np.savetxt("data/ggq.x",Q_ggq.x)
    np.savetxt("data/ggq.w",Q_ggq.w)
    np.savetxt("data/gl.x",Q_gl.x)
    np.savetxt("data/gl.w",Q_gl.w)


if __name__ == "__main__":
    main()
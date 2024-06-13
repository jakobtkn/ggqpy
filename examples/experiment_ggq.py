from tester import Tester
import sys
import os
import numpy.polynomial.legendre as legendre
import numpy as np
import sympy as sp

sys.path.append(os.path.abspath("."))
from ggqpy import generalized_gaussian_quadrature
from ggqpy.duffy import duffy_on_standard_triangle, duffy_quad
from ggqpy.secret import secret_trick
from ggqpy.utils import Interval, FunctionFamily, FunctionFamilySymbolic
from ggqpy.nystrom import SingularTriangleQuadFinder, QuadratureLoader
from ggqpy.parametrization import Parametrization


def main():
    F, f, fsym = FunctionFamilySymbolic.polynomials_and_singularity(Interval(0,1), order=30, number_of_polynomials=100, perturbed=True)
    Q_ggq, Q_disc, Q_cheb = generalized_gaussian_quadrature(F, 1e-7, 1e-7, 1e-6, 1e-6, 15, True)

    target = F.integral(fsym)
    rel = lambda x: abs((x-target)/target)

    print(f"DISC: {rel(Q_disc.eval(f))}")
    print(f"CHEB: {rel(Q_cheb.eval(f))}")
    print(f"GGQ: {rel(Q_ggq.eval(f))}")

if __name__ == "__main__":
    main()
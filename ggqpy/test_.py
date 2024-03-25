import numpy as np
import sympy
from pytest import approx

from ggqpy import *


def test_sherman_morrison():
    A = np.random.random(size=(5, 5))
    u = np.random.random(size=5)
    v = np.random.random(size=5)

    Ainv = np.linalg.inv(A)
    Ainv = np.asfortranarray(Ainv)

    sherman_morrison(Ainv, u, v)
    assert np.linalg.norm(Ainv - np.linalg.inv(A + np.outer(u, v))) == approx(0, 1e-11)


def test_end_to_end_polynomial():
    order = 9
    I = Interval(1e-8, 1)
    F = FunctionFamilySymbolic.polynomials_and_singularity(
        I, order=order, number_of_polynomials=30
    )
    x, w = generalized_gaussian_quadrature(
        F,
        min_length=1e-8,
        eps_disc=1e-12,
        eps_comp=1e-10,
        eps_quad=1e-7,
        interpolation_degree=15,
    )

    f_lambda, f_symbolic = F.generate_example_function()

    integral_numeric = f_lambda(x) @ w
    integral_analytic = F.integral(f_symbolic)

    assert len(x) == (order + 1) // 2
    assert integral_analytic == approx(integral_numeric)


def test_end_to_end_nystrom():
    order = 9
    F = FunctionFamily.nystrom_integral_functions(
        number_of_discretizations=5, order=order
    )
    eps_quad = 1e-7
    ggq, adap, cheb = generalized_gaussian_quadrature(
        F,
        min_length=1e-8,
        eps_disc=1e-12,
        eps_comp=1e-10,
        eps_quad=eps_quad,
        interpolation_degree=15,
        detailed_output=True
    )

    f = F.generate_example_function()

    integral_ggq = ggq.eval(f)
    integral_adap = adap.eval(f)

    assert ggq.size < cheb.size
    assert integral_ggq == approx(integral_adap, eps_quad)

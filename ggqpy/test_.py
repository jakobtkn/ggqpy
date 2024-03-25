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
    
    np.testing.assert_allclose(Ainv,np.linalg.inv(A + np.outer(u, v)))


def test_end_to_end_polynomial():
    order = 9
    I = Interval(1e-8, 1)
    function_family = FunctionFamilySymbolic.polynomials_and_singularity(
        I, order=order, number_of_polynomials=30
    )

    min_length=1e-8
    eps_disc=1e-12
    eps_comp=1e-10
    eps_quad=1e-7
    interpolation_degree=15

    discretizer = Discretizer(eps_disc, min_length, interpolation_degree)
    x_disc, w_disc = discretizer.adaptive_discretization(
        function_family
    )
    assert(sorted(discretizer.endpoints) == discretizer.endpoints)
    U_disc, rank = compress_sequence_of_functions(function_family.functions_lambdas, x_disc, w_disc, eps_comp)

    (x_cheb,), w_cheb = construct_Chebyshev_quadratures((x_disc,), w_disc, U_disc)
    r = U_disc.T @ w_disc
    U_family = discretizer.interpolate_piecewise_legendre(U_disc)
    optimizer = QuadOptimizer(U_family, r)
    x, w = optimizer.reduce_quadrature(x_cheb, w_cheb, eps_quad)


    f_lambda, f_symbolic = function_family.generate_example_function()

    integral_numeric = f_lambda(x) @ w
    integral_analytic = function_family.integral(f_symbolic)

    assert len(x) == (order + 1) // 2
    assert integral_analytic == approx(integral_numeric)


def test_end_to_end_nystrom():
    order = 4
    function_family = FunctionFamily.nystrom_integral_functions(
        number_of_discretizations=2, order=order
    )

    min_length=1e-4
    eps_disc=1e-12
    eps_comp=1e-10
    eps_quad=1e-7
    interpolation_degree=15   
    
    discretizer = Discretizer(eps_disc, min_length, interpolation_degree)
    x_disc, w_disc = discretizer.adaptive_discretization(
        function_family
    )
    assert(sorted(discretizer.endpoints) == discretizer.endpoints)
    # U_disc, rank = compress_sequence_of_functions(function_family.functions_lambdas, x_disc, w_disc, eps_comp)

    # (x_cheb,), w_cheb = construct_Chebyshev_quadratures((x_disc,), w_disc, U_disc)
    # r = U_disc.T @ w_disc
    # U_family = discretizer.interpolate_piecewise_legendre(U_disc)
    # optimizer = QuadOptimizer(U_family, r)
    # x, w = optimizer.reduce_quadrature(x_cheb, w_cheb, eps_quad)


    # f = function_family.generate_example_function()

    # integral_ggq = ggq.eval(f)
    # integral_adap = adap.eval(f)

    # assert ggq.size < cheb.size
    # assert integral_ggq == approx(integral_adap, eps_quad)


def test_piecewisepoly():
    endpoints = [-1.0,0,0.5,1.0]
    coef = [[0],[0],[1]]
    poly_list = [legendre.Legendre(coef[i], (endpoints[i], endpoints[i + 1])) for i in range(3)]
    P = PiecewiseLegendre(poly_list, endpoints)
    assert P([0.3]) == approx(0)
    assert P([0.5 + 1e-16]) == approx(1.0)
    assert P([-0.8]) == approx(0.0)

def test_piecewisepoly_family():
    discretize = Discretizer()
    discretize.endpoints = [-1.0,0,0.5,1.0]
    U = np.array([[1,1,0,0,0,0],[0,0,1,1,0,0],[0,0,0,0,1,1]]).T
    P = discretize.interpolate_piecewise_legendre(U)
    
    np.testing.assert_allclose(P([-0.8]),np.array([1,0,0])[:,np.newaxis])
    np.testing.assert_allclose(P([0.3]),np.array([0,1,0])[:,np.newaxis])
    np.testing.assert_allclose(P([0.5 + 1e-16]),np.array([0,0,1])[:,np.newaxis])


def test_optimization():
    res = lambda x: x**2
    jac = lambda x: np.array([[2*x[0],0],[0,2*x[1]]])
    y0 = np.array([1,10])
    y = dampened_gauss_newton(res, jac, y0, maxiter=1000, tol=1e-9)
    np.testing.assert_allclose(y,np.array([0,0]), atol=1e-7)
    
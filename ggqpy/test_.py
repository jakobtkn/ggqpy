import numpy as np
import sympy
from pytest import approx
from ggqpy import *

def test_interval():
    I = Interval(0.3,0.4)
    x = np.linspace(-1,1,100)
    y = I.translate(x)
    assert max(y) <= 0.4
    assert min(y) >= 0.3

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
        I, order=order, number_of_polynomials=40
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
    np.testing.assert_allclose(U_family(x_disc),U_disc.T)

    optimizer = QuadOptimizer(U_family, r)
    x, w = optimizer.reduce_quadrature(x_cheb, w_cheb, eps_quad)
    

    f_lambda, f_symbolic = function_family.generate_example_function()

    integral_numeric = f_lambda(x) @ w
    integral_analytic = function_family.integral(f_symbolic)

    assert len(x) == (order + 1) // 2
    assert integral_analytic == approx(integral_numeric)

    symx = sympy.Symbol("x", real=True)
    f = 1/symx
    assert (1/x)@w == approx(function_family.integral(f))

def test_end_to_end_nystrom(plt):
    order = 4
    function_family = FunctionFamily.nystrom_integral_functions(
        number_of_discretizations=8, order=order
    )

    min_length=1e-5
    eps_disc=1e-12
    eps_comp=1e-10
    eps_quad=1e-8
    interpolation_degree=15   
    
    discretizer = Discretizer(eps_disc, min_length, interpolation_degree)
    x_disc, w_disc = discretizer.adaptive_discretization(
        function_family
    )
    adap = Quadrature(x_disc,w_disc)

    assert(sorted(discretizer.endpoints) == discretizer.endpoints)
    np.testing.assert_allclose(x_disc, sorted(x_disc))

    U_disc, rank = compress_sequence_of_functions(function_family.functions_lambdas, x_disc, w_disc, eps_comp)

    (x_cheb,), w_cheb = construct_Chebyshev_quadratures((x_disc,), w_disc, U_disc)
    cheb = Quadrature(x_cheb,w_cheb)
    r = U_disc.T @ w_disc
    U_family = discretizer.interpolate_piecewise_legendre(U_disc)

    print(U_family.number_of_functions)
    xx = np.linspace(0.0,1.0)
    n = 0
    y = U_family(x_disc)[n,:]
    plt.plot(sorted(x_disc), U_disc[:,n],"*")
    plt.plot(x_disc,y, "--")
    for e in discretizer.endpoints:
        plt.axvline(e, color = "r")

    np.testing.assert_allclose(U_family(x_disc),U_disc.T)
    optimizer = QuadOptimizer(U_family, r)
    x, w = optimizer.reduce_quadrature(x_cheb, w_cheb, eps_quad)
    ggq = Quadrature(x,w)

    f = function_family.generate_example_function()
    integral_cheb = cheb.eval(f)
    integral_ggq = ggq.eval(f)
    integral_adap = adap.eval(f)


    assert integral_cheb == approx(integral_adap, eps_comp)
    assert integral_ggq == approx(integral_adap, eps_quad)
    assert ggq.size < cheb.size


def test_piecewisepoly():
    n = 5
    endpoints = [-1.0,0,0.5,1.0]
    coef = [[0],[0],[1]]
    poly_list = [legendre.Legendre(coef[i], (endpoints[i], endpoints[i + 1])) for i in range(3)]
    P = PiecewiseLegendre(poly_list, endpoints)
    assert P([0.3]) == approx(0)
    assert P([0.5 + 1e-16]) == approx(1.0)
    assert P([-0.8]) == approx(0.0)


    x,_ = np.polynomial.legendre.leggauss(n)
    x = Interval(0.0, 0.5).translate(x)
    f = lambda x: np.sin(x) + np.exp(x)
    u = PiecewiseLegendre.interpolate_gauss_legendre_points(f(x), [0.0,0.5])
    np.testing.assert_allclose(u(x), f(x))
    

def test_piecewise_poly_2():
    n = 10
    x,_ = np.polynomial.legendre.leggauss(n)
    endpoints = [-2.0,1.6,3.0,3.2]
    interval1 = Interval(-2.0,1.6)
    interval2 = Interval(1.6,3.0)
    interval3 = Interval(3.0,3.2)
    x1 = interval1.translate(x)
    x2 = interval2.translate(x)
    x3 = interval3.translate(x)
    x = np.concatenate((x1,x2,x3))
    print(x)
    
    y1 = np.sin(x1)
    y2 = 1/x2
    y3 = np.full_like(x3,np.pi)
    y = np.concatenate((y1,y2,y3))
    print(y)
    
    u = PiecewiseLegendre.interpolate_gauss_legendre_points(y, endpoints)
    np.testing.assert_allclose(u(x), y)


def test_piecewisepoly_family():
    discretize = Discretizer()
    discretize.endpoints = [-1.0,0,0.5,1.0]
    U = np.array([[1,1,0,0,0,0],[0,0,1,1,0,0],[0,0,0,0,1,1]]).T
    P = discretize.interpolate_piecewise_legendre(U)
    
    np.testing.assert_allclose(P([-0.8]),np.array([1,0,0])[:,np.newaxis])
    np.testing.assert_allclose(P([0.3]),np.array([0,1,0])[:,np.newaxis])
    np.testing.assert_allclose(P([0.5 + 1e-16]),np.array([0,0,1])[:,np.newaxis])

def test_piecewisepoly_family_2(plt):
    n = 10
    discretize = Discretizer()
    discretize.endpoints = [-1.0,0.5,3.0]

    x_gl,_ = np.polynomial.legendre.leggauss(n)
    x1 = Interval(-1,0.5).translate(x_gl)
    x2 = Interval(0.5,3.0).translate(x_gl)
    x = np.concatenate((x1,x2))

    y1 = np.sin(100*x1)
    y2 = np.full(n, np.pi)
    ya = np.concatenate((y1,y2))


    y1 = np.sin(10*x1)
    y2 = np.full(n, -np.pi)
    yb = np.concatenate((y1,y2))

    U = np.column_stack((ya,yb))
    P = discretize.interpolate_piecewise_legendre(U)
    np.testing.assert_allclose(P(x), U.T, atol = 2e-12)

    xx = np.linspace(-1.0,3.0)
    plt.figure()
    plt.plot(x,U[:,0],"*b")
    plt.plot(xx,P(xx)[0,:],"--b")
    plt.plot(x,U[:,1],"*r")
    plt.plot(xx,P(xx)[1,:], "--r")

def test_optimization():
    res = lambda x: x**2
    jac = lambda x: np.array([[2*x[0],0],[0,2*x[1]]])
    y0 = np.array([1,10])
    y = dampened_gauss_newton(res, jac, y0, maxiter=1000, tol=1e-9)
    np.testing.assert_allclose(y,np.array([0,0]), atol=1e-7)
    
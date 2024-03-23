import numpy as np
import sympy 
from pytest import approx

from ggqpy import *

def test_sherman_morrison():
    A = np.random.random(size=(5,5))
    u = np.random.random(size=5)
    v = np.random.random(size=5)

    Ainv = np.linalg.inv(A)
    Ainv = np.asfortranarray(Ainv)

    sherman_morrison(Ainv,u,v)
    assert(np.linalg.norm(Ainv - np.linalg.inv(A + np.outer(u,v))) == approx(0))

def test_end_to_end_pylonimal():
    I = Interval(1e-9,1)
    F = FunctionFamilySymbolic.polynomials_and_singularity(I)
    x,w = generalized_gaussian_quadrature(F)
    
    f_symbolic, f_lambda = F.draw_function()

    integral_numeric = f_lambda(x)@w
    integral_analytic = F.integral(f_symbolic)
    assert(integral_numeric == approx(integral_analytic))
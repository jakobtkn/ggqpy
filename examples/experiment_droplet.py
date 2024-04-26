import sys
import pandas as pd
import os
import sympy
import argparse
from numpy.polynomial.legendre import leggauss, legvander2d

sys.path.append(os.path.abspath("."))
from ggqpy import *
from ggqpy.nystrom import *
from ggqpy.parametrization import Parametrization
from itertools import product
import matplotlib.pyplot as plt

param = Parametrization.droplet()
rho, drho, jacobian, normal = param.get_lambdas()
k = 1.0
def kernel(x0,y0, s, t):
    q = rho(x0,y0)
    p = rho(s,t)
    n = normal(s,t)
    dist = np.linalg.norm(p - q[:, np.newaxis], axis=0)
    return (
       ( (1 / (4 * np.pi))
        * (np.sum(n * (p - q[:, np.newaxis]), axis=0))
        / dist ** 3) * np.exp(1j*k*dist)*(1.0 - 1j*k*dist)
    )

M = 5
N = 5
A, ss, tt, ww = construct_discretization_matrix(Interval(0,2*np.pi), Interval(0,np.pi), M, N, rho, drho, kernel, jacobian)
print(A@ww)
h, dh = param.directional_derivative_h()

f = dh(ss,tt)
A = A - 0.5 *np.identity(M*N)
q = np.linalg.solve(A,f)
print(q)
print(np.linalg.cond(A))
import sys
import pandas as pd
import os
import sympy
import argparse
from numpy.polynomial.legendre import leggauss, legvander2d

sys.path.append(os.path.abspath("."))
from ggqpy import *
from ggqpy.nystrom import *
from ggqpy.parametrization import parametrize_droplet
from itertools import product
import matplotlib.pyplot as plt

rho, drho, jacobian, normal, h = parametrize_droplet()
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
A = construct_discretization_matrix(Interval(0,2*np.pi), Interval(0,np.pi), M, N, rho, drho, kernel, jacobian)
A = A - 0.5 *np.identity(M*N)
print(np.linalg.cond(A))
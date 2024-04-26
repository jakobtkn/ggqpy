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


def kernel(x0, y0, s, t):
    q = rho(x0, y0)[:, np.newaxis]
    p = rho(s, t)
    n = normal(s, t)
    dist = np.linalg.norm(q - p, axis=0)
    return (
        (np.sum(n * (q - p), axis=0))
        / dist**3
        * np.exp(1j * k * dist)
        * (1.0 - 1j * k * dist)
    )


M = 18
N = 10
A, ss, tt, ww = construct_discretization_matrix(
    Interval(0, 2 * np.pi),
    Interval(0, np.pi),
    M,
    N,
    rho,
    drho,
    kernel,
    jacobian,
    order=8,
)
h, h_grad = param.h_and_hgrad()


def dh(s, t):
    x, y, z = rho(s, t)
    return np.sum(h(x, y, z) * h_grad(x, y, z), axis=0)


f = dh(ss, tt)
A = - 0.5 * np.diag(1/np.sqrt(ww)) + (4.0 * np.pi) ** (-1) * A
q = np.linalg.solve(A, f)

p0 = np.array([10, 20, 0])
def single_layer(s, t):
    p = rho(s, t)
    dist = np.linalg.norm(p - p0[:, np.newaxis], axis=0)
    return np.exp(1j * k * dist) / dist

target = np.array(h(*p0))
result = (4.0 * np.pi) ** (-1) * np.sum(single_layer(ss, tt) * q * jacobian(ss, tt) * np.sqrt(ww))

print("Relative error:", abs(result - target) / abs(target))
print("Result:", result)
print("Target:", target)

# print(q)
# print(h(*rho(ss, tt)))

print(np.linalg.cond(A))

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
    dist = np.linalg.norm(p - q, axis=0)
    return (
        (np.sum(n * (p - q), axis=0))
        / dist**3
        * np.exp(1j * k * dist)
        * (1.0 - 1j * k * dist)
    )


M = 7
N = 5
A, ss, tt, ww = construct_discretization_matrix(
    Interval(0, 2 * np.pi), Interval(0, np.pi), M, N, rho, drho, kernel, jacobian
)
h, h_grad = param.h_and_hgrad()


def dh(s, t):
    x, y, z = rho(s, t)
    return np.sum(h(x, y, z) * h_grad(x, y, z), axis=0)


f = dh(ss, tt)
A = -0.5 * np.identity(M * N) + (1.0 / 4.0 * np.pi) * A
q = np.linalg.solve(A, f) / np.sqrt(ww)

p0 = np.array([10, 0, 0])


def double_layer(s, t):
    p = rho(s, t)
    n = normal(s, t)
    dist = np.linalg.norm(p - p0[:, np.newaxis], axis=0)
    return (1 / (4 * np.pi)) * (np.sum(n * (p - p0[:, np.newaxis]), axis=0)) / (dist**3)

print(np.sum(ww))

target = np.array(h(10, 0, 0))
result = np.sum(double_layer(ss, tt) * q * jacobian(ss, tt) * ww)

print("Relative error:", abs(result - target) / abs(target))
print("Result:", result)
print("Target:", target)

# print(q)
# print(h(*rho(ss, tt)))

print(np.linalg.cond(A))

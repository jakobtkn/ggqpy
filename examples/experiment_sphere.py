import sys
import pandas as pd
import os
import sympy
import argparse
from numpy.polynomial.legendre import leggauss, legvander2d

sys.path.append(os.path.abspath("."))
from ggqpy import *
from ggqpy.nystrom import *
from itertools import product
import matplotlib.pyplot as plt

M = 6
N = 10
s, ws = leggauss(M)
t, wt = leggauss(N)

gls = Quadrature.gauss_legendre_on_interval(M, Interval(0, 2 * np.pi))
glt = Quadrature.gauss_legendre_on_interval(N, Interval(0, np.pi))
s, ws = gls.x, gls.w
t, wt = glt.x, glt.w

rho = lambda s, t: np.array([np.sin(t) * np.cos(s), np.sin(t) * np.sin(s), np.cos(t)])
drho = lambda s, t: np.array(
    [
        [-np.sin(t) * np.sin(s), np.cos(t) * np.cos(s)],
        [np.sin(t) * np.cos(s), np.cos(t) * np.sin(s)],
        [0, -np.sin(t)],
    ]
)


def normal(p):
    return p / np.linalg.norm(p, axis=0)


def kernel(q, p):
    n = normal(p)
    return (
        (1 / (4 * np.pi))
        * (np.sum(n * (p - q[:, np.newaxis]), axis=0))
        / np.linalg.norm(p - q[:, np.newaxis], axis=0) ** 3
    )


def jacobian(s, t):
    return np.sin(t)

A = np.zeros(shape=(N * M, N * M))
simplex = Rectangle((0, 0), (2 * np.pi, 0), (2 * np.pi, np.pi), (0, np.pi))

ss, tt = np.meshgrid(s, t)
ss, tt = ss.flatten(), tt.flatten()
wws, wwt = np.meshgrid(ws, wt)
ww = (wws * wwt).flatten()

Vin = np.linalg.inv(legvander2d(ss, tt, [M - 1, N - 1]))

for idx, singularity in enumerate(zip(ss, tt)):
    xs, yt, w = singular_integral_quad(drho, np.array([*singularity]), simplex)
    Vout = legvander2d(xs, yt, [M - 1, N - 1])
    K = w * kernel(rho(*singularity), rho(xs, yt)) * jacobian(xs, yt)
    A[idx, :] = K @ (Vout @ Vin)

print(A.max(), A.min(), A.mean())
print(np.linalg.cond(A * np.sqrt(ww[:, np.newaxis])))
plt.imshow(A)
plt.colorbar()
plt.show()

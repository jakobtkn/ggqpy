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

M = 4
N = 10
s, _ = leggauss(M)
t, _ = leggauss(N)
s = Interval(0, 2 * np.pi).translate(s)
t = Interval(0, np.pi).translate(t)

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
    return (1/(4*np.pi))*(np.sum(n*(p - q[:, np.newaxis]), axis=0)) / np.linalg.norm(p - q[:, np.newaxis], axis=0) ** 4 ##TODO: WHY 4


def jacobian(s, t):
    return np.sin(t)


x_global = np.array([])
y_global = np.array([])
z_global = np.array([])
A = np.zeros(shape=(N * M, N * M))
simplex = Rectangle((0, 0), (2 * np.pi, 0), (2 * np.pi, np.pi), (0, np.pi))

ss,tt = np.meshgrid(s,t)
ss,tt = ss.flatten(), tt.flatten()

Vin = legvander2d(ss, tt, [M - 1, N - 1])

for idx, singularity in enumerate(zip(ss,tt)):
    xs, yt, w = singular_integral_quad(drho, np.array([*singularity]), simplex)

    Vout = legvander2d(xs, yt, [M - 1, N - 1])
    K = w * kernel(rho(*singularity), rho(xs, yt)) * jacobian(xs, yt)
    A[idx, :] = K @ (Vout @ np.linalg.inv(Vin))


# fig = plt.figure(figsize=(10, 7))
# ax = plt.axes(projection="3d")
# # plt.show(

print(A.max(), A.min(), A.mean())
print(np.linalg.cond(A))
plt.imshow(A)
plt.colorbar()
plt.show()
# # Creating plot

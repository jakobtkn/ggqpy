from tester import Tester
import sys
import pandas as pd
import os
import sympy
import argparse
import numpy.polynomial.legendre as legendre
import matplotlib.pyplot as plt
import numpy as np


sys.path.append(os.path.abspath("."))
from itertools import product
from ggqpy.geometry import Rectangle
from ggqpy.duffy import duffy_on_standard_triangle, duffy_quad
from ggqpy.secret import secret_trick
from ggqpy.utils import Interval
from ggqpy.quad import SingularTriangleQuadrature
from ggqpy.nystrom import SingularTriangleQuadFinder, QuadratureLoader

def make_2d_quad(n):
    x_gl, w_gl = legendre.leggauss(n)
    x, y = np.meshgrid(x_gl, x_gl)
    xx = x.flatten()
    yy = y.flatten()
    wx, wy = np.meshgrid(w_gl, w_gl)
    ww = (wx * wy).flatten()
    return xx, yy, ww

def rho(s, t):
    return np.row_stack([s, t])


def drho(s, t):
    return np.array([[1, 0], [0, 1], [0, 0]])


def kernel(s0, t0, s, t, k=1.0):
    q = rho(s0, t0)
    p = rho(s, t)
    dist = np.linalg.norm(q[:, np.newaxis, :] - p[:, :, np.newaxis], axis=0)
    return np.exp(1j * k * dist) / dist

def kernel0(s0, t0, s, t, k=1.0):
    q = rho(s0, t0)
    p = rho(s, t)
    dist = np.linalg.norm(q - p, axis=0)
    return np.exp(1j * k * dist) / dist

def naive(uu, vv, rr, N):
    ss, tt, ww = make_2d_quad(N)
    z = np.sum(
        kernel(uu, vv, ss, tt) * (rr[np.newaxis, :] * ww[:, np.newaxis]),
        axis=(0, 1),
    )
    nodes = len(uu)*len(ss)
    return z, nodes

quadrature_precomputed = QuadratureLoader(4)
simplex = Rectangle(Interval(-1, 1), Interval(-1, 1))
def ggq(uu, vv, rr):
    z = 0.0
    nodes = 0
    for u, v, r in zip(uu, vv, rr):
        ss, tt, ww = quadrature_precomputed.singular_integral_quad(drho, np.array([u, v]), simplex)
        nodes += len(ss)
        z += r*(kernel0(u, v, ss, tt) @ ww)
    
    return z, nodes

quad_genarator = SingularTriangleQuadFinder(4)
quadrature_compute = QuadratureLoader(4, quad_genarator)
def ggq_compute(uu, vv, rr):
    z = 0.0
    nodes = 0
    for u, v, r in zip(uu, vv, rr):
        ss, tt, ww = quadrature_compute.singular_integral_quad(drho, np.array([u, v]), simplex)
        nodes += len(ss)
        z += r*(kernel0(u, v, ss, tt) @ ww)
    
    return z, nodes


def duffy(uu, vv, rr, N):
    z = 0
    nodes = 0
    for u, v, r in zip(uu, vv, rr):
        ss, tt, ww = duffy_quad(drho, np.array([u, v]), simplex, int(N//2))
        nodes += len(ss)
        z += r*(kernel0(u, v, ss, tt) @ ww)

    return z, nodes

if __name__ == "__main__":
    N = 30
    uu, vv, rr = make_2d_quad(N + 1)
    target,_ = secret_trick(uu, vv, rr, N, kernel0)

    NN = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    tester_naive = Tester(naive, "naive")
    tester_ggq = Tester(ggq, "ggq_precomputed")
    tester_duffy = Tester(duffy, "duffy")
    tester_trick = Tester(secret_trick, "ticra")
    tester_ggq_compute = Tester(ggq_compute, "ggq_exact_triangle")
    testers: list[Tester] = [tester_naive, tester_ggq,tester_duffy,tester_trick,tester_ggq_compute]

    for N in NN:
        uu, vv, rr = make_2d_quad(N + 1)
        tester_naive.perform_test(uu,vv,rr,N)
        tester_ggq.perform_test(uu,vv,rr)
        tester_duffy.perform_test(uu,vv,rr,N)
        tester_trick.perform_test(uu,vv,rr,N, kernel0)

    # for N in [2,4,6,8,12,16]:
    #     uu, vv, rr = make_2d_quad(N + 1)
    #     tester_ggq_compute.perform_test(uu,vv,rr)

    for tester in testers:
        tester.compute_error(target)
        tester.save()

    # tester_naive.compute_abs_error(target)
    # tester_ggq.compute_abs_error(target)
    # tester_duffy.compute_abs_error(target)
    # tester_trick.compute_abs_error(target)
    
    # df = pd.DataFrame({"naive":tester_naive})

    # for tester in testers:
    #     tester.compute_abs_error(target)
    #     plt.semilogy(tester.num_nodes, tester.error, "-*", label=tester.name)
    # plt.xlim((1000,50000))
    # plt.legend()
    # plt.savefig("output/mom.png")
    # plt.show()
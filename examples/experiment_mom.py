import sys
import pandas as pd
import os
import sympy
import argparse
import numpy.polynomial.legendre as legendre
import matplotlib.pyplot as plt


sys.path.append(os.path.abspath("."))
from ggqpy import *
from itertools import product
from ggqpy.duffy import *
from ggqpy.secret import mysterious_substitution, secret_trick


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

solver = IntegralOperator(4)
simplex = Rectangle(Interval(-1, 1), Interval(-1, 1))
def ggq(uu, vv, rr):
    z = 0.0
    nodes = 0
    for u, v, r in zip(uu, vv, rr):
        ss, tt, ww = solver.singular_integral_quad(drho, np.array([u, v]), simplex)
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
    N = 60
    uu, vv, rr = make_2d_quad(N + 1)
    target,_ = secret_trick(uu, vv, rr, N, kernel0)

    num_nodes_naive = list()
    error_naive = list()
    z_naive = list()
    num_nodes_ggq = list()
    error_ggq = list()
    z_ggq = list()
    num_nodes_duffy = list()
    error_duffy = list()
    z_duffy = list()
    num_nodes_trick = list()
    error_trick = list()
    z_trick = list()
    NN = [2,3,4,5,6,7,8,9,10,11,12,14, 15, 16]
    

    for N in NN:
        uu, vv, rr = make_2d_quad(N + 1)
        
        ## NAIVE
        z, nodes = naive(uu, vv, rr, N)
        num_nodes_naive.append(nodes)
        z_naive.append(z)
        
        ## GGQP
        z, nodes = ggq(uu, vv, rr)
        num_nodes_ggq.append(nodes)
        z_ggq.append(z)

        ## DUFFY
        z, nodes = duffy(uu, vv, rr, N)
        num_nodes_duffy.append(nodes)
        z_duffy.append(z)
        
        ## TRICK
        z, nodes = secret_trick(uu, vv, rr, N, kernel0)
        num_nodes_trick.append(nodes)
        z_trick.append(z)

    error_naive = abs(np.array(z_naive) - target)
    error_ggq = abs(np.array(z_ggq) - target)
    error_duffy = abs(np.array(z_duffy) - target)
    error_trick = abs(np.array(z_trick) - target)
    
    print(z_naive[-1],z_ggq[-1],z_duffy[-1])
    plt.semilogy(num_nodes_naive, error_naive, "-*", label="naive")
    plt.semilogy(num_nodes_ggq, error_ggq, "-*", label="ggq")
    plt.semilogy(num_nodes_duffy, error_duffy, "-*", label="duffy")
    plt.semilogy(num_nodes_trick, error_trick, "-*", label="trick")
    plt.xlim((0,5e4))
    plt.legend()
    plt.show()
import sys
import pandas as pd
import os
import sympy
import argparse
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath("."))
from ggqpy import *
from ggqpy.quad import Quadrature
from ggqpy.duffy import duffy_quad
from ggqpy.secret import mysterious_substitution
from experiment_mom import make_2d_quad, grid

if __name__ == "__main__":
    plt.figure()
    N = 14
    uu, vv, rr = make_2d_quad(N + 1)
    u = uu[len(uu)//2]
    v = vv[len(vv)//3]

    plt.subplot(2,2,1)
    ss, tt, ww = make_2d_quad(N)
    plt.scatter(ss,tt,c = "b")
    plt.scatter(u,v, c = "r", marker='x')
    plt.title(f"naive {len(ss)}")

    plt.subplot(2,2,2)
    solver = IntegralOperator(4)
    def drho(s, t):
        return np.array([[1, 0], [0, 1], [0, 0]])
    simplex = Rectangle(Interval(-1, 1), Interval(-1, 1))
    ss, tt, ww = solver.singular_integral_quad(drho, np.array([u, v]), simplex)
    plt.scatter(ss,tt,c = "b")
    plt.scatter(u,v, c = "r", marker='x')
    plt.title(f"ggq {len(ss)}")

    plt.subplot(2,2,3)
    ss, tt, ww = duffy_quad(drho, np.array([u, v]), simplex, int(N//2))
    plt.scatter(ss,tt,c = "b")
    plt.scatter(u,v, c = "r", marker='x')
    plt.title(f"duffy {len(ss)}")

    plt.subplot(2,2,4)
    GA, GW = legendre.leggauss(N)
    s, ws = mysterious_substitution(u, GA, GW)
    t, wt = mysterious_substitution(v, GA, GW)
    ss, tt, ww = grid(s,t,ws,wt)

    plt.scatter(ss,tt,c = "b")
    plt.scatter(u,v, c = "r", marker='x')
    plt.title(f"trick {len(ss)}")
    plt.show()
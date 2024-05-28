from tester import Tester
import sys
import os
import numpy.polynomial.legendre as legendre
import numpy as np

sys.path.append(os.path.abspath("."))
from ggqpy.geometry import Rectangle
from ggqpy.duffy import duffy_on_standard_triangle, duffy_quad
from ggqpy.secret import secret_trick
from ggqpy.utils import Interval
from ggqpy.nystrom import SingularTriangleQuadFinder, QuadratureLoader
from ggqpy.parametrization import Parametrization


def make_2d_quad(n):
    x_gl, w_gl = legendre.leggauss(n)
    x, y = np.meshgrid(x_gl, x_gl)
    xx = x.flatten()
    yy = y.flatten()
    wx, wy = np.meshgrid(w_gl, w_gl)
    ww = (wx * wy).flatten()
    return xx, yy, ww


quadrature_precomputed = QuadratureLoader(4)
simplex = Rectangle(Interval(-1, 1), Interval(-1, 1))
quad_genarator = SingularTriangleQuadFinder(4)
quadrature_compute = QuadratureLoader(4, quad_genarator)


def run_experiment(N_test, folder, param: Parametrization, f, g):
    rho, drho, jacobian, normal = param.get_lambdas()

    def kernel(s0, t0, s, t, k=1.0):
        q = rho(s0, t0)
        p = rho(s, t)
        dist = np.linalg.norm(q[:, np.newaxis] - p, axis=0)
        return np.exp(1j * k * dist) / dist

    def integrand(s0, t0, s, t):
        return f(s0, t0) * (jacobian(s, t) * g(s, t)) * kernel(s0, t0, s, t)

    def naive(uu, vv, rr, N):
        ss, tt, ww = make_2d_quad(N)
        z = 0.0
        nodes = 0
        for u, v, r in zip(uu, vv, rr):
            nodes += len(ss)
            z += r * integrand(u, v, ss, tt) @ ww

        return z, nodes

    def ggq(uu, vv, rr):
        z = 0.0
        nodes = 0
        for u, v, r in zip(uu, vv, rr):
            ss, tt, ww = quadrature_precomputed.singular_integral_quad(
                drho, np.array([u, v]), simplex
            )
            nodes += len(ss)
            z += r * (integrand(u, v, ss, tt) @ ww)

        return z, nodes

    def ggq_compute(uu, vv, rr):
        z = 0.0
        nodes = 0
        for u, v, r in zip(uu, vv, rr):
            ss, tt, ww = quadrature_compute.singular_integral_quad(
                drho, np.array([u, v]), simplex
            )
            nodes += len(ss)
            z += r * (integrand(u, v, ss, tt) @ ww)

        return z, nodes

    def duffy(uu, vv, rr, N):
        z = 0
        nodes = 0
        for u, v, r in zip(uu, vv, rr):
            ss, tt, ww = duffy_quad(drho, np.array([u, v]), simplex, int(N // 2))
            nodes += len(ss)
            z += r * (integrand(u, v, ss, tt) @ ww)

        return z, nodes

    uu, vv, rr = make_2d_quad(N_test + 1)
    rr = rr * jacobian(uu, vv)
    target, _ = secret_trick(uu, vv, rr, N_test, integrand)

    NN = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    tester_naive = Tester(naive, "naive")
    tester_ggq = Tester(ggq, "ggq_precomputed")
    tester_duffy = Tester(duffy, "duffy")
    tester_trick = Tester(secret_trick, "ticra")
    tester_ggq_compute = Tester(ggq_compute, "ggq_exact_triangle")
    testers: list[Tester] = [
        tester_naive,
        tester_ggq,
        tester_duffy,
        tester_trick,
        tester_ggq_compute,
    ]

    for N in NN:
        uu, vv, rr = make_2d_quad(N + 1)
        rr = rr * jacobian(uu, vv)

        tester_naive.perform_test(uu, vv, rr, N)
        tester_ggq.perform_test(uu, vv, rr)
        tester_duffy.perform_test(uu, vv, rr, N)
        tester_trick.perform_test(uu, vv, rr, N, integrand)

    for N in [2, 4, 6, 8, 12, 16]:
        uu, vv, rr = make_2d_quad(N + 1)
        rr = rr * jacobian(uu, vv)
        tester_ggq_compute.perform_test(uu, vv, rr)

    for tester in testers:
        tester.compute_error(target)
        tester.save(folder)


if __name__ == "__main__":
    N_test = 130

    param = Parametrization.plane()

    rho, drho, jacobian, normal = param.get_lambdas()

    def f(s, t):
        return np.full_like(s, 1.0)

    def g(s, t):
        return np.full_like(s, 1.0)

    run_experiment(N_test, "simple_patch", param, f, g)

    param = Parametrization.complicated_patch()

    rho, drho, jacobian, normal = param.get_lambdas()

    def f(s, t):
        cs = [0,0,0,1,0]
        ct = [0,0,0,1,0]
        return legendre.legval(s, cs)*legendre.legval(t, ct)

    def g(s, t):
        cs = [0,0,0,1,0]
        ct = [0,1,0,0,0]
        return legendre.legval(s, cs)*legendre.legval(t, ct)

    run_experiment(N_test, "high_order", param, f, g)

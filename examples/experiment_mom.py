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


def make_2d_quad(n, jacobian):
    x_gl, w_gl = legendre.leggauss(n)
    x, y = np.meshgrid(x_gl, x_gl)
    xx = x.flatten()
    yy = y.flatten()
    wx, wy = np.meshgrid(w_gl, w_gl)
    ww = (wx * wy).flatten()
    ww = ww * jacobian(xx, yy)

    return xx, yy, ww


quadrature_precomputed_order_4 = QuadratureLoader(4)
quadrature_precomputed_order_8 = QuadratureLoader(8)
quadrature_precomputed_order_16 = QuadratureLoader(16)
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
        return f(s0, t0) * g(s, t) * kernel(s0, t0, s, t)

    def naive(uu, vv, rr, N):
        ss, tt, ww = make_2d_quad(N, jacobian)
        z = 0.0
        nodes = 0
        for u, v, r in zip(uu, vv, rr):
            nodes += len(ss)
            z += r * integrand(u, v, ss, tt) @ ww

        return z, nodes

    def ggq(uu, vv, rr, loader):
        z = 0.0
        nodes = 0
        for u, v, r in zip(uu, vv, rr):
            ss, tt, ww = loader.singular_integral_quad(
                drho, np.array([u, v]), simplex
            )
            ww = jacobian(ss,tt)*ww
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
            ww = jacobian(ss,tt)*ww
            nodes += len(ss)
            z += r * (integrand(u, v, ss, tt) @ ww)

        return z, nodes

    def duffy(uu, vv, rr, N):
        z = 0
        nodes = 0
        for u, v, r in zip(uu, vv, rr):
            ss, tt, ww = duffy_quad(drho, np.array([u, v]), simplex, int(N // 2))
            ww = jacobian(ss,tt)*ww
            nodes += len(ss)
            z += r * (integrand(u, v, ss, tt) @ (ww))

        return z, nodes

    uu, vv, rr = make_2d_quad(N_test + 1, jacobian)
    target, _ = secret_trick(uu, vv, rr, N_test, integrand, jacobian)

    NN = [6, 8, 10, 12, 14, 16, 18, 20, 22]
    tester_naive = Tester(naive, "naive")
    tester_ggq4  = Tester(ggq, "ggq_precomputed_4")
    tester_ggq8  = Tester(ggq, "ggq_precomputed_8")
    tester_ggq16 = Tester(ggq, "ggq_precomputed_16")
    tester_duffy = Tester(duffy, "duffy")
    tester_trick = Tester(secret_trick, "ticra")
    tester_ggq_compute = Tester(ggq_compute, "ggq_exact_triangle")
    testers: list[Tester] = [
        tester_naive,
        tester_ggq4,
        tester_ggq8,
        tester_ggq16,
        tester_duffy,
        tester_trick,
        tester_ggq_compute,
    ]

    for N in NN:
        uu, vv, rr = make_2d_quad(N + 1, jacobian)

        tester_naive.perform_test(uu, vv, rr, N)
        tester_ggq4.perform_test(uu, vv, rr, quadrature_precomputed_order_4)
        tester_ggq8.perform_test(uu, vv, rr, quadrature_precomputed_order_8)
        tester_ggq16.perform_test(uu, vv, rr, quadrature_precomputed_order_16)
        tester_duffy.perform_test(uu, vv, rr, N)
        tester_trick.perform_test(uu, vv, rr, N, integrand, jacobian)

    for N in [2, 4, 6, 8, 12, 16]:
        uu, vv, rr = make_2d_quad(N + 1, jacobian)
        tester_ggq_compute.perform_test(uu, vv, rr)

    for tester in testers:
        tester.compute_error(target)
        tester.save(folder)


if __name__ == "__main__":
    N_test = 220

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
        cs = [0,1,0,0]
        ct = [1,0,0,0]
        return legendre.legval(s, cs)*legendre.legval(t, ct)

    def g(s, t):
        cs = [1,0,0,0]
        ct = [0,1,0,0]
        return legendre.legval(s, cs)*legendre.legval(t, ct)

    run_experiment(N_test, "high_order", param, f, g)

from __future__ import annotations
import numpy as np
from math import isclose
from ggqpy.geometry import Quadrilateral, Rectangle, standard_radial_triangle_transform
from ggqpy.quad import *
from ggqpy.utils import *
from numpy.polynomial.legendre import leggauss, legvander2d
from ggqpy import generalized_gaussian_quadrature

def construct_discretization_matrix(
        I: Interval,
        J: Interval,
        M: int,
        N: int,
        rho: Callable,
        drho: Callable,
        kernel: Callable,
        jacobian: Callable,
        loader: QuadratureLoader
    ):
        ss, tt, ww = gl_nodes2d(I, J, M, N)
        ww = ww * jacobian(ss, tt)
        simplex = Rectangle(I, J)

        Vin = np.linalg.inv(
            legvander2d(I.itranslate(ss), J.itranslate(tt), [M - 1, N - 1])
        )
        A = np.zeros(shape=(N * M, N * M), dtype=complex)
        for idx, singularity in enumerate(zip(ss, tt)):
            xs, yt, w = loader.singular_integral_quad(
                drho, np.array([*singularity]), simplex
            )

            K = w * kernel(*singularity, xs, yt) * jacobian(xs, yt)
            Vout = legvander2d(I.itranslate(xs), J.itranslate(yt), [M - 1, N - 1])
            interpolation_matrix = Vout @ Vin

            A[idx, :] = K @ interpolation_matrix

        A = np.sqrt(ww)[:, np.newaxis] * A / np.sqrt(ww)[np.newaxis, :]

        return A, ss, tt, ww


def ensure_conformal_mapping(jacobian, x0):
    """
    Returns B such that
    Ax = Bx + x0
    ensures tilde{rho}=rho.A such that A(0,0) = x and the Jacobian of tilde{rho} is orthogonal at the point (0,0).

    Parameters
    ----------
    jacobian: lamda that takes input x and returns Jacobian matrix
    x0: Evaluation point

    Returns
    -------
    B
    Binv
    """
    J = jacobian(x0[0], x0[1])

    _, s, Vh = np.linalg.svd(J, full_matrices=False)
    V = Vh.T
    S = np.diag(s)
    Sinv = np.diag(s ** (-1))

    B = V @ Sinv
    Binv = S @ Vh

    return B, Binv


def gl_nodes2d(
    I: Interval,
    J: Interval,
    M: int,
    N: int,
):
    gls = Quadrature.gauss_legendre_on_interval(M, I)
    glt = Quadrature.gauss_legendre_on_interval(N, J)
    ss, tt = np.meshgrid(gls.x, glt.x)
    ss, tt = ss.flatten(), tt.flatten()
    wws, wwt = np.meshgrid(gls.w, glt.w)
    ww = (wws * wwt).flatten()
    return ss, tt, ww


class QuadratureLoader:
    def __init__(self, order, quad_generator = None):
        self.order = order

        if quad_generator == None:
            self.quad_generator = SingularTriangleQuadrature(order)
        else:
            self.quad_generator = quad_generator

        self.x_gl, self.w_gl = leggauss(order)

    def _adapt_gen_quad(self, interval: Interval):
        x = interval.translate(self.x_gl)
        w = (self.w_gl / 2.0) * interval.length()
        return Quadrature(x, w)

    def quad_on_standard_triangle(self, r0, theta0):
        gamma = (
            lambda u: r0
            * np.sin(theta0)
            / (r0 * np.sin(theta0 - theta0 * u) + np.sin(theta0 * u))
        )

        ggq = self.quad_generator.get_quad(r0, theta0)
        
        w_global = list()
        theta_global = list()
        r_global = list()

        gammau = gamma(ggq.x)
        for u, w, R in zip(ggq.x, ggq.w, gammau):
            glx = R * (self.x_gl + 1) / 2.0
            glw = R * self.w_gl / 2.0

            w_global.append(w * glw * theta0 * glx)
            r_global.append(glx)
            theta_global.append(np.full_like(glx, u * theta0))

        r = np.concatenate(r_global)
        theta = np.concatenate(theta_global)
        w = np.concatenate(w_global)
        return r, theta, w

    def singular_integral_quad(self, drho, x0, simplex):
        B, Binv = ensure_conformal_mapping(drho, x0)
        R = Quadrilateral(*[Binv @ (np.array(v) - x0) for v in iter(simplex)])
        x_list = list()
        y_list = list()
        w_list = list()

        detB = abs(np.linalg.det(B))
        for T in [*R.split_into_triangles_around_point((0, 0))]:
            scale, angle, A, Ainv, detA = standard_radial_triangle_transform(
                T.vertices[1], T.vertices[2]
            )
            r, theta, w = self.quad_on_standard_triangle(scale, angle)
            x_local = np.cos(theta) * r
            y_local = np.sin(theta) * r

            v = np.row_stack([x_local, y_local])
            v = B @ (Ainv @ v) + x0[:, np.newaxis]

            x_list.append(v[0, :])
            y_list.append(v[1, :])
            w_list.append((w / detA) * detB)

        x = np.concatenate(x_list)
        y = np.concatenate(y_list)
        w = np.concatenate(w_list)

        return x, y, w


class TriangleQuadKey():
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __eq__(self, y: TriangleQuadKey):
        return isclose(self.a, y.a) and isclose(self.b, y.b)
    
    def __hash__(self):
        return hash(tuple(round(x, 10) for x in [self.a, self.b]))
class SingularTriangleQuadFinder:
    def __init__(self, order):
        self.order = order
        self.computedquads = dict[TriangleQuadKey,Quadrature]()

    def _generate_exact_quad(self, r0, theta0):
        count = 1
        min_length = 1e-10
        eps_disc = 1e-7
        eps_comp = 1e-6
        eps_quad = 1e-6
        F = FunctionFamily.nystrom_integral_functions(count, self.order, r0, r0, theta0, theta0)
        x, w = generalized_gaussian_quadrature(F, min_length, eps_disc, eps_comp, eps_quad)
        quad = Quadrature(x, w)
        return quad

    def get_quad(self, r0, theta0):
        
        key = TriangleQuadKey(r0, theta0)
        if key in self.computedquads:
            quad = self.computedquads[key]
            print("Succesfully reused quadrature")
        else:
            quad = self._generate_exact_quad(r0, theta0)
            self.computedquads[key] = quad

        return quad
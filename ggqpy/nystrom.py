from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from itertools import pairwise
from ggqpy.quad import *
from ggqpy.utils import *
from numpy.polynomial.legendre import leggauss, legvander2d

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


def standard_radial_triangle_transform(a, b):
    """
    Given arbritrary triangle, returns scale and transformation to turn it into the triangle
    (0,0), (1,0), (r0cos(theta0), r0sin(theta0))

    Triangle is assumed to be on the form.
    b
    |\\
    | \\
    |  \\
    0---a

    Parameters
    ----------
    b: Point mapped to (1,0)
    c: Point mapped to (r0cos(theta0), r0sin(theta0))

    Returns
    -------
    scale: Scale
    transform: Mapping from reference triangle.
    A: Mapping to standard triangle
    Ainv: Mapping from standard triangle to original triangle.
    """

    length_a = np.linalg.norm(a)
    length_b = np.linalg.norm(b)

    if length_a < length_b:
        u = b
        v = a
    else:
        u = a
        v = b

    r = np.linalg.norm(u)
    theta = np.arctan2(u[1], u[0])
    c, s = np.cos(theta), np.sin(theta)
    A = np.array([[c, s], [-s, c]]) / r
    Ainv = np.array([[c, -s], [s, c]]) * r
    det = (1 / r) ** 2

    x = A @ v
    angle = np.arctan2(x[1], x[0])
    scale = np.linalg.norm(x)

    if angle < 0.0:
        Ainv[:, 1] = -Ainv[:, 1]
        A[1, :] = -A[1, :]
        angle = -angle

    return scale, angle, A, Ainv, det


class Quadrilateral:
    """
    Rectangle class. Corners are presented as numpy arrays x,y where

    (x3,y3)----(x2,y2)
       |          |
       |          |
       |          |
    (x0,y0)----(x1,y1)
    """

    def __init__(self, a, b, c, d):
        self.vertices = [a, b, c, d]
        self.x = np.array([a[0], b[0], c[0], d[0]])
        self.y = np.array([a[1], b[1], c[1], d[1]])

    def __iter__(self):
        for v in self.vertices:
            yield v

    def split_into_triangles_around_point(self, x0: tuple):
        for p, q in pairwise(self.vertices + [self.vertices[0]]):
            yield Triangle(x0, p, q)


class Rectangle(Quadrilateral):
    def __init__(self, I: Interval, J: Interval):
        self.I = I
        self.J = J
        a = (I.a, J.a)
        b = (I.b, J.a)
        c = (I.b, J.b)
        d = (I.a, J.b)
        super().__init__(a, b, c, d)

    def get_intervals(self):
        return self.I, self.J


class Triangle:
    """
    Triangle class. Corners are presented as numpy arrays x,y where

    (x2,y2)
       |\\
       | \\
       |  \\
       |   \\      
       |    \\      
       |     \\     
    (x0,y0)--(x1,y1)
    """

    def __init__(self, a: tuple, b: tuple, c: tuple):
        self.vertices = [a, b, c]
        self.x = np.array([a[0], b[0], c[0]])
        self.y = np.array([a[1], b[1], c[1]])

    def __repr__(self) -> str:
        repr = list()
        for v in self.vertices:
            repr.append(str(v))
        return str(repr)

    def __eq__(self, other: Triangle) -> bool:
        return len(set(self.vertices + other.vertices)) == 3

    def __iter__(self):
        for v in self.vertices:
            yield v

    def is_in(self, P):
        A, B, C = self.vertices

        denominator = (B[1] - C[1]) * (A[0] - C[0]) + (C[0] - B[0]) * (A[1] - C[1])
        a = (
            (B[1] - C[1]) * (P[0] - C[0]) + (C[0] - B[0]) * (P[1] - C[1])
        ) / denominator
        b = (
            (C[1] - A[1]) * (P[0] - C[0]) + (A[0] - C[0]) * (P[1] - C[1])
        ) / denominator
        c = 1 - a - b

        return a >= 0 and b >= 0 and c >= 0




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
    wins, wint = np.meshgrid(gls.w, glt.w)
    win = (wins * wint).flatten()
    return ss, tt, win

class IntegralOperator:
    def __init__(self,order):
        self.order = order
        self.quad_generator = SingularTriangleQuadrature(order)
        self.x_gl, self.w_gl = leggauss(order)
    
    def _adapt_gen_quad(self, interval: Interval):
        x = interval.translate(self.x_gl)
        w = (self.w_gl / 2.0) * interval.length()
        return Quadrature(x,w)

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

        for u, w in [*ggq]:
            gammau = gamma(u)
            gl = self._adapt_gen_quad(Interval(0, gammau))

            w_global.append(w * gl.w * theta0 * gl.x)
            r_global.append(gl.x)
            theta_global.append(np.full_like(gl.x, u * theta0))

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

    def construct_discretization_matrix(
        self,
        I: Interval,
        J: Interval,
        M: int,
        N: int,
        rho: Callable,
        drho: Callable,
        kernel: Callable,
        jacobian: Callable,
    ):
        ss, tt, win = gl_nodes2d(I, J, M, N)
        wout = win*jacobian(ss,tt)
        simplex = Rectangle(I, J)

        Vin = np.linalg.inv(legvander2d(I.itranslate(ss), J.itranslate(tt), [M - 1, N - 1]))
        A = np.zeros(shape=(N * M, N * M), dtype=complex)
        for idx, singularity in enumerate(zip(ss, tt)):
            xs, yt, w = self.singular_integral_quad(
                drho, np.array([*singularity]), simplex
            )

            K = w * kernel(*singularity, xs, yt) * jacobian(xs, yt)
            Vout = legvander2d(I.itranslate(xs), J.itranslate(yt), [M - 1, N - 1])
            interpolation_matrix = Vout @ Vin

            A[idx, :] = np.sqrt(wout[idx]) * ((K @ interpolation_matrix) / np.sqrt(win))

        return A, ss, tt, win, wout

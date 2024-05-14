from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from itertools import pairwise, product
from ggqpy.quad import *
from ggqpy.utils import *
from numpy.polynomial.legendre import leggauss, legvander2d
from typing import List


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
    wws, wwt = np.meshgrid(gls.w, glt.w)
    ww = (wws * wwt).flatten()
    return ss, tt, ww


class Patch:
    def __init__(
        self,
        m,
        n,
        M,
        N,
        I: Interval,
        J: Interval,
        nodes,
        weights,
        jacobian,
        id,
        rho,
    ):
        self.I = Interval(I.a + (I.b - I.a) * m / M, (I.b - I.a) * (m + 1) / M)
        self.J = Interval(J.a + (J.b - J.a) * n / N, (J.b - J.a) * (n + 1) / N)

        s = self.I.translate(nodes)
        t = self.J.translate(nodes)

        ss, tt = np.meshgrid(s, t)
        self.ss, self.tt = ss.flatten(), tt.flatten()
        self.nodes = zip(self.ss, self.tt)

        ws, wt = (I.length() / 2.0) * weights / M, (J.length() / 2.0) * weights / N
        wws, wwt = np.meshgrid(ws, wt)

        self.ww_no_jacobian = (wws * wwt).flatten()
        self.ww = self.ww_no_jacobian * jacobian(self.ss, self.tt)

        self.id = id
        self.scale = I.length() * J.length() / 4.0
        disc_nodes = len(nodes)
        self.start = id * disc_nodes**2
        self.end = (id + 1) * disc_nodes**2
        self.center = rho(I.mid(), J.mid())
        corners = rho(np.array([I.a, I.a, I.b, I.b]), np.array([J.a, J.b, J.a, J.b]))
        self.bounding_distance = np.max(
            np.linalg.norm(corners - self.center[:, np.newaxis])
        )

    def subdivide(self):
        yield Patch()
        yield Patch()
        yield Patch()
        yield Patch()


class IntegralOperator:
    def __init__(self, order):
        self.order = order
        self.quad_generator = SingularTriangleQuadrature(order)
        self.x_gl, self.w_gl = leggauss(order + 1)
        xx, yy = np.meshgrid(self.x_gl, self.x_gl)
        self.xx, self.yy = xx.flatten(), yy.flatten()
        w1, w2 = np.meshgrid(self.w_gl, self.w_gl)
        w = w1.flatten() * w2.flatten()

        ## Prepare interpolation matrices for near interactions
        Vin = np.linalg.inv(legvander2d(self.xx, self.yy, [self.order, self.order]))

        x = [
            Interval(-1.0, 0.0).translate(self.xx),
            Interval(0.0, 1.0).translate(self.xx),
        ]
        y = [
            Interval(-1.0, 0.0).translate(self.yy),
            Interval(0.0, 1.0).translate(self.yy),
        ]
        V = [
            legvander2d(u, v, [self.order, self.order])
            for u, v in [(x[0], y[0]), (x[1], y[0]), (x[1], y[1]), (x[0], y[1])]
        ]
        self.interpolation_matrix = np.vstack(V) @ Vin
        self.sub_xx = np.concatenate([x[0], x[1], x[1], x[0]])
        self.sub_yy = np.concatenate([y[0], y[0], y[1], y[1]])
        self.sub_ww = np.concatenate([w, w, w, w]) / 4.0

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

    
    def submatrix_far(self, s, t, kernel, source: Patch):
        return kernel(s, t, source.ss, source.tt) * source.ww

    def submatrix_near(self, s, t, kernel, source: Patch, jacobian, subdivisions):
        sub_length = 2/subdivisions
        x = Interval(-1, -1 + sub_length).translate(self.x_gl)
        xx = np.concatenate([x + offset for offset in sub_length*np.arange(subdivisions)])
        # TODO

        ss = source.I.translate(self.sub_xx)
        tt = source.J.translate(self.sub_yy)
        ww = np.tile(source.ww_no_jacobian, 4)/4.0
        return (
            kernel(s, t, ss, tt) * ww * jacobian(ss, tt)
        ) @ self.interpolation_matrix

    def submatrix_self(
        self,
        patch: Patch,
        disc_nodes,
        rho: Callable,
        drho: Callable,
        kernel: Callable,
        jacobian: Callable,
    ):
        I, J = patch.I, patch.J
        ss, tt = patch.ss, patch.tt
        simplex = Rectangle(I, J)

        Vin = np.linalg.inv(
            legvander2d(I.itranslate(ss), J.itranslate(tt), [self.order, self.order])
        )
        A = np.zeros(shape=(disc_nodes**2, disc_nodes**2), dtype=complex)
        for idx, singularity in enumerate(zip(ss, tt)):
            xs, yt, w = self.singular_integral_quad(
                drho, np.array([*singularity]), simplex
            )

            K = w * kernel(*singularity, xs, yt) * jacobian(xs, yt)
            Vout = legvander2d(
                I.itranslate(xs), J.itranslate(yt), [self.order, self.order]
            )
            interpolation_matrix = Vout @ Vin

            A[idx, :] = K @ interpolation_matrix

        return A

    def construct_discretization_matrix(
        self,
        disc_nodes,
        M,
        N,
        I: Interval,
        J: Interval,
        rho: Callable,
        drho: Callable,
        kernel: Callable,
        jacobian: Callable,
    ):
        mm = np.arange(M)
        nn = np.arange(N)
        A = np.zeros(
            shape=(disc_nodes**2 * N * M, disc_nodes**2 * N * M), dtype=complex
        )

        ss_global, tt_global, ww_global = list(), list(), list()
        patches = list[Patch]()
        for id, (m, n) in enumerate(product(mm, nn)):
            patch = Patch(m, n, M, N, I, J, self.x_gl, self.w_gl, jacobian, id, rho)
            patches.append(patch)
            ss_global.append(patch.ss)
            tt_global.append(patch.tt)
            ww_global.append(patch.ww)

        near_interactions = 0
        for target in patches:
            for source in patches:
                if target.id == source.id:
                    A[target.start : target.end, source.start : source.end] = (
                        self.submatrix_self(
                            target, disc_nodes, rho, drho, kernel, jacobian
                        )
                    )
                else:
                    for idx, (s, t) in enumerate(target.nodes):

                        dist = np.linalg.norm(rho(s, t) - source.center)
                        if dist > 2.0 * source.bounding_distance:
                            subdivisions_needed = np.ceil(np.log2(2*source.bounding_distance/dist))
                            A[target.start + idx, source.start : source.end] = self.submatrix_far(s, t, kernel, source, subdivisions_needed)
                        else:
                            A[target.start + idx, source.start : source.end] = (
                                self.submatrix_near(s, t, kernel, source, jacobian)
                            )
                            near_interactions = near_interactions + 1

        ss = np.concatenate(ss_global)
        tt = np.concatenate(tt_global)
        ww = np.concatenate(ww_global)
        A = np.sqrt(ww)[:, np.newaxis] * A / np.sqrt(ww)[np.newaxis, :]
        print(f"(M,N) = {(M,N)}, Near interactions {near_interactions}")
        return A, ss, tt, ww

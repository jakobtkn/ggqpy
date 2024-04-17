from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from itertools import pairwise
from ggqpy.utils import *


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
    A: affinemapping matrix
    transform: Mapping from reference triangle.
    """
    J = jacobian(x0)

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
        Ainv[:,1] = -Ainv[:,1]
        A[1,:] = -A[1,:]
        angle = -angle

    return scale, angle, A, Ainv, det


class Rectangle:
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


def load_ggq_quad(r0, theta0):
    ###!!! pick correct ###
    order = 4
    quad = Quadrature.load_from_file(f"quads/nystrom.{16}.{order}.quad")
    return quad, order


def quad_on_standard_triangle(r0, theta0):
    gamma = (
        lambda u: r0
        * np.sin(theta0)
        / (r0 * np.sin(theta0 - theta0 * u) + np.sin(theta0 * u))
    )

    ggq, order = load_ggq_quad(r0, theta0)
    w_global = list()
    theta_global = list()
    r_global = list()

    for u, w in [*ggq]:
        gammau = gamma(u)
        gl = Quadrature.gauss_legendre_on_interval(order, Interval(0, gammau))

        w_global.append(w * gl.w * theta0)
        r_global.append(gl.x)
        theta_global.append(np.full_like(gl.x, u * theta0))

    r = np.concatenate(r_global)
    theta = np.concatenate(theta_global)
    w = np.concatenate(w_global)
    return r, theta, w


def singular_integral_quad(drho, x0, simplex):
    B, Binv = ensure_conformal_mapping(drho, x0)
    R = Rectangle(*[Binv @ (np.array(v) - x0) for v in iter(simplex)])

    x_list = list()
    y_list = list()
    w_list = list()

    for T in [*R.split_into_triangles_around_point((0,0))]:
        scale, angle, A, Ainv, det = standard_radial_triangle_transform(
            T.vertices[1], T.vertices[2]
        )
        r, theta, w = quad_on_standard_triangle(scale, angle)
        x_local = np.cos(theta) * r
        y_local = np.sin(theta) * r

        v = np.row_stack([x_local, y_local])
        v = B @ (Ainv @ v) + x0[:, np.newaxis]

        x_list.append(v[0, :])
        y_list.append(v[1, :])
        w_list.append(w)

    x = np.concatenate(x_list)
    y = np.concatenate(y_list)

    return x, y, w

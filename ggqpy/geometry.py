from itertools import pairwise
from numpy.typing import NDArray
import numpy as np

from ggqpy.utils import Interval

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
from ggqpy.utils import Interval
from itertools import pairwise
from numpy.typing import ArrayLike
import numpy as np


from typing import Callable


class Quadrature:
    def __init__(self, x: np.ndarray, w: np.ndarray) -> None:
        assert len(x) == len(w)
        self.x = x
        self.w = w
        self.size = len(x)
        return

    def save_to_file(self, file_name: str):
        np.savetxt(file_name, np.column_stack((self.x, self.w)))
        return

    @classmethod
    def load_from_file(cls, file_name: str):
        data = np.genfromtxt(file_name)
        x, w = np.hsplit(data, 2)
        return cls(x, w)

    @classmethod
    def gauss_legendre_on_interval(cls, order: int, interval: Interval):
        x_gl, w_gl = np.polynomial.legendre.leggauss(order)
        x = interval.translate(x_gl)
        w = (w_gl / 2.0) * interval.length()
        return cls(x, w)

    def eval(self, f: Callable):
        return f(self.x) @ self.w

    def __iter__(self):
        for node in zip(self.x, self.w):
            yield node

class SingularTriangleQuadrature():
    r0_breakpoints = [1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,0.2,0.4,0.7,0.8,1.0]
    theta0_breakpoints = [1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,0.25,0.5,1.0,1.5,3.0, 3.14, np.pi]
    r0_intervals = pairwise(r0_breakpoints)
    theta0_intervals = pairwise(theta0_breakpoints)
    
    def __init__(self, quad_folder = "quads/nystrom"):

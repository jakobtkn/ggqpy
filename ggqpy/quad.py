from ggqpy.utils import Interval
from itertools import pairwise
from numpy.typing import ArrayLike
import numpy as np
import glob
import bisect

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


class SingularTriangleQuadrature:
    def __init__(self, quad_folder="quads/nystrom.4/", order = 4):
        self.order = 4
        self.breakpoints = {
            "r0": np.loadtxt(quad_folder + "breakpoints_r"),
            "theta0": np.loadtxt(quad_folder + "breakpoints_theta"),
        }
        self.intervals = {
            "r0": pairwise(self.breakpoints["r0"]),
            "theta0": pairwise(self.breakpoints["theta0"]),
        }

        self.quadratures = dict()
        for filepath in glob.glob(quad_folder + "*.quad"):
            r0_index = filepath[0]
            theta0_index = filepath[2]
            self.quadratures[(r0_index,theta0_index)] = Quadrature.load_from_file(filepath)

    def get_quad(self, r0, theta0):
        r0_index = bisect.bisect(self.breakpoints["r0"], r0)
        theta0_index = bisect.bisect(self.breakpoints["theta0"], theta0)

        assert (r0_index,theta0_index) in self.quadratures
        
        return self.quadratures[(r0_index,theta0_index)]
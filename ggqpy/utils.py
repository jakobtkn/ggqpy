import numpy as np
import scipy as sp
import bisect
from typing import Callable
from numpy.typing import ArrayLike
from sympy import Symbol, Expr, integrate, lambdify, nan

x = Symbol("x", real=True)


class Interval:
    def __init__(self, start: float, end: float) -> None:
        if start > end:
            raise Exception("end must be greater than start")
        self.a = start
        self.b = end
        self.translate = sp.interpolate.interp1d([-1.0, 1.0], [start, end])
        return

    def __repr__(self):
        return (self.a, self.b)

    def __str__(self):
        return "(" + str(self.a) + "," + str(self.b) + ")"

    def __iter__(self):
        yield self.a
        yield self.b

    def length(self):
        return self.b - self.a

    def is_in(self, x):
        return np.logical_and((self.a <= x), (x <= self.b))


class Quadrature:
    def __init__(self, x: ArrayLike, w: ArrayLike) -> None:
        self.x = x
        self.w = w
        return

    def save_as_file(self, file_name: str):
        np.savetxt(self.file_name, np.column_stack((self.x, self.w)))
        return

    @classmethod
    def from_file(cls, file_name: str):
        data = np.genfromtxt(file_name)
        x, w = np.hsplit(data)
        return cls(x, w, file_name)

    def eval(self, f: Callable):
        return f(self.x).w


class FunctionFamily:
    I = None
    functions_symbolic = None
    functions_lambdas = None

    def __init__(
        self, I: Interval, functions_lambdas, functions_symbolic=None
    ) -> None:
        self.I = I
        self.functions_evaluations = functions_lambdas

    @classmethod
    def from_symbolic(cls, I: Interval, functions_symbolic):
        functions_evaluations = list()
        for expr in functions_symbolic:
            functions_evaluations.append(lambdify(x, expr, "numpy"))
        return cls(I, functions_evaluations, functions_symbolic)

    def target_integral(self, f: Expr) -> float:
        integral = integrate(f, (x, self.I.a, self.I.b))
        return integral

    def generate_example_function(self, loc=0, scale=1) -> Expr:
        number_of_functions = len(self.functions_symbolic)
        c = np.random.normal(loc, scale, size=number_of_functions)
        expr = sum(np.array(self.functions_symbolic) * c)
        f = lambdify(x, expr, "numpy")
        return f, expr


class PiecewiseLegendre:
    def __init__(self, poly_list, endpoints) -> None:
        self.poly_list = poly_list
        self.endpoints = endpoints
        self.endpoints_bisect = endpoints.copy()
        self.endpoints_bisect[0] = -np.inf
        self.endpoints_bisect[-1] = np.inf
        return

    def __call__(self, x):
        y = np.zeros_like(x)
        for k in range(len(x)):
            i = bisect.bisect_right(self.endpoints_bisect, x[k]) - 1
            y[k] = self.poly_list[i](x[k])
        return y

    def deriv(self):
        deriv_poly_list = [p.deriv() for p in self.poly_list]
        return PiecewiseLegendre(deriv_poly_list, self.endpoints)


class PiecewiseLegendreFamily:
    def __init__(self, poly_list, endpoints) -> None:
        self.number_of_functions = len(poly_list)
        self.piecewise_poly_list = poly_list
        self.piecewise_poly_deriv_list = [p.deriv() for p in poly_list]
        self.endpoints = endpoints
        self.endpoints_bisect = endpoints.copy()
        self.endpoints_bisect[0] = -np.inf
        self.endpoints_bisect[-1] = np.inf
        return

    def __call__(self, x):
        """Evaluates functions in list such that row k contains
        [P_k(x)]
        """
        y = np.zeros(shape=(self.number_of_functions, len(x)), dtype=float)
        for k in range(len(x)):
            i = bisect.bisect_right(self.endpoints_bisect, x[k]) - 1
            y[:, k] = np.array([p.poly_list[i](x[k]) for p in self.piecewise_poly_list])
        return y

    def eval_block(self, x):
        """Evaluates functions in list such that row k contains
        [P_k(x) P_k'(x)]
        """
        y = np.zeros(shape=(self.number_of_functions, 2 * len(x)), dtype=float)
        n = len(x)
        for k in range(n):
            i = bisect.bisect_right(self.endpoints_bisect, x[k]) - 1
            y[:, k] = np.array(
                [p.poly_list[i](x[k]) for p in self.piecewise_poly_deriv_list]
            )
            y[:, k + n] = np.array(
                [p.poly_list[i](x[k]) for p in self.piecewise_poly_list]
            )
        return y

    def deriv(self):
        deriv_poly_list = [p.deriv() for p in self.poly_list]
        return PiecewiseLegendre(deriv_poly_list, self.endpoints)

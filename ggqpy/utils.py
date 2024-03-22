import numpy as np
import scipy as sp
import sympy
import bisect
from typing import Callable
from numpy.typing import ArrayLike


class Interval:
    def __init__(self, start: float, end: float) -> None:
        if start > end:
            raise Exception("end must be greater than start")
        self.a = start
        self.b = end
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

    def save_to_file(self, file_name: str):
        np.savetxt(file_name, np.column_stack((self.x, self.w)))
        return

    @classmethod
    def load_from_file(cls, file_name: str):
        data = np.genfromtxt(file_name)
        x, w = np.hsplit(data, 2)
        return cls(x, w)

    def eval(self, f: Callable):
        return f(self.x).w


class FunctionFamily:
    I = None
    functions_lambdas = None
    functions_symbolic = None

    def __init__(self, I: Interval, functions_lambdas, functions_symbolic=None) -> None:
        self.I = I
        self.functions_lambdas = functions_lambdas
        self.functions_symbolic = functions_symbolic

    @classmethod
    def polynomials_and_singularity(
        cls,
        I: Interval,
        order: int = 5,
        number_of_polynomials: int = 100,
        rng_gen: np.random.Generator = np.random.default_rng(0),
    ):
        x = sympy.Symbol("x", real=True)
        functions_lambdas = list()
        functions_symbolic = list()

        for _ in range(number_of_polynomials):
            c = rng_gen.integers(-10, 10, size=order)
            f = sympy.Poly(c, x).as_expr()
            functions_symbolic.append(f)

        functions_symbolic.append(1 / x)

        for fsym in functions_symbolic:
            functions_lambdas.append(sympy.lambdify(x, fsym, "numpy"))

        return cls(I, functions_lambdas, functions_symbolic)

    @classmethod
    def nystrom_integral_functions(
        cls, number_of_discretizations=16, order=4
    ):
        gamma = (
            lambda r0, theta0, u: r0
            * np.sin(theta0)
            / (r0 * np.sin(theta0 - theta0 * u) + np.sin(theta0 * u))
        )

        x_gl, _ = np.polynomial.legendre.leggauss(number_of_discretizations)

        (amin, amax) = (1e-7, 1)
        alphas = (amax - amin) * (x_gl + 1) / 2 + amin

        (bmin, bmax) = (1e-7, np.pi)
        betas = (bmax - bmin) * (x_gl + 1) / 2 + bmin

        functions = [
            lambda u, alpha=alpha, beta=beta: beta
            * gamma(alpha, beta, u) ** (i + 2)
            / (i + 2)
            * trig(j * beta * u)
            for trig in [np.cos,np.sin]
            for i in range(-1, order + 1)
            for j in range(0, 3 * (i + 1) + 2 + 1)
            for alpha in alphas
            for beta in betas
        ]

        return cls(Interval(0,1), functions)


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

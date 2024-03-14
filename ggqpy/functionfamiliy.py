import numpy as np
import scipy as sp
from sympy import Symbol, Expr, integrate, lambdify, nan

symx = Symbol("x", real=True)


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


class FunctionFamily:
    I = None
    sym_functions = None
    functions = None

    def __init__(
        self, I: Interval, sym_functions: list[Expr], lambda_functions=None
    ) -> None:
        self.I = I
        if sym_functions is not None:
            self.sym_functions = sym_functions
            self._gen_lambdas_from_sym_exprs()
        else:
            self.functions = lambda_functions
        return

    def _gen_lambdas_from_sym_exprs(self):
        self.functions = list()
        for expr in self.sym_functions:
            self.functions.append(lambdify(symx, expr, "numpy"))

    def target_integral(self, f: Expr) -> float:
        integral = integrate(f, (symx, self.I.a, self.I.b))
        return integral

    def generate_example_function(self, loc=0, scale=1) -> Expr:
        number_of_functions = len(self.sym_functions)
        c = np.random.normal(loc, scale, size=number_of_functions)
        expr = sum(np.array(self.sym_functions) * c)
        f = lambdify(symx, expr, "numpy")
        return f, expr


class PiecewiseLegendre:
    def __init__(self, poly_list, endpoints) -> None:
        self.endpoints = endpoints
        self.PPoly = sp.interpolate.PPoly(poly_list, endpoints, extrapolate=False)
        return

    def __call__(self, x):
        return self.PPoly(x)

    def deriv(self):
        deriv_poly_list = self.PPoly.derivative()
        return PiecewiseLegendre(deriv_poly_list, self.endpoints)

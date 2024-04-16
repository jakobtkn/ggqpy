import numpy as np
import scipy as sp
import sympy
import bisect
import numpy.polynomial.legendre as legendre
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

    def __lt__(self, other):
        return self.a < other.a

    def length(self):
        return self.b - self.a

    def is_in(self, x):
        return np.logical_and((self.a <= x), (x <= self.b))

    def translate(self, x):
        return (x + 1.0) * (self.b - self.a) / 2.0 + self.a


class Quadrature:
    def __init__(self, x: ArrayLike, w: ArrayLike) -> None:
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

    def eval(self, f: Callable):
        return f(self.x) @ self.w


class FunctionFamily:
    I = None
    functions_lambdas = None

    def __init__(self, I: Interval, functions_lambdas) -> None:
        self.I = I
        self.functions_lambdas = functions_lambdas

    @classmethod
    def nystrom_integral_functions(
        cls,
        number_of_discretizations=16,
        order=4,
        amin=1e-7,
        amax=1.0,
        bmin=1e-7,
        bmax=np.pi,
    ):
        gamma = (
            lambda r0, theta0, u: r0
            * np.sin(theta0)
            / (r0 * np.sin(theta0 - theta0 * u) + np.sin(theta0 * u))
        )

        x_gl, _ = legendre.leggauss(number_of_discretizations)
        alphas = (amax - amin) * (x_gl + 1) / 2 + amin
        betas = (bmax - bmin) * (x_gl + 1) / 2 + bmin

        functions = [
            lambda u, alpha=alpha, beta=beta, i=i, j=j, trig=trig: beta
            * gamma(alpha, beta, u) ** (i + 2)
            / (i + 2)
            * trig(j * beta * u)
            for trig in [np.cos, np.sin]
            for i in range(-1, order + 1)
            for j in range(0, 3 * (i + 1) + 2 + 1)
            for alpha in alphas
            for beta in betas
        ]

        return cls(Interval(0, 1), functions)
    
    def generate_example_function(self):
        n = len(self.functions_lambdas)
        c = np.random.randint(-10, 10, size=n).astype(float)
        f_lambda = lambda x: c @ np.array([f(x) for f in self.functions_lambdas])
        return f_lambda


class FunctionFamilySymbolic(FunctionFamily):
    x = sympy.Symbol("x", real=True)
    functions_lambdas = list()
    functions_symbolic = None

    def __init__(self, I: Interval, functions_symbolic):
        self.I = I
        self.functions_symbolic = functions_symbolic

        for fsym in functions_symbolic:
            self.functions_lambdas.append(sympy.lambdify(self.x, fsym, "numpy"))

    @classmethod
    def polynomials_and_singularity(
        cls,
        I: Interval = Interval(1e-8,1),
        order: int = 9,
        number_of_polynomials: int = 20,
        rng_gen: np.random.Generator = np.random.default_rng(0),
    ):
        x = sympy.Symbol("x", real=True)

        functions_symbolic = list()
        for _ in range(number_of_polynomials):
            c = rng_gen.integers(-10, 10, size=order)
            f = sympy.Poly(c, x).as_expr()
            functions_symbolic.append(f)

        functions_symbolic.append(sympy.ln(x))

        return cls(I, functions_symbolic)

    def generate_example_function(self):
        n = len(self.functions_symbolic)
        c = np.random.randint(-10, 10, size=n)
        f_symbolic = sum(np.array(self.functions_symbolic) * c)
        f_lambda = sympy.lambdify(self.x, f_symbolic, "numpy")
        return f_lambda, f_symbolic

    def integral(self, f):
        return sympy.integrate(f, (self.x, self.I.a, self.I.b))


class PiecewiseLegendre:
    def __init__(self, poly_list, endpoints) -> None:
        self.poly_list = poly_list
        self.endpoints = endpoints
        self.endpoints_bisect = endpoints.copy()[1:-1]
        return

    @classmethod
    def interpolate_gauss_legendre_points(cls, u, endpoints):
        number_of_intervals = len(endpoints) - 1
        points_total = len(u)
        points_per_interval = points_total // number_of_intervals
        poly_list = list()

        x, _ = legendre.leggauss(points_per_interval)
        for i in range(number_of_intervals):
            interval = Interval(endpoints[i], endpoints[i + 1])
            u_local = u[points_per_interval * i : points_per_interval * (i + 1)]

            p = legendre.Legendre.fit(
                interval.translate(x),
                u_local,
                domain=[*interval],
                window=[-1,1],
                deg=points_per_interval - 1,
                rcond=1e-16
            )
            poly_list.append(p)

        return cls(poly_list, endpoints)

    def __call__(self, x):
        y = np.zeros_like(x)
        for k in range(len(x)):
            i = bisect.bisect_right(self.endpoints_bisect, x[k])
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
        self.endpoints_bisect = endpoints.copy()[1:-1]
        return

    def __call__(self, x):
        """Evaluates functions in list such that row k contains
        [P_k(x)]
        """
        np.clip(x, self.endpoints[0], self.endpoints[-1], out=x)
        n = len(x)
        y = np.zeros(shape=(self.number_of_functions, len(x)), dtype=float)
        for k in range(n):
            i = bisect.bisect_right(self.endpoints_bisect, x[k])
            y[:, k] = np.array([p.poly_list[i](x[k]) for p in self.piecewise_poly_list])
        return y

    def eval_block(self, x):
        """Evaluates functions in list such that row k contains
        [P_k(x) P_k'(x)]
        """
        np.clip(x, self.endpoints[0], self.endpoints[-1], out=x)
        n = len(x)
        y = np.zeros(shape=(self.number_of_functions, 2 * len(x)), dtype=float)
        for k in range(n):
            i = bisect.bisect_right(self.endpoints_bisect, x[k])
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

import numpy as np
from typing import Callable
from sympy import Symbol, Expr, Poly, cos

from ggqpy.utils import FunctionFamily, Interval


x = Symbol("x", real=True)


def example_problem(
    I: Interval,
    number_of_functions: int,
    expr_gen: Callable[[int], Expr],
    rng_gen=np.random.default_rng(0),
):
    functions = list()

    for i in range(number_of_functions):
        functions.append(expr_gen(i, rng_gen))

    return FunctionFamily(I, functions)


def gen_trig(i, rng_gen) -> Expr:
    freq = rng_gen.uniform(-1, 1, size=None)
    phase = rng_gen.uniform(-1, 1, size=None)
    f = cos(x * freq + phase)
    return f


def gen_sing(i, rng_gen) -> Expr:
    c = rng_gen.uniform(1, 2, size=None)
    f = x ** (-c)
    return f


def gen_poly(i, rng_gen) -> Expr:
    degree = 11
    c = rng_gen.uniform(-1, 1, size=degree)
    f = Poly(c, x).as_expr()
    return f


def gen_poly_and_sing(i, rng_gen):
    if i == 0:
        return x ** (-1)
    else:
        return gen_poly(i, rng_gen)

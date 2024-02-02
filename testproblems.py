import numpy as np
from typing import Callable
from sympy import *

from functionfamiliy import FunctionFamily


x = Symbol("x")

def example_problem(I: Interval, number_of_functions: int, expr_gen: Callable[[int],Expr]):
    functions = list()
    
    for i in range(number_of_functions):
        functions.append(expr_gen(i))
        
    return FunctionFamily(I, functions)

def gen_trig(i) -> Expr: 
    freq = np.random.uniform(-1, 1, size = None)
    phase = np.random.uniform(-1, 1, size = None)
    f = cos(x*freq + phase)
    return f

def gen_sing(i) -> Expr:
    c = np.random.uniform(1, 2, size = None)
    f = x**(-c)
    return f

def gen_poly(i) -> Expr:
    degree = 5
    c = np.random.uniform(-1, 1, size = degree)
    f = Poly(c,x).as_expr()
    return f

def gen_poly_and_sing(i):
    if i == 0:
        return gen_sing(i)
    else:
        return gen_poly(i)

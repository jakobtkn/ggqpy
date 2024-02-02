import numpy as np
import scipy as sp
from sympy import *

x = Symbol("x")
class Interval:
    def __init__(self, start: float, end: float) -> None:
        if (start > end):
            raise Exception("end must be greater than start") 
        self.a = start
        self.b = end
        self.translate = sp.interpolate.interp1d([-1.0,1.0], [start,end])
        return
    
    def __repr__(self):
        return (self.a,self.b)
    
    def __str__(self):
        return "(" + str(self.a) + "," +str(self.b) + ")"

class FunctionFamily:
    I = None
    sym_functions = None
    functions = None
    
    def __init__(self, I: Interval, sym_functions: list[Expr]) -> None:
        self.I = I
        self.sym_functions = sym_functions
        self._gen_lambdas_from_sym_exprs()
        return
    
    def _gen_lambdas_from_sym_exprs(self):
        self.functions = list()
        for expr in self.sym_functions:
            self.functions.append(lambdify(x,expr,"numpy"))
    
    def target_integral(self, f: Expr) -> float:
        return integrate(f, (x,self.I.a,self.I.b))
    
    def generate_example_function(self, loc = 0, scale = 1) -> Expr:
        number_of_functions = len(self.sym_functions)
        c = np.random.normal(loc, scale, size=number_of_functions)
        expr = sum(np.array(self.sym_functions)*c)
        f = lambdify(x,expr,"numpy")
        return f, expr
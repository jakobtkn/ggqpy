import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from typing import Callable
import numpy.polynomial.legendre as legendre

class Interval:
    def __init__(self, start: float, end: float) -> None:
        if (start > end):
            raise Exception("end must be greater than start") 
        self.a = start
        self.b = end
        return
    
    def __repr__(self):
        return (self.a,self.b)
    
    def __str__(self):
        return "(" + str(self.a) + "," +str(self.b) + ")"

class FunctionFamily:
    def __init__(self, I: Interval, functions: list[Callable[[float],float]]) -> None:
        self.I = I
        self.functions = functions
        return

## Example problems
def example_problem(I, number_of_functions, scale=1):
    functions = list()
    
    for _ in range(number_of_functions//2):
        c = scale*np.random.uniform(-1, 1, size = None)
        phi_cos = lambda x, c=c: np.cos(x*c)
        phi_sin = lambda x, c=c: np.sin(x*c)
        functions.append(phi_cos)
        functions.append(phi_sin)
        
    return FunctionFamily(I, functions)

def singularity_problem(I, number_of_functions, scale=1):
    functions = list()
    
    for _ in range(number_of_functions):
        c = scale*np.random.uniform(1, 2, size = None)
        phi = lambda x, c=c: x**(-c)
        functions.append(phi)
        
    return FunctionFamily(I, functions)

def polynomial_problem(I, number_of_functions, degree, scale):
    functions = list()
    
    for _ in range(number_of_functions):
        c = scale*np.random.uniform(-1, 1, size = degree)
        phi = lambda x, c=c: np.polyval(c,x)
        functions.append(phi)
    return FunctionFamily(I, functions)

def pairwise(iterable):
    it = iter(iterable)
    a = next(it, None)

    for b in it:
        yield (a, b)
        a = b

def adaptive_discretization(function_family, precision, k, verbose = False):
    '''
    Adaptive disrectization using nested Gaussian Legendre polynomial interpolation.
    Procedure described in "A nonlinear optimization procedure for generalized Gaussian quadratures" p.12-13
    '''
    I = function_family.I
    endpoints = [I.a, I.b]
    
    ## Stage 1.
    def interval_compatible(I, phi):
        translate = sp.interpolate.interp1d([-1.0,1.0], [I.a,I.b])
        x,w = legendre.leggauss(2*k)
        alpha = legendre.legfit(x, y=phi(translate(x)), deg=2*k-1) # Fit to Legendre Polynomials on [a,b]
        high_freq_sq_residuals = np.sum(abs(alpha[k:])**2)
        
        if verbose:
            print("Residual: ", high_freq_sq_residuals, " found on interval ", I)
        
        return high_freq_sq_residuals < precision
    
    def add_endpoints(sub_interval, phi):
        if interval_compatible(sub_interval, phi):
            return
        else:
            midpoint = (sub_interval.a+sub_interval.b)/2.0
            endpoints.append(midpoint)
            add_endpoints(Interval(sub_interval.a,midpoint), phi)
            add_endpoints(Interval(midpoint,sub_interval.b), phi)
            return
        
    for phi in function_family.functions:
        add_endpoints(I, phi)
        
    ## Stage 2.
    endpoints = sorted(set(endpoints))
    
    if verbose:
        print("Endpoints found: ", endpoints)
    
    ## Stage 3.
    x_global = []
    w_global = []
    for (start,end) in pairwise(endpoints):
        x,w = legendre.leggauss(2*k)
        translate = sp.interpolate.interp1d([-1.0,1.0], [start,end])
        x_global.append(translate(x))
        w_global.append(w)
    
    x_global = np.concatenate(x_global)
    w_global = np.concatenate(w_global)
    
    return x_global, w_global, endpoints

def compress_sequence_of_functions(function_family, x, w, precision):
    
    A = np.column_stack([phi(x)*np.sqrt(w) for phi in function_family.functions])
    Q,R,P = sp.linalg.qr(A, pivoting=True)
    
    U = np.array([])
    return U
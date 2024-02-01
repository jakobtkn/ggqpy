import numpy as np
from sympy import *

from functionfamiliy import FunctionFamily

x = Symbol("x")

## Example problems
def example_problem(I, number_of_functions, scale=1):
    functions = list()
    analytic_integrals = list()
    
    for _ in range(number_of_functions):
        freq = scale*np.random.uniform(-1, 1, size = None)
        phase = np.random.uniform(-1, 1, size = None)
        
        f = cos(x*freq + phase)
        functions.append(lambdify(x,f,"numpy"))
        
        ## Analytical solution for testing
        analytic_integrals.append(integrate(f,(x,I.a,I.b)))
        
    return FunctionFamily(I, functions, analytic_integrals)

def singularity_problem(I, number_of_functions, scale=1):
    functions = list()
    analytic_integrals = list()
    
    for _ in range(number_of_functions):
        c = scale*np.random.uniform(1, 2, size = None)
        
        f = x**(-c)
        functions.append(lambdify(x,f,"numpy"))
        
        ## Analytical solution for testing
        analytic_integrals.append(integrate(f,(x,I.a,I.b)))
        
    return FunctionFamily(I, functions, analytic_integrals)

def polynomial_problem(I, number_of_functions, degree, scale):
    functions = list()
    analytic_integrals = list()
    
    for _ in range(number_of_functions):
        c = scale*np.random.uniform(-1, 1, size = degree)
        f = Poly(c,x)
        functions.append(lambdify(x,f,"numpy"))
        
        ## Analytical solution for testing
        analytic_integrals.append(integrate(f,(x,I.a,I.b)))
    return FunctionFamily(I, functions, analytic_integrals)

def poly_and_sing_problem(I, number_of_functions, degree, scale):
    functions = list()
    analytic_integrals = list()
    
    f = x**(-1)
    print(f)
    functions.append(lambdify(x,f,"numpy"))
    
    ## Analytical solution for testing
    analytic_integrals.append(integrate(f,(x,I.a,I.b)))
    
    for _ in range(number_of_functions-1):
        c = scale*np.random.uniform(-1, 1, size = degree)
        f = Poly(c,x).as_expr()
        print(f)
        functions.append(lambdify(x,f,"numpy"))
        
        ## Analytical solution for testing
        analytic_integrals.append(integrate(f,(x,I.a,I.b)))
    return FunctionFamily(I, functions, analytic_integrals)

# FunctionFamily(I, [lambda x: x**(-1.0), lambda x: 99*np.ones_like(x), lambda x: x + 3, lambda x: x**2, lambda x: x**3 - 10*x**2 + x -1])
import numpy as np
import scipy as sp
import numpy.polynomial.legendre as legendre

from functionfamiliy import FunctionFamily, Interval

def pairwise(iterable):
    it = iter(iterable)
    a = next(it, None)

    for b in it:
        yield (a, b)
        a = b

def adaptive_discretization(function_family, precision, k, verbose = False):
    """
    Adaptive disrectization using nested Gaussian Legendre polynomial interpolation.
    Procedure described in "A nonlinear optimization procedure for generalized Gaussian quadratures" p.12-13
    """
    I = function_family.I
    endpoints = [I.a, I.b]
    
    ## Stage 1.
    def interval_compatible(I, phi):
        translate = sp.interpolate.interp1d([-1.0,1.0], [I.a,I.b])
        x,_ = legendre.leggauss(2*k)
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
        w = w*(end-start)/2
        translate = sp.interpolate.interp1d([-1.0,1.0], [start,end])
        x_global.append(translate(x))
        w_global.append(w)
    
    x_global = np.concatenate(x_global)
    w_global = np.concatenate(w_global)
    
    return x_global, w_global, endpoints

def compress_sequence_of_functions(function_family, x, w, precision):
    A = np.column_stack([phi(x)*np.sqrt(w) for phi in function_family.functions])
    Q,R,perm = sp.linalg.qr(A, pivoting=True)
    rank = np.sum(np.abs(np.diag(R)) > precision) + 1
    ## Construct rank revealing QR s.t. sp.linalg.norm(A[:,perm] - Q[:,:k]@R[:k,:]) <= precision
    U = Q[:,:rank]*(np.sqrt(w)[:,np.newaxis])**(-1)
    return U, A, rank

def construct_Chevyshev_quadratures(x,w,U):
    r = U.T@w
    (_,k) = U.shape
    
    B = U.T*np.sqrt(w)
    Q,R,perm = sp.linalg.qr(B, pivoting=True)
    z = np.linalg.solve(R[:k,:k],Q.T@r) ## No Hermetian transpose in python? For now only support real functions.
    
    x = x[perm]
    w = z*np.sqrt(w[perm])
    return x,w
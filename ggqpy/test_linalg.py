import numpy as np
import sympy
from pytest import approx
from ggqpy import *
from ggqpy.quad import Quadrature

def test_qr():
    precision = 1e-11
    m,n = 100,100
    A = np.random.normal(loc=0.0, scale=100.0, size=(m,n))
    # A[:,6:20] = A[:,0][:,np.newaxis]
    Q, R, perm = sp.linalg.qr(A, pivoting=True, mode="economic")
    k = np.sum(np.abs(np.diag(R)) > precision)
    print(np.diag(R))
    print(sp.linalg.norm(A[:,perm] - Q[:,:k]@R[:k,:]))
    tol = (n-k+1)/np.sqrt(2)*precision
    print(k)
    # assert k < n
    assert sp.linalg.norm(A[:,perm] - Q@R) <= 1e-11
    assert sp.linalg.norm(A[:,perm] - Q[:,:k]@R[:k,:]) <= tol
import numpy as np
from ggqpy.optimize import *
from pytest import approx

def test_sherman_morrison():
    A = np.random.random(size=(5,5))
    u = np.random.random(size=5)
    v = np.random.random(size=5)

    Ainv = np.linalg.inv(A)
    Ainv = np.asfortranarray(Ainv)

    sherman_morrison(Ainv,u,v)
    assert(np.linalg.norm(Ainv - np.linalg.inv(A + np.outer(u,v))) == approx(0))
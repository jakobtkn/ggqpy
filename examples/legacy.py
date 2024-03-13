def Legendre_derivative(x, I):
    n = len(x)
    A = np.polynomial.legendre.legder(np.eye(n), m=1)
    V1 = np.polynomial.legendre.legvander(x, n - 1)
    V2 = np.polynomial.legendre.legvander(x, n - 2)
    D = (I.length() / 2.0) * V2 @ A @ np.linalg.inv(V1)
    return D
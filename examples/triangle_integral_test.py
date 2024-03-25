import sys
import os
import sympy
sys.path.append(os.path.abspath("."))
from ggqpy import *


def main(alpha):
    r0 = alpha
    theta0 = np.pi / 2
    gamma = (
        lambda u: r0
        * np.sin(theta0)
        / (r0 * np.sin(theta0 - theta0 * u) + np.sin(theta0 * u))
    )
    f = lambda r, theta: np.cos(2 * theta) / r

    order = 4

    def evaluate_integrand(u):
        gammau = gamma(u)
        x, w = np.polynomial.legendre.leggauss(order)
        gl = Quadrature(gammau * (x + 1) / 2, w * gammau / 2)
        return f(gl.x, theta0 * u) @ gl.w

    # quad = Quadrature.load_from_file(f"quads/nystrom.8.{order}.quad")
    F = FunctionFamily.nystrom_integral_functions(
        number_of_discretizations=5,
        order=7,
        amin=r0 - 1e-3,
        amax=r0 + 1e-3,
        bmin=theta0 - 1e-3,
        bmax=theta0 + 1e-3,
    )
    x, w = generalized_gaussian_quadrature(
        F,
        min_length=1e-8,
        eps_disc=1e-12,
        eps_comp=1e-10,
        eps_quad=1e-7,
        interpolation_degree=30,
    )
    quad = Quadrature(x, w)
    quad.save_to_file("quads/test.quad")
    sum = np.array([evaluate_integrand(x) for x in quad.x]) @ quad.w

    print(sum)
    return


if __name__ == "__main__":
    main(0.5)

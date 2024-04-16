import numpy as np
import sympy
from pytest import approx
from ggqpy import *

np.seterr(all="raise")


def test_radial_transform():
    a = (-1.0, 0)
    b = (0, 2.0)
    alpha, phi, A, Ainv, det = standard_radial_triangle_transform(a, b)
    assert alpha == approx(1 / 2.0)
    assert phi == approx(np.pi / 2.0)
    np.testing.assert_allclose(A @ b, np.array([1, 0]), atol=1e-15)
    np.testing.assert_allclose(
        A @ a, np.array([alpha * np.cos(phi), alpha * np.sin(phi)]), atol=1e-15
    )
    np.testing.assert_allclose(Ainv @ np.array([1, 0]), b, atol=1e-15)

    a = (-1.0, 0)
    b = (0, 2.0)
    alpha, phi, A, Ainv, det = standard_radial_triangle_transform(a, b)
    assert alpha == approx(1 / 2.0)
    assert phi == approx(np.pi / 2.0)

    a = (-np.pi, 0)
    b = (0, 1.0)
    alpha, phi, A, Ainv, det = standard_radial_triangle_transform(a, b)
    assert alpha == approx(1 / np.pi)
    assert phi == approx(np.pi / 2.0)

    a = (-1.0, 1.0)
    b = (0, 1.0)
    alpha, phi, A, Ainv, det = standard_radial_triangle_transform(a, b)
    assert alpha == approx(1 / np.sqrt(2))
    assert phi == approx(3 * np.pi / 4.0)


def test_conformal_mapping():
    rho = lambda x: np.array(1, x[0], x[1] ** 2)
    drho = lambda x: np.array([[0, 0], [1, 0], [0, 2 * x[1]]])
    x0 = np.array([2.0, 1.0])

    affine, affine_inverse = ensure_conformal_mapping(drho, x0)

    np.testing.assert_allclose(affine(np.array([0, 0])), x0)


def test_geometry():
    R = Rectangle((-1, -1), (1, -1), (1, 1), (-1, 1))
    assert [*R.split_into_triangles_around_point((0, 0))] == [
        Triangle((0, 0),(-1, -1),(1, -1)),
        Triangle((0, 0),(1, -1),(1, 1)),
        Triangle((0, 0),(1, 1),(-1, 1)),
        Triangle((0, 0),(-1, 1),(-1, -1)),
    ]

from examples.experiment_triangle import analytic_integral

def test_quad_on_standard_triangle():
    r,theta,w = quad_on_standard_triangle(0.5, np.pi/2)
    print(np.shape(r), np.shape(theta), np.shape(w))
    assert len(r) == len(theta) == len(w)
    f = lambda r, theta: np.cos(2 * theta)
    assert f(r,theta)@w == approx(analytic_integral(0.5))

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

    np.testing.assert_allclose(A @ Ainv, np.identity(2))
    np.testing.assert_allclose(A @ b, np.array([1, 0]), atol=1e-15)
    np.testing.assert_allclose(
        A @ a, np.array([alpha * np.cos(phi), alpha * np.sin(phi)]), atol=1e-15
    )
    np.testing.assert_allclose(Ainv @ np.array([1, 0]), b, atol=1e-15)
    np.testing.assert_allclose(
        Ainv @ np.array([alpha * np.cos(phi), alpha * np.sin(phi)]), a, atol=1e-15
    )

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
    assert phi == approx(np.pi / 4.0)

    a = (-9.0, 3)
    b = (-20, 2.0)
    alpha, phi, A, Ainv, det = standard_radial_triangle_transform(a, b)

    np.testing.assert_allclose(A @ Ainv, np.identity(2), atol=1e-8)
    np.testing.assert_allclose(A @ b, np.array([1, 0]), atol=1e-15)
    np.testing.assert_allclose(
        A @ a, np.array([alpha * np.cos(phi), alpha * np.sin(phi)]), atol=1e-15
    )
    np.testing.assert_allclose(Ainv @ np.array([1, 0]), b, atol=1e-15)
    np.testing.assert_allclose(
        Ainv @ np.array([alpha * np.cos(phi), alpha * np.sin(phi)]), a, atol=1e-15
    )


def test_conformal_mapping():
    rho = lambda x: np.array(1, x[0], x[1] ** 2)
    drho = lambda x: np.array([[0, 0], [1, 0], [0, 2 * x[1]]])
    x0 = np.array([2.0, 1.0])

    B, Binv = ensure_conformal_mapping(drho, x0)

    np.testing.assert_allclose(B @ (np.array([0, 0])) + x0, x0)
    np.testing.assert_allclose(B @ Binv, np.identity(2))


def test_geometry():
    R = Rectangle((-1, -1), (1, -1), (1, 1), (-1, 1))
    assert [*R.split_into_triangles_around_point((0, 0))] == [
        Triangle((0, 0), (-1, -1), (1, -1)),
        Triangle((0, 0), (1, -1), (1, 1)),
        Triangle((0, 0), (1, 1), (-1, 1)),
        Triangle((0, 0), (-1, 1), (-1, -1)),
    ]


from examples.experiment_triangle import analytic_integral
def test_quad_on_standard_triangle():
    r, theta, w = quad_on_standard_triangle(0.5, np.pi / 2)
    assert len(r) == len(theta) == len(w)
    f = lambda r, theta: np.cos(2 * theta)
    assert f(r, theta) @ w == approx(analytic_integral(0.5))


def test_singular_integral(plt):
    drho = lambda x: np.array([[0, 0], [1, 0], [0, 2 * x[1]]])
    x0 = np.array([0.5, 0.5])
    simplex = Rectangle((-1, -1), (1, -1), (1, 1), (-1, 1))
    x, y, w = singular_integral_quad(drho, x0, simplex)
    plt.figure()
    plt.scatter(x, y, marker="x")
    plt.scatter(x0[0], x0[1], c="r")
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)

def test_singular_integral_2(plt):
    drho = lambda x: np.array([[0, 0], [1, 0+ x[0]], [0, 2 * x[1] + x[0]]])
    x0 = np.array([-0.1, 0.5])
    simplex = Rectangle((-1, -1), (-1, 1), (1, 1), (1, -1))
    x, y, w = singular_integral_quad(drho, x0, simplex)
    plt.figure()
    plt.scatter(x, y, marker="x")
    plt.scatter(x0[0], x0[1], c="r")
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)


def test_node_placement(plt):
    drho = lambda x: np.array([[0, 0], [1, 0], [0, 2 * x[1]]])
    x0 = np.array([0.5, 0.5])
    simplex = Rectangle((-1, -1), (1, -1), (1, 1), (-1, 1))

    B, Binv = ensure_conformal_mapping(drho, x0)
    R = Rectangle(*[Binv @ (np.array(v) - x0) for v in iter(simplex)])

    T, _, _, _ = [*R.split_into_triangles_around_point(np.array([0, 0]))]

    scale, angle, A, Ainv, det = standard_radial_triangle_transform(
        T.vertices[1], T.vertices[2]
    )
    T0 = Triangle((0, 0), (1, 0), (scale * np.cos(angle), scale * np.sin(angle)))
    r, theta, w = quad_on_standard_triangle(scale, angle)
    x_local = np.cos(theta) * r
    y_local = np.sin(theta) * r

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title("T0")
    for x, y in [*T0]:
        plt.scatter(x, y, c="r")
    for x, y in zip(x_local, y_local):
        plt.scatter(x, y, c="b")
        assert T0.is_in((x, y))

    v = np.row_stack([x_local, y_local])
    np.testing.assert_allclose(v[0, :], x_local)
    np.testing.assert_allclose(v[1, :], y_local)
    print(v)
    plt.subplot(1, 2, 2)
    plt.title("T")
    for x, y in [*T]:
        plt.scatter(x, y, c="r")

    for x, y in zip((Ainv @ v)[0, :], (Ainv @ v)[1, :]):
        plt.scatter(x, y, c="b")
        assert T.is_in((x,y))

    v = B @ (Ainv @ v) + x0[:, np.newaxis]
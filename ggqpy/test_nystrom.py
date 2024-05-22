import numpy as np
from numpy.polynomial.legendre import leggauss
import sympy
from pytest import approx
from ggqpy import *
from ggqpy.geometry import *
from ggqpy.nystrom import *

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

    assert det == approx(abs(np.linalg.det(A)))
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
    drho = lambda s, t: np.array([[0, 0], [1, 0], [0, 2 * t]])
    x0 = np.array([2.0, 1.0])

    B, Binv = ensure_conformal_mapping(drho, x0)

    np.testing.assert_allclose(B @ (np.array([0, 0])) + x0, x0)
    np.testing.assert_allclose(B @ Binv, np.identity(2))


def test_geometry():
    R = Quadrilateral((-1, -1), (1, -1), (1, 1), (-1, 1))
    assert [*R.split_into_triangles_around_point((0, 0))] == [
        Triangle((0, 0), (-1, -1), (1, -1)),
        Triangle((0, 0), (1, -1), (1, 1)),
        Triangle((0, 0), (1, 1), (-1, 1)),
        Triangle((0, 0), (-1, 1), (-1, -1)),
    ]


from examples.experiment_triangle import analytic_integral
solver = QuadratureLoader(order = 4)

def test_quad_on_standard_triangle():
    r, theta, w = solver.quad_on_standard_triangle(0.5, np.pi / 2)
    assert len(r) == len(theta) == len(w)
    f = lambda r, theta: np.cos(2 * theta) / r
    assert f(r, theta) @ w == approx(analytic_integral(0.5))


def test_singular_integral_0(plt):
    drho = lambda s, t: np.array([[1, 0], [0, 1], [0, 0]])
    x0 = np.array([0.5, 0.0])
    simplex = Quadrilateral((-1, -1), (-1, 1), (1, 1), (1, -1))
    x, y, w = solver.singular_integral_quad(drho, x0, simplex)

    plt.figure()
    plt.title(f"Nodes = {len(x)}")
    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
    ax.stem(x, y, w)

    assert np.sum(w) == approx(4.0)


def test_singular_integral_1(plt):
    drho = lambda s, t: np.array([[0, 0], [1, 0], [0, 3 * t]])
    x0 = np.array([0.5, 0.5])
    simplex = Quadrilateral((-1, -1), (1, -1), (1, 1), (-1, 1))
    x, y, w = solver.singular_integral_quad(drho, x0, simplex)
    plt.figure()
    plt.title(f"Nodes = {len(x)}")
    plt.scatter(x, y, marker="x")
    plt.scatter(x0[0], x0[1], c="r")
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)


def test_singular_integral_2(plt):
    drho = lambda s, t: np.array([[0, 0], [1, 0 + s], [0, 2 * t + s]])
    x0 = np.array([-0.1, 0.5])
    simplex = Quadrilateral((-1, -1), (-1, 1), (1, 1), (1, -1))
    x, y, w = solver.singular_integral_quad(drho, x0, simplex)
    plt.figure()
    plt.title(f"Nodes = {len(x)}")
    plt.scatter(x, y, marker="x")
    plt.scatter(x0[0], x0[1], c="r")
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)


def test_singular_integral_3(plt):
    a = 4.0
    b = 2.0
    # drho = lambda s, t: np.array([[a, 0], [0, b], [0, 0]])

    theta = 0.8
    R = np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ]
    )

    def drho(s, t):
        return R @ np.array([[a, 0], [0, b], [0, 0]])

    def jacobian(s, t):
        J = drho(s, t)
        return np.sqrt(np.linalg.det(J.T @ J))

    rho = lambda s, t: R @ np.array([a * s, b * t, 0])
    x0 = np.array([-0.0, 0.8])

    B, Binv = ensure_conformal_mapping(drho, x0)
    simplex = Quadrilateral((-1, -1), (-1, 1), (1, 1), (1, -1))
    x, y, w = solver.singular_integral_quad(drho, x0, simplex)
    assert np.sum(w * jacobian(x, y)) == approx(4.0 * a * b)
    assert np.sum(
        (w * jacobian(x, y))[
            np.linalg.norm(np.column_stack([rho(s, t) for s, t in zip(x, y)]), axis=0)
            < 2
        ]
    ) == approx(4 * np.pi, abs=1.0)
    plt.figure()
    plt.title(f"Nodes = {len(x)}")
    plt.scatter(x, y, marker="x")
    plt.scatter(x0[0], x0[1], c="r")
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)


def test_node_placement(plt):
    drho = lambda s, t: np.array([[0, 0], [1, 0], [0, 2 * t]])
    x0 = np.array([0.5, 0.5])
    simplex = Quadrilateral((-1, -1), (1, -1), (1, 1), (-1, 1))

    B, Binv = ensure_conformal_mapping(drho, x0)
    R = Quadrilateral(*[Binv @ (np.array(v) - x0) for v in iter(simplex)])

    T, _, _, _ = [*R.split_into_triangles_around_point(np.array([0, 0]))]

    scale, angle, A, Ainv, det = standard_radial_triangle_transform(
        T.vertices[1], T.vertices[2]
    )
    T0 = Triangle((0, 0), (1, 0), (scale * np.cos(angle), scale * np.sin(angle)))
    r, theta, w = solver.quad_on_standard_triangle(scale, angle)
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
        assert T.is_in((x, y))

    v = B @ (Ainv @ v) + x0[:, np.newaxis]


def test_area_sphere():
    M = 4
    N = 10
    s, ws = leggauss(M)
    t, wt = leggauss(N)

    gls = Quadrature.gauss_legendre_on_interval(M, Interval(0, 2 * np.pi))
    glt = Quadrature.gauss_legendre_on_interval(N, Interval(0, np.pi))
    s, ws = gls.x, gls.w
    t, wt = glt.x, glt.w

    rho = lambda s, t: np.array(
        [np.sin(t) * np.cos(s), np.sin(t) * np.sin(s), np.cos(t)]
    )
    drho = lambda s, t: np.array(
        [
            [-np.sin(t) * np.sin(s), np.cos(t) * np.cos(s)],
            [np.sin(t) * np.cos(s), np.cos(t) * np.sin(s)],
            [0, -np.sin(t)],
        ]
    )

    def normal(p):
        return p / np.linalg.norm(p, axis=0)

    def jacobian(s, t):
        return np.sin(t)

    simplex = Quadrilateral((0, 0), (2 * np.pi, 0), (2 * np.pi, np.pi), (0, np.pi))
    ss, tt = np.meshgrid(s, t)
    ss, tt = ss.flatten(), tt.flatten()

    xs, yt, w = solver.singular_integral_quad(drho, np.array([2.0, 2.0]), simplex)

    assert np.sum(w * jacobian(xs, yt)) == approx(4 * np.pi)


def test_rectangle():
    I = Interval(0, 3)
    J = Interval(2, 4)
    R = Rectangle(I, J)
    assert [*R] == [(0, 2), (3, 2), (3, 4), (0, 4)]

from ggqpy.parametrization import *
import sympy
def test_conformal2():
    s,t = sympy.Symbol('s'), sympy.Symbol('t')
    param = Parametrization(s,t,s,s*t**10,s*t + 10)
    param = Parametrization.droplet()
    rho,drho,jacobian,normal = param.get_lambdas()
    x0 = np.array([np.pi,0.7])
    A, Ainv = ensure_conformal_mapping(drho,x0)
    for x in [np.array([0,0]),np.array([2*np.pi,0]),np.array([2*np.pi,np.pi]),np.array([0,np.pi])]:
        print(Ainv @ (x - x0))
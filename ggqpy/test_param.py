import numpy as np
from ggqpy import *
from ggqpy.parametrization import *

from pytest import approx
np.seterr(all="raise")


def test_droplet(plt):
    param = Parametrization.droplet()
    rho, drho, jacobian, normal = param.get_lambdas()
    s = np.linspace(0.1, 2 * np.pi - 0.1, 20)
    t = np.linspace(0.1, np.pi - 0.1, 10)
    s, t = np.meshgrid(s, t)
    s, t = s.flatten(), t.flatten()

    x0 = np.array([0.5, 0.5])
    B, Binv = ensure_conformal_mapping(drho, x0)


    print([Binv@(np.array([*x]) - x0) for x in [(0,0),(2*np.pi,0), (2*np.pi, np.pi), (0,np.pi)]])
    print(rho(*x0))
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(*rho(s, t))

    sk, tk = s[10:102], t[10:102]
    ax.quiver(*rho(sk, tk), *normal(sk, tk) * 0.2)


def test_sphere(plt):
    param = Parametrization.sphere()
    rho, drho, jacobian, normal = param.get_lambdas()
    s = np.linspace(0.1, 2 * np.pi - 0.1, 20)
    t = np.linspace(0.1, np.pi - 0.1, 10)
    s, t = np.meshgrid(s, t)
    s, t = s.flatten(), t.flatten()

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(*rho(s, t))

    sk, tk = s[10:102], t[10:102]
    ax.quiver(*rho(sk, tk), *normal(sk, tk) * 0.2)

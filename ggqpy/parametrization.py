import sympy as sp
import numpy as np

verbose = False


def _de_dimension(_f):  ## Correct for fixed 2d arrays in sympy
    def f(*args):
        return np.squeeze(_f(*args))

    return f


class Parametrization:
    def __init__(self, s, t, x, y, z, flip_normal=False):
        self.s = s
        self.t = t
        self.x = x
        self.y = y
        self.z = z

        self.rho = sp.Matrix([x, y, z])
        self.variables = sp.Matrix([s, t])
        self.drho = self.rho.jacobian(self.variables)

        cross_product = self.drho[:, 0].cross(self.drho[:, 1])
        
        self.jacobian = cross_product.norm()
        if flip_normal:
            self.normal = cross_product.normalized()
        else:
            self.normal = -cross_product.normalized()

    def get_lambdas(self):
        s, t = self.s, self.t
        _rho = sp.lambdify((s, t), self.rho, modules="numpy")
        _drho = sp.lambdify((s, t), self.drho, modules="numpy")
        _jacobian = sp.lambdify((s, t), self.jacobian, modules="numpy")
        _normal = sp.lambdify((s, t), self.normal, modules="numpy")
        for f in [_rho, _drho, _jacobian, _normal]:
            yield _de_dimension(f)

    @classmethod
    def sphere(cls):
        s = sp.symbols("s", real=True, domain=(0, 2 * sp.pi))
        t = sp.symbols("t", real=True, domain=(0, sp.pi))

        x = sp.cos(s) * sp.sin(t)
        y = sp.sin(s) * sp.sin(t)
        z = sp.cos(t)
        return cls(s, t, x, y, z, flip_normal=False)

    @classmethod
    def droplet(cls):
        s = sp.symbols("s", real=True, domain=(0, 2 * sp.pi))
        t = sp.symbols("t", real=True, domain=(0, sp.pi))

        x = 2 * sp.sin(t / 2)
        y = sp.cos(s) * sp.sin(t)
        z = sp.sin(s) * sp.sin(t)

        return cls(s, t, x, y, z, flip_normal=True)

    def h_and_hgrad(self, k=1.0, p0=[1, 0, 0]):
        p0 = sp.Matrix(p0)
        x, y, z = sp.symbols("x y z", real=True)

        h = (
            sp.exp(sp.I * k * (sp.Matrix([x, y, z]) - p0).norm())
            / (sp.Matrix([x, y, z]) - p0).norm()
        )
        h_grad = sp.Matrix([h.diff(var) for var in [x, y, z]])

        _h = sp.lambdify((x, y, z), h, modules="numpy")
        _h_grad = sp.lambdify((x, y, z), h_grad, modules="numpy")
        for f in [_h, _h_grad]:
            yield _de_dimension(f)

    def print(self):
        print("drho:")
        print(self.drho)
        print("Jacobian:")
        print(self.jacobian)
        print("normal:")
        print(self.normal)
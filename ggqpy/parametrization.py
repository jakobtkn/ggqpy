import sympy as sp
import numpy as np

verbose = False


def _de_dimension(_f):  ## Correct for fixed 2d arrays in sympy
    def f(*args):
        return np.squeeze(_f(*args))

    return f


class Parametrization:
    def __init__(self, s, t, x, y, z):
        self.s = s
        self.t = t
        self.x = x
        self.y = y
        self.z = z

        self.rho = sp.Matrix([x, y, z])
        self.variables = sp.Matrix([s, t])
        self.drho = self.rho.jacobian(self.variables)
        
        cross_product = self.drho[:, 0].cross(self.drho[:, 1])
        self.jacobian = sp.simplify(cross_product.norm())
        self.normal = -cross_product.normalized()

    def get_lambdas(self):
        s,t = self.s,self.t
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
        return cls(s,t,x,y,z)

    @classmethod
    def droplet(cls):
        s = sp.symbols("s", real=True, domain=(0, 2 * sp.pi))
        t = sp.symbols("t", real=True, domain=(0, sp.pi))

        x = 2 * sp.sin(t / 2)
        y = sp.cos(s) * sp.sin(t)
        z = sp.sin(s) * sp.sin(t)
        return cls(s,t,x,y,z)
    
    def directional_derivative_h(self, p0 = sp.Matrix([10,0,0])):
        h = sp.exp(sp.I * (self.rho - p0).norm()) / (self.rho - p0).norm()
        gradient = sp.Matrix([sp.diff(h, var) for var in self.variables])

        h_derivative = gradient.dot(self.normal)
        _h = sp.lambdify(self.variables, h, modules="numpy")
        
        _h_derivative = sp.lambdify(self.variables, h_derivative, modules="numpy")
        for f in [_h, _h_derivative]:
            yield _de_dimension(f)
    
    def print(self):
        print("drho:")
        print(drho)
        print("Jacobian:")
        print(jacobian)
        print("normal:")
        print(normal)

def parametrize_sphere():
    s = sp.symbols("s", real=True, domain=(0, 2 * sp.pi))
    t = sp.symbols("t", real=True, domain=(0, sp.pi))

    x = sp.cos(s) * sp.sin(t)
    y = sp.sin(s) * sp.sin(t)
    z = sp.cos(t)

    rho = sp.Matrix([x, y, z])
    return determine_parametrization(rho, s, t)





def determine_parametrization(rho, s, t):
    variables = sp.Matrix([s, t])
    drho = rho.jacobian(variables)
    cross_product = drho[:, 0].cross(drho[:, 1])
    jacobian = sp.simplify(cross_product.norm())
    normal = -cross_product.normalized()

    if verbose:
        print("drho:")
        print(drho)
        print("Jacobian:")
        print(jacobian)
        print("normal:")
        print(normal)

    _rho = sp.lambdify((s, t), rho, modules="numpy")
    _drho = sp.lambdify((s, t), drho, modules="numpy")
    _jacobian = sp.lambdify((s, t), jacobian, modules="numpy")
    _normal = sp.lambdify((s, t), normal, modules="numpy")

    for f in [_rho, _drho, _jacobian, _normal]:
        yield _de_dimension(f)


if __name__ == "__main__":
    rho, drho, jacobian, normal = Parametrization.sphere().get_lambdas()
    print(normal(2, 3).shape)
    print(normal(np.array([2, 2, 6, 0, 1, 1]), np.array([3, 3, 0, 0, 1, 1])))
    # print(normal(np.array([2]),np.array([3])))
    # print(normal((2,2),(3,3)))
    # print(normal(2,3))

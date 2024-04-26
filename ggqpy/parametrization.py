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
    
    def h_and_hgrad(self, p0 = [1,0,0]):
        p0 = sp.Matrix(p0)
        x,y,z = sp.symbols("x y z", real=True)

        h = sp.exp(sp.I * (sp.Matrix([x,y,z]) - p0).norm()) / (sp.Matrix([x,y,z]) - p0).norm()
        h_grad = sp.simplify(sp.Matrix([h.diff(var) for var in [x,y,z]]))

        _h = sp.lambdify((x,y,z), h, modules="numpy")        
        _h_grad = sp.lambdify((x,y,z), h_grad, modules="numpy")
        for f in [_h, _h_grad]:
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
    param = Parametrization.droplet()
    rho, drho, jacobian, normal = param.get_lambdas()
    h, h_grad = param.h_and_hgrad()
    # print(h_grad(1,1,1))
    # print(h(1,1,1))
    s = np.arange(10)
    t = np.arange(10)
    x,y,z = rho(s,t)
    print(normal(0,1))
    print(h_grad(*rho(0,1)))

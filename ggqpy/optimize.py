import numpy as np


def dampened_gauss_newton(r, jac, x0, step_size=0.3, maxiter=100, tol=1e-6):
    x = x0
    for _ in range(maxiter):
        J = jac(x)
        rx = r(x)

        # d = np.linalg.solve((J.T@J), J.T@rx)
        d, _, _, _ = np.linalg.lstsq(J, rx, rcond=None)
        x = x - step_size * d
        if np.linalg.norm(d) < tol:
            break

    return x


class QuadOptimizer:
    u_list = None
    du_list = None
    r = None
    step_size = 0.8
    maxiter = int(1e3)
    tol = 1e-8
    args = (step_size, maxiter, tol)
    verbose = False
    rank = None

    def __init__(self, u_list, r) -> None:
        self.u_list = u_list
        self.du_list = [u.deriv() for u in u_list]
        self.r = r
        self.rank = len(u_list)
        return

    def set_parameters(self, step_size, maxiter, tol):
        self.step_size = step_size
        self.maxiter = maxiter
        self.tol = tol
        self.args = (step_size, maxiter, tol)

    def jacobian(self, y):
        x, w = np.split(y, 2)
        U = np.column_stack([u(x) for u in self.u_list])
        dU = np.column_stack([du(x) for du in self.du_list])
        J = np.hstack([dU.T * w, U.T])
        return J

    def residual(self, y):
        x, w = np.split(y, 2)
        U = np.column_stack([u(x) for u in self.u_list])
        return U.T @ w - self.r

    def naive_optimization(self, n, I):
        x_gl,w_gl = np.polynomial.legendre.leggauss(n)
        w_gl = w_gl*0.5*I.length()
        x_gl = I.translate(x_gl)
        y0 = np.concatenate([x_gl,w_gl])
        y = dampened_gauss_newton(self.residual, self.jacobian, y0, *self.args)
        x, w = np.split(y, 2)
        return x, w

    def rank_remaining_nodes(self, x, w):
        y0 = np.concatenate([x, w])
        rx = self.residual(y0)

        n = len(x)

        eta = np.zeros(n)
        for k in range(len(x)):
            y = np.delete(y0, (k, n + k))
            Jk = self.jacobian(y)
            rx = self.residual(y)

            d, _, _, _ = np.linalg.lstsq(Jk, rx, rcond=None)
            eta[k] = np.linalg.norm(d)

        idx_sorted = np.argsort(eta)

        return idx_sorted

    def attempt_to_remove_node(self, x, w, eps_quad):
        n = len(x)

        idx_sorted = self.rank_remaining_nodes(x, w)
        for k in idx_sorted:
            mask = np.full(n, True)
            mask[k] = False
            y0 = np.concatenate([x[mask], w[mask]])

            y = dampened_gauss_newton(self.residual, self.jacobian, y0, *self.args)
            eps = np.linalg.norm(self.residual(y)) ** 2

            if eps < eps_quad**2:
                x, w = np.split(y, 2)
                return x, w, True

            if self.verbose:
                print("No improvement found")

        return x, w, False

    def point_reduction(self, x0, w0, eps_quad):
        x = x0
        w = w0

        improvement_found = True
        while improvement_found:
            x, w, improvement_found = self.attempt_to_remove_node(x, w, eps_quad)
            if len(x) <= self.rank // 2:
                break

        return x, w

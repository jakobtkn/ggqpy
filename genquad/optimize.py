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

    def naive_optimization(self, n, interval, args):
        y0 = np.concatenate([np.linspace(*interval, n), np.ones(n)])
        y = dampened_gauss_newton(self.residual, self.jacobian, y0, *args)
        x, w = np.split(y, 2)
        return x, w

    def __init__(self, u_list, r) -> None:
        self.u_list = u_list
        self.du_list = [u.deriv() for u in u_list]
        self.r = r
        return

    def rank_remaining_nodes(self, x, w):
        J = self.jacobian(np.concatenate([x, w]))
        U = np.column_stack([u(x) for u in self.u_list])
        n = len(x)

        step_directions = list()
        eta = np.zeros(n)
        for k in range(len(x)):
            Jk = J
            Jk[:, k] = 0
            Jk[:, n + k] = 0

            delta_r = U[k, :] * w[k]
            delta_xk, _, _, _ = np.linalg.lstsq(
                Jk, -delta_r, rcond=-1
            )  # Improve by using SMW
            eta[k] = np.linalg.norm(delta_xk)
            step_directions.append(delta_xk)

        idx_sorted = np.argsort(eta)

        return idx_sorted

    def attempt_to_remove_node(self, x, w, eps_quad):
        n = len(x)
        step_size = 0.2

        idx_sorted = self.rank_remaining_nodes(x,w)
        for k in idx_sorted:
            mask = np.full(n, True)
            mask[k] = False
            y0 = np.concatenate([x[mask], w[mask]])

            y = dampened_gauss_newton(self.residual, self.jacobian, y0, step_size)
            eps = np.linalg.norm(self.residual(y)) ** 2

            if eps < eps_quad**2:
                x, w = np.split(y, 2)
                return x, w, True

        return x, w, False


    def point_reduction(self, x0, w0, eps_quad):
        x = x0
        w = w0
        
        improvement_found = True
        while improvement_found:
            x, w, improvement_found = self.attempt_to_remove_node(
                x, w, eps_quad
            )

        return x, w

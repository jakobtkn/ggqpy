import numpy as np
import scipy as sp
from tqdm import tqdm


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


def sherman_morrison(Ainv, u, v):
    """finds (A + uv^T)^-1 using a rank-1 update"""
    assert Ainv.flags["F_CONTIGUOUS"]
    Ainvu = Ainv @ u
    alpha = -1 / (1 + v.T @ Ainvu)
    sp.linalg.blas.dger(alpha, Ainvu, v.T @ Ainv, a=Ainv, overwrite_a=1)


class QuadOptimizer:
    u_list = None
    du_list = None
    r = None
    step_size = 0.8
    maxiter = int(1e3)
    tol = 1e-5
    args = (step_size, maxiter, tol)
    verbose = False
    rank = None

    def __init__(self, u_list, r, verbose=False) -> None:
        self.u_list = u_list
        self.du_list = [u.derivative() for u in u_list]
        self.r = r
        self.rank = len(u_list)
        self.verbose = verbose
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
        print(J)
        return J

    def residual(self, y):
        x, w = np.split(y, 2)
        U = np.column_stack([u(x) for u in self.u_list])
        return U.T @ w - self.r

    def naive_optimization(self, n, I):
        x_gl, w_gl = np.polynomial.legendre.leggauss(n)
        w_gl = w_gl * 0.5 * I.length()
        x_gl = I.translate(x_gl)
        y0 = np.concatenate([x_gl, w_gl])
        y = dampened_gauss_newton(self.residual, self.jacobian, y0, *self.args)
        x, w = np.split(y, 2)
        return x, w

    def rank_remaining_nodes(self, x, w):
        y0 = np.concatenate([x, w])
        J = self.jacobian(y0)
        A = np.linalg.inv(J @ np.transpose(J))
        A = np.asfortranarray(A)
        n = len(x)

        eta = np.zeros(n)
        for k in range(len(x)):
            rk = J[:, n + k] * w[k]

            Ak = A
            sherman_morrison(Ak, -J[:, k], J[:, k])
            sherman_morrison(Ak, -J[:, k + n], J[:, k + n])
            
            Jk = J
            Jk[:, (k, n + k)] = 0

            d = Jk.T @ Ak @ rk
            eta[k] = np.linalg.norm(d)

        idx_sorted = np.argsort(eta)

        return idx_sorted

    def attempt_to_remove_node(self, x, w, eps_quad):
        n = len(x)

        idx_sorted = self.rank_remaining_nodes(x, w)
        for (iteration,k) in enumerate(idx_sorted):
            mask = np.full(n, True)
            mask[k] = False
            y0 = np.concatenate([x[mask], w[mask]])

            # y = dampened_gauss_newton(self.residual, self.jacobian, y0, *self.args)
            res = sp.optimize.least_squares(
                self.residual, y0, jac=self.jacobian, method="dogbox", x_scale=1, ftol=self.tol
            )
            y = res.x

            eps = np.linalg.norm(self.residual(y)) ** 2

            if eps < eps_quad**2:
                x, w = np.split(y, 2)
                if self.verbose:
                    print("Removed node ", k, "This was the ",iteration, " checked")
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

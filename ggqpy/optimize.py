import numpy as np
import scipy as sp
from scipy.optimize import least_squares
from tqdm import tqdm

from ggqpy.utils import PiecewiseLegendreFamily

verbose = False
if verbose:

    def vprint(*messages) -> None:
        for message in messages:
            print(message)
        print()
        return

else:

    def vprint(*messages) -> None:
        return


def levenberg_marquadt(residual, jacobian, x0, maxiter=1000, tol=1e-6):
    x = x0
    for _ in range(maxiter):
        J = jacobian(x)
        r = residual(x)

        A = J.T @ J
        g = J.T @ r


def dampened_gauss_newton(r, jac, x0, step_size=0.3, maxiter=1000, tol=1e-6):

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


def sherman_morrison(Ainv, u, v) -> None:
    """
    Finds (A + uv^T)^{-1} using a rank-1 update

    Parameters
    ----------
    Ainv: Inverse of matrix A. Updated as  (A + uv^T)^{-1} in place
    u: Vector u
    v: Vector v

    """
    assert Ainv.flags["F_CONTIGUOUS"]
    Ainvu = Ainv @ u
    k = 1 + v.T @ Ainvu
    if abs(k) < 1e-16:
        print("Unlikely event?")
        return
    alpha = -1 / k
    sp.linalg.blas.dger(alpha, Ainvu, v.T @ Ainv, a=Ainv, overwrite_a=1)


class QuadOptimizer:
    def __init__(self, legendre_family: PiecewiseLegendreFamily, r, ftol=1e-8) -> None:
        self.legendre_family = legendre_family
        self.r = r
        self.rank = legendre_family.number_of_functions
        self.ftol = ftol

        self.verbose = verbose
        return

    def jacobian(self, y):
    
        n = len(y) // 2
        J = self.legendre_family.eval_block(y[:n])
        J[:, :n] = J[:, :n] * y[n:]
        return J

    def residual(self, y):
    
        n = len(y) // 2
        U = self.legendre_family(y[:n])
        return U @ y[n:] - self.r

    def naive_optimization(self, n, I, step_size, maxiter, ftol):
    
        x_gl, w_gl = np.polynomial.legendre.leggauss(n)
        w_gl = w_gl * 0.5 * I.length()
        x_gl = I.translate(x_gl)
        y0 = np.concatenate([x_gl, w_gl])
        y = dampened_gauss_newton(
            self.residual, self.jacobian, y0, step_size, maxiter, ftol
        )
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
            Ak = A
            sherman_morrison(Ak, -J[:, k], J[:, k])
            sherman_morrison(Ak, -J[:, k + n], J[:, k + n])

            Jk = J
            Jk[:, (k, n + k)] = 0

            rk = J[:, n + k] * w[k]
            d = Jk.T @ Ak @ rk
            eta[k] = np.linalg.norm(d)

        idx_sorted = np.argsort(eta)

        return idx_sorted

    def attempt_to_remove_node(self, x, w, eps_quad):
    
        n = len(x)

        def res_lm(y):
            residual = np.zeros(2 * (n - 1))
            residual[: self.rank] = self.residual(y)
            return residual

        def jac_lm(y):
            jac = np.zeros((2 * (n - 1), 2 * (n - 1)))
            jac[: self.rank, :] = self.jacobian(y)
            return jac

        idx_sorted = self.rank_remaining_nodes(x, w)
        for iteration, k in enumerate(idx_sorted):
            mask = np.full(n, True)
            mask[k] = False
            y0 = np.concatenate([x[mask], w[mask]])

            res = least_squares(
                res_lm,
                y0,
                jac=jac_lm,
                method="lm",
                x_scale="jac",
                max_nfev=50,
                verbose=self.verbose,
            )
            y = res.x
            eps = 2 * np.sqrt(res.cost)

            if eps < eps_quad:
                x, w = np.split(y, 2)

                vprint(f"Node removed, a total of {iteration + 1} have been checked")
                return x, w, True

            vprint(f"Node kept, a total of {iteration + 1} have been checked")

        return x, w, False

    def reduce_quadrature(self, x0, w0, eps_quad):
    
        x = x0
        w = w0

        for _ in tqdm(range(self.rank // 2)):
            x, w, improvement_found = self.attempt_to_remove_node(x, w, eps_quad)
            if not improvement_found:
                vprint(
                    f"Broke preemptively, generated a quadrature consisting of {len(x)} nodes (down from {len(x0)})"
                )
                return x, w

        idx = np.argsort(x)
        x = x[idx]
        w = w[idx]

        vprint(
            f"Succesfully generated a generalized Gaussian quadrature consisting of {len(x)} nodes"
        )
        return x, w

import numpy as np
import scipy as sp
from tqdm import tqdm

verbose = False
if verbose:

    def vprint(self, *messages) -> None:
        for message in messages:
            print(message)
        print()
        return

else:

    def vprint(self, *messages) -> None:
        return


def dampened_gauss_newton(r, jac, x0, step_size=0.3, maxiter=100, tol=1e-6):
    """

    Parameters
    ----------
    :
    Returns
    -------
    :
    """
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
    alpha = -1 / (1 + v.T @ Ainvu)
    sp.linalg.blas.dger(alpha, Ainvu, v.T @ Ainv, a=Ainv, overwrite_a=1)


class QuadOptimizer:
    def __init__(self, legendre_family, r, ftol=1e-8) -> None:
        self.legendre_family = legendre_family
        self.r = r
        self.rank = legendre_family.number_of_functions
        self.ftol = ftol

        self.verbose = verbose
        return

    def jacobian(self, y):
        """

        Parameters
        ----------
        :
        Returns
        -------
        :
        """
        x, w = np.split(y, 2)
        J = self.legendre_family.eval_block(x)
        J[:, : len(x)] = J[:, : len(x)] * w
        return J

    def residual(self, y):
        """

        Parameters
        ----------
        :
        Returns
        -------
        :
        """
        x, w = np.split(y, 2)
        U = self.legendre_family(x)
        return U @ w - self.r

    def naive_optimization(self, n, I, step_size, maxiter, ftol):
        """

        Parameters
        ----------
        :
        Returns
        -------
        :
        """
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
        """

        Parameters
        ----------
        :
        Returns
        -------
        :
        """
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
        """

        Parameters
        ----------
        :
        Returns
        -------
        :
        """
        n = len(x)

        idx_sorted = self.rank_remaining_nodes(x, w)
        for iteration, k in enumerate(idx_sorted):
            mask = np.full(n, True)
            mask[k] = False
            y0 = np.concatenate([x[mask], w[mask]])

            res = sp.optimize.least_squares(
                self.residual,
                y0,
                jac=self.jacobian,
                method="dogbox",
                x_scale=1,
                ftol=self.ftol,
                gtol=None,
                xtol=None,
                verbose=self.verbose,
            )
            y = res.x
            eps = 2 * res.cost

            if eps < eps_quad**2:
                x, w = np.split(y, 2)

                vprint(f"Node removed, a total of {iteration + 1} have been checked")
                return x, w, True

            vprint(f"Node kept, a total of {iteration + 1} have been checked")

        return x, w, False

    def reduce_quadrature(self, x0, w0, eps_quad):
        """

        Parameters
        ----------
        :
        Returns
        -------
        :
        """
        x = x0
        w = w0

        for _ in tqdm(range(self.rank // 2)):
            x, w, improvement_found = self.attempt_to_remove_node(x, w, eps_quad)
            if not improvement_found:
                vprint(
                    f"Succesfully generated a generalized Gaussian quadrature consisting of {len(x)} nodes"
                )
                break

        vprint(
            f"Broke preemptively, generated a quadrature consisting of {len(x)} nodes (down from {len(x0)})"
        )
        return x, w

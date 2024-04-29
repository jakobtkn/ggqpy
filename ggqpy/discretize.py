import numpy as np
import scipy as sp
import numpy.polynomial.legendre as legendre
from tqdm import tqdm
from ggqpy.utils import (
    PiecewiseLegendre,
    PiecewiseLegendreFamily,
    Interval,
    FunctionFamily,
)


def pairwise(iterable):
    it = iter(iterable)
    a = next(it, None)

    for b in it:
        yield (a, b)


class Discretizer:
    precision = None
    min_length = None
    verbose = True
    interpolation_degree = None
    endpoints = None

    def __init__(
        self, precision=1e-6, min_length=1e-4, interpolation_degree=30, verbose=False
    ) -> None:
        self.precision = precision
        self.min_length = min_length
        self.verbose = verbose
        self.interpolation_degree = interpolation_degree
        self.x_gl, _ = legendre.leggauss(2 * self.interpolation_degree)
        return

    def interval_compatible(self, I: Interval, function_family: FunctionFamily):
        """

        Parameters
        ----------
        :
        Returns
        -------
        :
        """
        if self.min_length is not None and I.length() < self.min_length:
            return True

        x = (I.b - I.a) * (self.x_gl + 1.0) / 2.0 + I.a
        testing_degree = 2 * self.interpolation_degree - 1
        A = np.column_stack([phi(x) for phi in function_family.functions_lambdas])

        alpha = legendre.legfit(
            self.x_gl, y=A, deg=testing_degree
        )  # Fit to Legendre Polynomials on [a,b]

        normalization_factor = np.sqrt((2 * np.arange(testing_degree + 1) + 1) / 2)
        alpha_normalized = alpha * normalization_factor[:, np.newaxis]

        high_freq_coefficients = alpha_normalized[self.interpolation_degree:, :]
        high_freq_sq_residuals = np.sqrt(
            np.sum(abs(high_freq_coefficients) ** 2, axis=0)
        )
        interval_weight = I.length() / function_family.I.length()
        interval_weight = 1.0

        is_compatible = np.all(
            high_freq_sq_residuals * interval_weight < self.precision
        )

        if self.verbose:
            print("Residual: ", high_freq_sq_residuals, " found on interval ", I)

        return is_compatible

    def adaptive_discretization(self, function_family):
        """
        Adaptive disrectization using nested Gaussian Legendre polynomial interpolation.
        Procedure described in "A nonlinear optimization procedure for generalized Gaussian quadratures" p.12-13

        Parameters
        ----------
        :
        Returns
        -------
        :
        """

        ## Stage 1.
        intervals_to_check = [function_family.I]
        intervals = list()

        while intervals_to_check:
            I = intervals_to_check.pop()

            if self.interval_compatible(I, function_family):
                intervals.append(I)
            else:
                midpoint = (I.a + I.b) / 2.0
                intervals_to_check.append(Interval(I.a, midpoint))
                intervals_to_check.append(Interval(midpoint, I.b))

        ## Stage 2.
        self.intervals = sorted(intervals)
        self.endpoints = sorted(
            set([a for (a, b) in intervals] + [function_family.I.b])
        )

        if self.verbose:
            print("Endpoints found: ", self.endpoints)

        ## Stage 3.
        x_global = list()
        w_global = list()
        x, w = legendre.leggauss(2 * self.interpolation_degree)
        for I in self.intervals:
            x_local = I.translate(x)
            w_local = w * I.length() / 2

            x_global.append(x_local)
            w_global.append(w_local)

        x_global = np.concatenate(x_global)
        w_global = np.concatenate(w_global)

        return x_global, w_global

    def naive_discretize2d(self, N, M, Ix, Iy):  ## So far only -1 to 1
        """

        Parameters
        ----------
        :
        Returns
        -------
        :
        """
        x, wx = legendre.leggauss(N)
        wx = wx * (Ix.b - Ix.a) / 2
        tx = sp.interpolate.interp1d([-1.0, 1.0], [*Ix])
        x = tx(x)

        y, wy = legendre.leggauss(M)
        wy = wy * (Iy.b - Iy.a) / 2
        ty = sp.interpolate.interp1d([-1.0, 1.0], [*Iy])
        y = ty(y)

        x_gl = np.kron(x, np.ones_like(y))
        y_gl = np.kron(np.ones_like(x), y)
        w_gl = np.kron(wx, wy)
        return x_gl, y_gl, w_gl, x, y

    def naive_discretize2d_sphere(self, N=10, M=10):
        """
        int_[0,2pi] int_[-1,1] dt dphi

        Parameters
        ----------
        :
        Returns
        -------
        :
        """

        theta, w_theta = legendre.leggauss(N)

        phi = np.arange(M) * 2 * np.pi / M
        w_phi = np.full(shape=M, fill_value=2 * np.pi / M)

        theta_gl = np.kron(theta, np.ones_like(phi))
        phi_gl = np.kron(np.ones_like(theta), phi)
        w_gl = np.kron(w_theta, w_phi)

        return theta_gl, phi_gl, w_gl, theta, phi

    def interpolate_piecewise_legendre(self, U):
        """
        Interpolate point values as family of Piecewise Legendre polynomials.

        Parameters
        ----------
        U : array_like
            2-D array containing the point evaluations U_{ij} = u_j(x_i)

        Returns
        -------
        c : PiecewiseLegendreFamily
        """

        _, number_of_polynomials = U.shape
        piecewise_poly_list = list()
        for n in range(number_of_polynomials):
            p = PiecewiseLegendre.interpolate_gauss_legendre_points(
                U[:, n], self.endpoints
            )
            piecewise_poly_list.append(p)

        return PiecewiseLegendreFamily(piecewise_poly_list, self.endpoints)


def construct_A_matrix(eval_points, weights, functions):
    """

    Parameters
    ----------
    :
    Returns
    -------
    :
    """
    if type(eval_points) is not tuple:
        eval_points = (eval_points,)

    A = np.column_stack([phi(*eval_points) * np.sqrt(weights) for phi in functions])
    return A


def compress_sequence_of_functions(functions, eval_points, weights, precision):
    """
    Construct rank revealing QR s.t. sp.linalg.norm(A[:,perm] - Q[:,:k]@R[:k,:]) <= precision]
    Parameters
    ----------
    :
    Returns
    -------
    :
    """
    A = construct_A_matrix(eval_points, weights, functions)
    Q, R, perm = sp.linalg.qr(A, pivoting=True, mode="economic")
    rank = np.sum(np.abs(np.diag(R)) > precision)

    U = Q[:, :rank] * (np.sqrt(weights)[:, np.newaxis]) ** (-1)
    return U, rank

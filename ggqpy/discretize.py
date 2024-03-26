import numpy as np
import scipy as sp
import numpy.polynomial.legendre as legendre
from tqdm import tqdm
from ggqpy.utils import PiecewiseLegendre, PiecewiseLegendreFamily
from ggqpy.utils import Interval


def pairwise(iterable):
    it = iter(iterable)
    a = next(it, None)

    for b in it:
        yield (a, b)


class Discretizer:
    precision = None
    min_length = None
    verbose = None
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

    def interval_compatible(self, I, phi):
        """

        Parameters
        ----------
        :
        Returns
        -------
        :
        """
        if I.length() < self.min_length:
            return True

        x = (I.b - I.a) * (self.x_gl + 1) / 2 + I.a

        alpha = legendre.legfit(
            self.x_gl, y=phi(x), deg=2 * self.interpolation_degree - 1
        )  # Fit to Legendre Polynomials on [a,b]
        high_freq_sq_residuals = np.sum(abs(alpha[self.interpolation_degree :]) ** 2)

        if self.verbose:
            print("Residual: ", high_freq_sq_residuals, " found on interval ", I)

        return high_freq_sq_residuals < self.precision

    def add_endpoints(self, intervals, phi):
        """

        Parameters
        ----------
        :
        Returns
        -------
        :
        """
        intervals_to_check = intervals.copy()
        intervals_out = list()

        while intervals_to_check:
            I = intervals_to_check.pop()

            if self.interval_compatible(I, phi):
                intervals_out.append(I)
            else:
                midpoint = (I.a + I.b) / 2.0
                intervals_to_check.append(Interval(I.a, midpoint))
                intervals_to_check.append(Interval(midpoint, I.b))

        return intervals_out

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
        I = function_family.I
        intervals = [I]

        ## Stage 1.
        for phi in tqdm(function_family.functions_lambdas):
            intervals = self.add_endpoints(intervals, phi)

        ## Stage 2.
        self.intervals = sorted(intervals)
        self.endpoints = sorted(set([a for (a, b) in intervals] + [I.b]))

        if self.verbose:
            print("Endpoints found: ", self.endpoints)

        ## Stage 3.
        x_global = list()
        w_global = list()
        for I in self.intervals:
            x, w = legendre.leggauss(2 * self.interpolation_degree)
            w = w * I.length() / 2
            x_global.append(I.translate(x))
            w_global.append(w)

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
        int_[0,2\pi] int_[-1,1] dt d\phi

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

        points_total, number_of_polynomials = U.shape
        number_of_intervals = len(self.endpoints) - 1
        points_per_interval = points_total // number_of_intervals
        u_list = list()
        x, _ = legendre.leggauss(points_per_interval)
        for n in range(number_of_polynomials):
            piecewise_poly = list()
            for i in range(number_of_intervals):
                u_local = U[points_per_interval * i : points_per_interval * (i + 1), n]
                coef = np.polynomial.legendre.legfit(
                    x, u_local, deg=points_per_interval - 1
                )
                p_local = legendre.Legendre(
                    coef, (self.endpoints[i], self.endpoints[i + 1])
                )
                piecewise_poly.append(p_local)

            u_list.append(PiecewiseLegendre(piecewise_poly, self.endpoints))

        return PiecewiseLegendreFamily(u_list, self.endpoints)


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

    Parameters
    ----------
    :
    Returns
    -------
    :
    """
    ## Construct rank revealing QR s.t. sp.linalg.norm(A[:,perm] - Q[:,:k]@R[:k,:]) <= precision]
    A = construct_A_matrix(eval_points, weights, functions)
    Q, R, _ = sp.linalg.qr(A, pivoting=True, mode="economic")
    rank = np.sum(np.abs(np.diag(R)) > precision)

    U = Q[:, :rank] * (np.sqrt(weights)[:, np.newaxis]) ** (-1)
    return U, rank

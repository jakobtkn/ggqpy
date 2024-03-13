import numpy as np
import scipy as sp
import numpy.polynomial.legendre as legendre
from tqdm import tqdm
from ggqpy.functionfamiliy import Interval


def pairwise(iterable):
    it = iter(iterable)
    a = next(it, None)

    for b in it:
        yield (a, b)
        a = b


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
        return

    def interval_compatible(self, I, phi):
        translate = sp.interpolate.interp1d([-1.0, 1.0], [I.a, I.b])
        x, _ = legendre.leggauss(2 * self.interpolation_degree)
        alpha = legendre.legfit(
            x, y=phi(translate(x)), deg=2 * self.interpolation_degree - 1
        )  # Fit to Legendre Polynomials on [a,b]
        high_freq_sq_residuals = np.sum(abs(alpha[self.interpolation_degree :]) ** 2)

        if self.verbose:
            print("Residual: ", high_freq_sq_residuals, " found on interval ", I)

        return high_freq_sq_residuals < self.precision

    def add_endpoints(self, sub_interval, phi):
        if (
            self.interval_compatible(sub_interval, phi)
            or sub_interval.length() < self.min_length
        ):
            return
        else:
            midpoint = (sub_interval.a + sub_interval.b) / 2.0
            self.endpoints.append(midpoint)
            self.add_endpoints(Interval(sub_interval.a, midpoint), phi)
            self.add_endpoints(Interval(midpoint, sub_interval.b), phi)
            return

    def adaptive_discretization(self, function_family):
        """
        Adaptive disrectization using nested Gaussian Legendre polynomial interpolation.
        Procedure described in "A nonlinear optimization procedure for generalized Gaussian quadratures" p.12-13
        """
        I = function_family.I
        self.endpoints = [I.a, I.b]

        ## Stage 1.
        for phi in tqdm(function_family.functions):
            self.add_endpoints(I, phi)

        ## Stage 2.
        self.endpoints = sorted(set(self.endpoints))

        if self.verbose:
            print("Endpoints found: ", self.endpoints)

        ## Stage 3.
        x_global = list()
        w_global = list()
        intervals = list()
        for start, end in pairwise(self.endpoints):
            intervals.append(Interval(start, end))
            x, w = legendre.leggauss(2 * self.interpolation_degree)
            w = w * (end - start) / 2
            translate = sp.interpolate.interp1d([-1.0, 1.0], [start, end])
            x_global.append(translate(x))
            w_global.append(w)

        x_global = np.concatenate(x_global)
        w_global = np.concatenate(w_global)

        return x_global, w_global, self.endpoints, intervals

    def naive_discretize2d(self, N, M, Ix, Iy):  ## So far only -1 to 1
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
        ''' int_[0,2\pi] int_[-1,1] dt d\phi'''
        theta, w_theta = legendre.leggauss(N)

        phi = np.arange(M)*2*np.pi/M
        w_phi = np.full(shape=M, fill_value=2 * np.pi / M)

        theta_gl = np.kron(theta, np.ones_like(phi))
        phi_gl = np.kron(np.ones_like(theta), phi)
        w_gl = np.kron(w_theta, w_phi)
        
        return theta_gl, phi_gl, w_gl, theta, phi

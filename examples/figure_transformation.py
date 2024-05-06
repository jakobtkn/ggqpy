import numpy as np
import sys, os

sys.path.append(os.path.abspath("."))
from ggqpy import *
from ggqpy.parametrization import *

if __name__ == "__main__":
    param = Parametrization.droplet()
    rho, drho, jacobian, normal = param.get_lambdas()
    x0 = np.array([2, 1])
    B, Binv = ensure_conformal_mapping(drho, x0)

    simplex = Rectangle(Interval(0,2*np.pi), Interval(0, np.pi))
    T = Quadrilateral(*[Binv @ (np.array(v) - x0) for v in iter(simplex)])
    print("Simplex:")
    for x in [*simplex]:
        print(*x)
    
    print("x0:")
    print(x0)

    print("T:")
    for x in [*T]:
        print(*x)

    print("rho(x0):")
    print(rho(*x0))



    for (idx, Ti) in enumerate([*T.split_into_triangles_around_point((0,0))]):
        scale, angle, A, Ainv, detA = standard_radial_triangle_transform(
            Ti.vertices[1], Ti.vertices[2]
        )
        print(f"T{idx + 1}:")
        for x in [*Ti]:
            print(*x)

        print(f"T{idx + 1} (scaled):")
        for x in [(0,0), (1,0), (scale*np.cos(angle), scale*np.sin(angle))]:
            print(*x)
import sys, os
sys.path.append(os.path.abspath("."))
from ggqpy import *
import matplotlib.pyplot as plt
interpolation_degree =3
disc = Discretizer(1e-2,1e-10,interpolation_degree,True)

gamma = (
    lambda r0, theta0, u: r0
    * np.sin(theta0)
    / (r0 * np.sin(theta0 - theta0 * u) + np.sin(theta0 * u))
)

i = 4
j = 6
alpha = 0.9
beta = 3*np.pi/4
f =  lambda u: beta * gamma(alpha, beta, u) ** (i + 2) / (i + 2) * np.cos(j * beta * u)

F = FunctionFamily(Interval(0,1), [f])
x,w = disc.adaptive_discretization(F)

print(list(x))
U_disc = f(x)[:,None]
U_family = disc.interpolate_piecewise_legendre(U_disc)


for k in range(len(disc.endpoints) - 1):
    a = disc.endpoints[k]
    b = disc.endpoints[k + 1]
    xx = np.linspace(a,b,20)
    print(f"\\def\\x{chr(k+97)}" + "{", *list(np.around(xx,3)), "}", sep=",")
    plt.plot(xx, U_family.piecewise_poly_list[0](xx))
    print(f"\\def\\u{chr(k+97)}" + "{", *list(np.around(U_family.piecewise_poly_list[0](xx),3)), "}", sep=",")
plt.show()


import sys, os

sys.path.append(os.path.abspath("."))
import matplotlib.pyplot as plt

from ggqpy import *
from ggqpy.nystrom import *
from ggqpy.duffy import *


quad_loader = QuadratureLoader(4)
r0 = 0.9
theta0 = np.pi / 2
r, theta, w = quad_loader.quad_on_standard_triangle(r0, theta0)
x = r * np.cos(theta)
y = r * np.sin(theta)
plt.figure()
plt.scatter(x, y)
plt.title(len(x))
plt.show()
# print("x", *np.around(x, 3), sep=",")
# print("y", *np.around(y, 3), sep=",")
for (x,y) in zip(np.around(x,3),np.around(y,3)):
    print(x, " ", y)
print(r)


n = int(np.floor(np.sqrt(len(r))))
print(n)
uu, vv, ww = make_2d_qaud_unit_square(n)
x, y, ww = duffy_on_standard_triangle(r0, theta0, uu, vv, ww)
plt.figure()
plt.scatter(x, y)
plt.title(len(x))
plt.show()
for (x,y) in zip(np.around(x,3),np.around(y,3)):
    print(x, " ", y)
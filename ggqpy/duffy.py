from ggqpy.geometry import Quadrilateral, standard_radial_triangle_transform
from ggqpy.nystrom import *

def make_2d_qaud_unit_square(n):
    x_gl, w_gl = legendre.leggauss(n)
    x_gl = (x_gl+1.0)/2.0
    w_gl = w_gl/2

    x, y = np.meshgrid(x_gl, x_gl)
    xx = x.flatten()
    yy = y.flatten()
    wx, wy = np.meshgrid(w_gl, w_gl)
    ww = (wx * wy).flatten()
    return xx, yy, ww

def duffy_on_standard_triangle(scale, angle, uu, vv, ww):
    x = uu
    y = vv*uu
    X = np.array([[1,scale*np.cos(angle)-1],[0,scale*np.sin(angle)]]) @ np.row_stack([x,y])
    x = X[0,:]
    y = X[1,:]
    w = ww*uu*scale*np.sin(angle)
    return x,y,w

def duffy_quad(drho, x0, simplex, n):
    
    B, Binv = ensure_conformal_mapping(drho, x0)
    R = Quadrilateral(*[Binv @ (np.array(v) - x0) for v in iter(simplex)])
    x_list = list()
    y_list = list()
    w_list = list()

    uu,vv,ww = make_2d_qaud_unit_square(n)

    detB = abs(np.linalg.det(B))
    for T in [*R.split_into_triangles_around_point((0, 0))]:
        scale, angle, A, Ainv, detA = standard_radial_triangle_transform(
            T.vertices[1], T.vertices[2]
        )
        x_local, y_local, w_local = duffy_on_standard_triangle(scale, angle, uu, vv, ww)

        v = np.row_stack([x_local, y_local])
        v = B @ (Ainv @ v) + x0[:, np.newaxis]

        x_list.append(v[0, :])
        y_list.append(v[1, :])
        w_list.append((w_local / detA) * detB)

    x = np.concatenate(x_list)
    y = np.concatenate(y_list)
    w = np.concatenate(w_list)

    return x, y, w


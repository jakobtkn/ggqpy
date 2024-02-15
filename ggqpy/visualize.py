import matplotlib.pyplot as plt
import numpy as np

def plot_points(theta,phi):
    # Create a sphere
    r = 1
    pi = np.pi
    cos = np.cos
    sin = np.sin
    _phi, _theta = np.mgrid[0.0:pi:100j, 0.0:2.0*pi:100j]
    x = r*sin(_phi)*cos(_theta)
    y = r*sin(_phi)*sin(_theta)
    z = r*cos(_phi)

    #Import data
    # data = np.genfromtxt('leb.txt')
    # theta, phi, r = np.hsplit(data, 3) 
    xx = sin(phi)*cos(theta)
    yy = sin(phi)*sin(theta)
    zz = cos(phi)

    #Set colours and render
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(
        x, y, z,  rstride=1, cstride=1, color='c', alpha=0.3, linewidth=0)

    ax.scatter(xx,yy,zz,color="k",s=20)

    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()
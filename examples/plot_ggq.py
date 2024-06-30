import numpy as np
import matplotlib.pyplot as plt

def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)


x_ggq = np.loadtxt("data/ggq.x")
w_ggq = np.loadtxt("data/ggq.w")
x_gl = np.loadtxt("data/gl.x")
w_gl = np.loadtxt("data/gl.w")

colors = [
        "#377eb8",
        "#ff7f00",
        "#4daf4a",
        "#f781bf",
        "#a65628",
        "#984ea3",
        "#999999",
        "#e41a1c",
        "#dede00",
    ]
fig = plt.figure()
plt.stem(x_ggq,w_ggq, colors[2], markerfmt = "^", basefmt=colors[1], label = "GGQ")
plt.stem(x_gl,w_gl, colors[0], markerfmt = "s", basefmt=colors[1], label="Gauss-Legendre")
plt.legend(["GGQ", "Gauss-Legendre"])
fig = tikzplotlib_fix_ncols(fig)
plt.xlim(0,1)
plt.xlabel("Nodes")
plt.ylabel("Weights")

try:
    import tikzplotlib
    tikzplotlib.save(f"output/ggq.tex", axis_width="0.8\\textwidth", axis_height="0.5\\textwidth", textsize = 12)

except ImportError as e:
    print("No tikzplotlib")
    pass
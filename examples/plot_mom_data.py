from tester import Tester, load_tester
import itertools
import pickle
import matplotlib.pyplot as plt

import sys, os

sys.path.append(os.path.abspath("."))


def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)


tester_names = [
    "naive",
    "duffy",
    "ggq_precomputed_4",
    "ggq_precomputed_8",
    "ggq_precomputed_16",
    "ggq_exact_triangle",
    "ticra",
]
display_names = [
    "Gauss-Legendre",
    "Duffy",
    "GGQ, $n=4$",
    "GGQ, $n=8$",
    "GGQ, $n=16$",
    "GGQ, $n=4$, Exact triangle",
    "TICRA's method",
]

def plot_data(folder):
    testers = list[Tester]()

    for name in tester_names:
        tester = load_tester(folder, name)
        testers.append(tester)
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
    markers = ["o", "s", "+", "x", "*", ".", "^", "v", "s"]

    from cycler import cycler
    custom_cycler = cycler(color=colors) + cycler(marker=markers)
    plt.rc('axes', prop_cycle=custom_cycler)
    # plt.style.use("tableau-colorblind10")

    fig, ax = plt.subplots()
    for tester, display_name in zip(testers, display_names):
        plt.loglog(tester.num_nodes, tester.rel_error, label="\\footnotesize " + display_name)

    plt.legend(loc=3)
    plt.xlabel("Number of quadrature nodes")
    plt.xlim(1e3, 1e6)
    plt.ylabel("Relative error")
    plt.grid(True)
    # plt.savefig(f"output/mom.{folder}.pdf")

    fig = tikzplotlib_fix_ncols(fig)
    import tikzplotlib

    tikzplotlib.save(f"output/mom.{folder}.tex", axis_height="9cm", axis_width="\\textwidth", textsize = 2)
    plt.show()


if __name__ == "__main__":
    plot_data("simple_patch")
    plot_data("high_order")

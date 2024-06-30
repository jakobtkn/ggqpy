from tester import Tester, load_tester
import itertools
import pickle
import matplotlib.pyplot as plt
import tikzplotlib
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

    # colors = ["#e41a1c"  "#dede00"  ]


class Data:
    def __init__(self, tester_name, display_name, color, marker, markersize=7):
        self.tester_name = tester_name
        self.display_name = display_name
        self.color = color
        self.marker = marker
        self.markersize = markersize

    def plot(self, folder):
        tester = load_tester(folder, self.tester_name)
        plt.loglog(
            tester.num_nodes,
            tester.rel_error,
            label=self.display_name,
            c=self.color,
            marker=self.marker,
            markersize=8,
        )


data_naive = Data("naive", "Gauss-Legendre", "#377eb8", "s", 7)
data_duffy = Data("duffy", "Duffy", "#ff7f00", "o")
data_4 = Data("ggq_precomputed_4", "GGQ, $n=4$", "#4daf4a", "^")
data_8 = Data("ggq_precomputed_8", "GGQ, $n=8$", "#f781bf", "<")
data_16 = Data("ggq_precomputed_16", "GGQ, $n=16$", "#a65628", ">")
data_4_exact = Data(
    "ggq_exact_triangle", "GGQ, $n=4$, Exact triangle", "#984ea3", "D", 7
)
data_ticra = Data("ticra", "TICRA's method", "#999999", "v", 10)


def plot_data(testers, input_folder, output_name):
    fig, ax = plt.subplots()
    for data in testers:
        data.plot(input_folder)

    plt.legend(loc=3)
    plt.xlabel("Number of quadrature nodes")
    plt.xlim(1e3, 1e6)
    plt.ylabel("Relative error")
    plt.grid(True)
    fig = tikzplotlib_fix_ncols(fig)

    try:
        import tikzplotlib

        tikzplotlib.save(
            f"output/mom.{output_name}.tex",
            axis_height="9cm",
            axis_width="\\textwidth",
            textsize=2,
        )

    except ImportError as e:
        print("No tikzplotlib")
        pass


if __name__ == "__main__":
    plot_data(
        [data_naive, data_duffy, data_4, data_8, data_ticra],
        "simple_patch",
        "simple_patch",
    )
    plot_data([data_4, data_4_exact, data_ticra], "simple_patch", "exact_triangle")
    plot_data(
        [data_duffy, data_4, data_8, data_16, data_ticra], "high_order", "high_order"
    )

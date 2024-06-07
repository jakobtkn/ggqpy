from tester import Tester, load_tester
import itertools
import pickle
import matplotlib.pyplot as plt

import sys,os
sys.path.append(os.path.abspath("."))



tester_names = ["naive","duffy","ggq_precomputed_4","ggq_precomputed_8","ggq_precomputed_16","ggq_exact_triangle","ticra"]
display_names = ["Gauss-Legendre","Duffy","GGQ, $n=4$","GGQ, $n=8$","GGQ, $n=16$","GGQ, $n=4$, Exact triangle","TICRA's method"]

def plot_data(folder):
    testers = list[Tester]()
    
    for name in tester_names:
        tester = load_tester(folder, name)
        testers.append(tester)

    marker = itertools.cycle(('s', '+', '.', 'o', '*','^','v')) 
    plt.figure()
    for tester, display_name in zip(testers,display_names):
        plt.loglog(tester.num_nodes, tester.rel_error, "-*", label=display_name, marker = next(marker))

    plt.legend()
    plt.xlabel("Number of quadrature nodes")
    plt.xlim(1e3,1e6)
    plt.ylabel("Relative error")
    plt.grid(True)
    # plt.savefig(f"output/mom.{folder}.pdf")

    import tikzplotlib
    tikzplotlib.save(f"output/{folder}.tex")


if __name__ == "__main__":
    plot_data("simple_patch")
    plot_data("high_order")


    
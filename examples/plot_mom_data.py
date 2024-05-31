from tester import Tester, load_tester
import pickle
import matplotlib.pyplot as plt
import sys,os
sys.path.append(os.path.abspath("."))



tester_names = ["naive","duffy","ggq_precomputed_4","ggq_precomputed_8","ggq_precomputed_16","ggq_exact_triangle","ticra"]
def plot_data(folder):
    testers = list[Tester]()
    
    for name in tester_names:
        tester = load_tester(folder, name)
        testers.append(tester)

    plt.figure()
    for tester in testers:
        plt.loglog(tester.num_nodes, tester.rel_error, "-*", label=tester.name)

    plt.legend()
    plt.xlabel("Number of quadrature nodes")
    plt.xlim(1e3,1e6)
    plt.ylabel("Relative error")
    plt.savefig(f"output/mom.{folder}.pdf")


if __name__ == "__main__":
    plot_data("simple_patch")
    plot_data("high_order")


    
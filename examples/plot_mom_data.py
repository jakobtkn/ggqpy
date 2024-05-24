from tester import Tester, load_tester
import pickle
import matplotlib.pyplot as plt
import sys,os
sys.path.append(os.path.abspath("."))



tester_names = ["naive","duffy","ggq_exact_triangle","ggq_precomputed","ticra"]
def plot_data(folder):
    testers = list[Tester]()
    
    for name in tester_names:
        tester = load_tester(folder, name)
        testers.append(tester)

    for tester in testers:
        plt.semilogy(tester.num_nodes, tester.rel_error, "-*", label=tester.name)

    plt.xlim((1000,50000))
    plt.legend()
    plt.savefig(f"output/mom.{folder}.pdf")
    plt.xlabel("Number of quadrature nodes")
    plt.ylabel("Relative error")
    plt.show()


if __name__ == "__main__":
    plot_data("simple_patch")


    
import sys
import os

from ggqpy.quad import Quadrature

sys.path.append(os.path.abspath("."))
from ggqpy import *


def main():
    quad = Quadrature.load_from_file("quads/out.quad")
    print(np.sum(quad.w))
    return


if __name__ == "__main__":
    main()

#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import scipy as sp

from ggqpy.quad import Quadrature

sys.path.append(os.path.abspath("."))

import numpy as np
from ggqpy import *


def generate_quad(count, order, filename):
    min_length = 1e-10
    eps_disc = 1e-14
    eps_comp = 1e-12
    eps_quad = 1e-12

    F = FunctionFamily.nystrom_integral_functions(count, order)
    x, w = generalized_gaussian_quadrature(F, min_length, eps_disc, eps_comp, eps_quad)
    quad = Quadrature(x, w)
    quad.save_to_file(filename)

 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("order", default=4)
    parser.add_argument("r_interval")
    parser.add_argument("theta_interval")

    intervals = np.load("quad/nystrom/config.npz")

    parser.add_argument("filename", default="quads/out.quad")
    args = parser.parse_args()
    generate_quad(int(args.count), int(args.order), args.filename)

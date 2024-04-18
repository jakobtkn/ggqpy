#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import scipy as sp

sys.path.append(os.path.abspath("."))

import numpy as np
from ggqpy import *
from ggqpy.quad import Quadrature

def generate_quad(order, r0_index, theta0_index, filename):
    count = 16
    min_length = 1e-10
    eps_disc = 1e-13
    eps_comp = 1e-12
    eps_quad = 1e-12
    
    r = np.loadtxt(f"quads/nystrom.{order}/breakpoints_r")
    theta = np.loadtxt(f"quads/nystrom.{order}/breakpoints_theta")
    amin = r[r0_index]
    amax = r[r0_index + 1]
    bmin = theta[theta0_index]
    bmax = theta[theta0_index + 1]
    
    F = FunctionFamily.nystrom_integral_functions(count, order, amin, amax, bmin, bmax)
    x, w = generalized_gaussian_quadrature(F, min_length, eps_disc, eps_comp, eps_quad)
    quad = Quadrature(x, w)
    quad.save_to_file(filename)

 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("order", default=4)
    parser.add_argument("r0_index")
    parser.add_argument("theta0_index")
    parser.add_argument("filename", default="quads/out.quad")
    args = parser.parse_args()
    generate_quad(int(args.order), int(args.r0_index), int(args.theta0_index), args.filename)

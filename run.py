#!/usr/bin/python3


from cmath  import sqrt
from collections import OrderedDict

from mpi4py import MPI

import matplotlib.pyplot as plt
import plotting 
import time_evolve
import measures

# Execute simulations
# ===================

# lists of parameters to simulate
# -------------------------------

eq = 1.0/sqrt(2.0)

output_dir = 'testing/run'

mode_list = ['block', 'sweep']

IC_list = [ 'l0' ]

R_list = [147]

V_list = [['H']]

L_list = [11]

T_list = [60]

params_list = [ 
        { 
            'output_dir' : output_dir,
            'mode' : mode,
            'V' : V,
            'R' : R, 
            'IC': IC, 
            'L' : L, 
            'T' : T
             }
        for mode in mode_list  \
        for V    in V_list \
        for R    in R_list     \
        for IC   in IC_list   \
        for L    in L_list     \
        for T    in T_list  ]

# run independent simulations in parallel
# ---------------------------------------
if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    for i, params in enumerate(params_list):
        if i % nprocs == rank: 
            time_evolve.run_sim(params, force_rewrite=False)
            measures.measure(params, force_rewrite=False)
            plotting.plot(params)
            plt.clf
    plt.close('all')


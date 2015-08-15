#!/usr/bin/python


from cmath  import sqrt

from mpi4py import MPI

import plotting  as pt
import sweep_eca as sweep


# Execute simulations
# ===================

# lists of parameters to simulate
# -------------------------------


output_name = 'comp'

R_list = [91]

eq = 1.0/sqrt(2.0)
IC_list = [ [('c2l0', eq), ('c2l1', eq)], [('c2E0_1', 1.0)], [('c2s1', 1.0)] ]

L_list = [10]

tmax_list = [35]

params_list = [ (output_name, R, IC, L, tmax) \
        for R   in R_list     \
        for IC   in IC_list   \
        for L    in L_list    \
        for tmax in tmax_list ]



# run independent simulations in parallel
# ---------------------------------------

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    for i, params in enumerate(params_list):
        if i % nprocs == rank: 
            sweep.run_sim(params, force_rewrite=True)
            pt.plot_main(params)

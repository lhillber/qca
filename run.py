#!/usr/bin/python


from cmath  import sqrt

from mpi4py import MPI

import plotting  as pp
import sweep_eca as sweep


# Execute simulations
# ===================

# lists of parameters to simulate
# -------------------------------
output_name = 'sweep/testing'

R_list = [90,91,102,60]

IC_list = [ [ ('qT1_p0', 1.0) ], [ ('qT2_p0', 1.0) ]  ]

L_list = [15]

tmax_list = [21]




params_list = [ (output_name, R, IC, L, tmax) \
        for R   in R_list     \
        for IC   in IC_list   \
        for L    in L_list    \
        for tmax in tmax_list ]

# run independent simulations in parallel
# ---------------------------------------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()
for i, params in enumerate(params_list):
    if i % nprocs == rank: 
        sweep.run_sim(params, force_rewrite=True)
        pp.plot_time_series(params)

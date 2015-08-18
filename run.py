#!/usr/bin/python


from cmath  import sqrt

from mpi4py import MPI
import matplotlib.pyplot as plt
import plotting      as pt
import sweep_eca     as sweep
import classical_eca as eca

# Execute simulations
# ===================

# lists of parameters to simulate
# -------------------------------


output_name = 'comp3'

eq = 1.0/sqrt(2.0)




L = 10


QIC_list = [[('E0_'+str(j), 1.0)] for j in range(1, int(L))]


CIC_list = [ [('z', 0.5), ('i0_'+str(j), 0.5)] for j in range(1, int(L)) ]


R_list = [60]

L_list = [L]

tmax_list = [200]

Qparams_list = [ (output_name, R, IC, L, tmax) \
        for R    in R_list     \
        for IC   in QIC_list   \
        for L    in L_list     \
        for tmax in tmax_list  ]

Cparams_list = [ (output_name, R, IC, L, tmax) \
        for R    in R_list     \
        for IC   in CIC_list   \
        for L    in L_list     \
        for tmax in tmax_list  ]


params_lists = [Qparams_list, Cparams_list]

# run independent simulations in parallel
# ---------------------------------------
if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    for i, params in enumerate(Qparams_list):
        if i % nprocs == rank: 
            sweep.run_sim(params, force_rewrite=True)
            pt.plot_main(params)
            plt.clf
    plt.close('all')
    for i, params in enumerate(Cparams_list):
        if i % nprocs == rank: 
            eca.run_mixture(params, force_rewrite=True) 
            eca.plot_main(params)
            plt.clf
    plt.close('all')

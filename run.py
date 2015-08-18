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


output_name = 'comp2'

eq = 1.0/sqrt(2.0)

QIC_list = [ [('c2l0', eq), ('c2l1', eq)], [('c2E0_1', 1.0)], [('c2s1', 1.0)] ]

CIC_list = [[('z', 0.5), ('c2i0_1', 0.5)], [('c2i0', 0.5), ('c2i1', 0.5)], [('z', 0.5), ('c2i1', 0.5)] ]



QIC_list = [ [('qt90_p0', 1.0)], [('qt90_p90', 1.0)], [('qt90_P1', 1.0)] ]

CIC_list = [ [('o', 0.5), ('z', 0.5)]]




CIC_list = [[('a',1)]]

R_list = [91]

L_list = [10]

tmax_list = [35]

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

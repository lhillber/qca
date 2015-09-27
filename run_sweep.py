#!/usr/bin/python


from cmath  import sqrt
from collections import OrderedDict

from mpi4py import MPI

import matplotlib.pyplot as plt
import plotting          as pt
import sweep_eca         as sweep
import ham_eca           as ham
import classical_eca     as eca

# Execute simulations
# ===================

# lists of parameters to simulate
# -------------------------------



eq = 1.0/sqrt(2.0)

output_name = 'R_leak_study/0p0'

eq = 1.0/sqrt(2.0)

L =12

QIC_list = [[('E0_'+str(j), 1.0)] for j in range(1, int(L))]

CIC_list = [ [('z', 0.5), ('i0_'+str(j), 0.5)] for j in range(1, int(L)) ]

QIC_list = [[('E'+str(L-2)+'_' + str(L-1), 1.0)]]


R_list = [60]

L_list = [L]

tmax_list = [20]

Qparams_list = [ 
        OrderedDict( [ 
            ('output_name', output_name), 
            ('R', R), 
            ('IC', IC), 
            ('L', L), 
            ('tmax', tmax) 
            ] )

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
    
    '''
    for i, params in enumerate(Cparams_list):
        if i % nprocs == rank: 
            eca.run_mixture(params, force_rewrite=False) 
            eca.plot_main(params)
            plt.clf
    plt.close('all')
    '''

#!/usr/bin/python3


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

output_name = 'block_eca/102_150'

QIC_list = [ 's12' ]

R_list = [102, 150]

center_op_list = [['X'], ['H','X'], ['H','X','T']]


L_list = [13]

tmax_list = [70]

Qparams_list = [ 
        OrderedDict( [ 
            ('output_name', output_name), 
            ('center_op', center_op),
            ('R', R), 
            ('IC', IC), 
            ('L', L), 
            ('tmax', tmax) 
            ] )

        for center_op in center_op_list \
        for R    in R_list     \
        for IC   in QIC_list   \
        for L    in L_list     \
        for tmax in tmax_list  ]


# run independent simulations in parallel
# ---------------------------------------
if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    for i, params in enumerate(Qparams_list):
        if i % nprocs == rank: 
            sweep.run_sim(params, force_rewrite=True)
            #pt.plot_main(params)
            #plt.clf
    #plt.close('all')
    
    '''
Cparams_list = [ (output_name, R, IC, L, tmax) \
        for R    in R_list     \
        for IC   in CIC_list   \
        for L    in L_list     \
        for tmax in tmax_list  ]


params_lists = [Qparams_list, Cparams_list]
    for i, params in enumerate(Cparams_list):
        if i % nprocs == rank: 
            eca.run_mixture(params, force_rewrite=False) 
            eca.plot_main(params)
            plt.clf
    plt.close('all')
    '''

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

output_name = 'alt_ops/'

QIC_list = ['c1d1']

R_list = [51, 54, 60, 150, 156]

center_op_list = [['X'], ['H'], ['X','H']]

L_list = [14]

tmax_list = [30]

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

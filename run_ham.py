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

eq = 1.0/sqrt(2.0)    # a useful constant

output_name = 'ham_eca/R105_gsweep_J0/'

g_list = [0.5, 0.7, 1.0, 1.3, 2.0]

R_list = [105]

L_list = [12]

QIC_list = [[('c1d1', 1.0)]]

tmax_list = [1000]

dt_list = [0.1]


# nested list of the above parameters as an ordered dictionary
# ------------------------------------------------------------
params_list = [ 
    OrderedDict( 
                [ 
            ( 'output_name',  output_name ), 
            (           'R',  R           ), 
            (           'g',  g           ), 
            (           'L',  L           ), 
            (          'dt',  dt          ), 
            (        'tmax',  tmax        ),
            (          'IC',  IC          ) 
                 ] 
                  )
                   for R    in R_list     \
                   for g    in g_list     \
                   for L    in L_list     \
                   for dt   in dt_list    \
                   for tmax in tmax_list  \
                   for IC   in QIC_list   \
                                          ]


# run independent simulations in parallel
# ---------------------------------------
if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    for i, params in enumerate(params_list):
        if i % nprocs == rank: 
            ham.run_sim(params, force_rewrite=True)
            pt.plot_main(params)
            plt.clf
    plt.close('all')
    

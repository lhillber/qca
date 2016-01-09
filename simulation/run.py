#!/usr/bin/python3


from cmath  import sqrt, sin, cos, pi
from collections import OrderedDict
from os.path import isfile
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
import plotting
import time_evolve
import measures
import fio as io
import time

# Execute simulations
# ===================

# lists of parameters to simulate
# -------------------------------
output_dir = 'tmp'

mode_list = ['sweep']

#IC_list = [[ ('W', sin(th)), ('c1l0', cos(th))]
#        for th in np.linspace(0, pi/2.0, 10)]
IC_list = ['c1l0']

S_list = [6]

V_list = ['H']

L_list = [18]

T_list = [60]

params_list = [
           {
            'output_dir' : output_dir,
            'mode' : mode,
            'V' : V,
            'S' : S,
            'IC': IC,
            'L' : L,
            'T' : T
             }

        for mode in mode_list  \
        for V    in V_list     \
        for S    in S_list     \
        for IC   in IC_list    \
        for L    in L_list     \
        for T    in T_list     ]



# run independent simulations in parallel
# ---------------------------------------
if __name__ == '__main__':
    # initialize communication
    comm = MPI.COMM_WORLD
    # get the rank of the processor
    rank = comm.Get_rank()
    # get the number of processsors
    nprocs = comm.Get_size()
    # use rank 0 to give each simulation a file name
    if rank == 0:
        for params in params_list:
            # iterate version numbers for random throw IC's
            if params['IC'][0] == 'r':
                fname = io.make_file_name(params, iterate = True)
            # don't iterate file names with a unique IC name
            else:
                fname = io.make_file_name(params, iterate = False)
            # set the file name for each simulation
            params['fname'] = fname

    # boradcast updated params list to each core
    params_list = comm.bcast(params_list, root=0)
    for i, params in enumerate(params_list):

        # each core selects params to simulate without the need for a master
        if i % nprocs == rank:
            t0 = time.time()
            fname = time_evolve.run_sim(params, force_rewrite=True)
            t1 = time.time()
            print(t1-t0 )

            measures.measure(params, fname, force_rewrite=True)
            plotting.plot(params, fname, force_rewrite=True)
            plt.clf
    plt.close('all')

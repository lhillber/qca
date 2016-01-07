#!/usr/bin/python3


from cmath  import sqrt
from collections import OrderedDict
from os.path import isfile
from mpi4py import MPI
import matplotlib.pyplot as plt
import plotting
import time_evolve
import measures
import fio as io

# Execute simulations
# ===================

# lists of parameters to simulate
# -------------------------------
output_dir = 'testing'

mode_list = ['block']

IC_list = ['G']

S_list = [6]

V_list = ['HX','HXT']

L_list = [15]

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
            fname = time_evolve.run_sim(params, force_rewrite=False)
            measures.measure(params, fname, force_rewrite=True)
            plotting.plot(params, fname, force_rewrite=False)
            plt.clf
    plt.close('all')


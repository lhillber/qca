#!/usr/bin/python3

# =============================================================================
# launches simulations across several cores
# =============================================================================

from mpi4py import MPI
from os.path import isfile
import simulation.plotting as plotting
import simulation.time_evolve as time_evolve
import simulation.measures as measures
import simulation.fio as io
import time



# prepare file for writing in single simulation
def init_files_single(params):
    if 'fname' in params:
        fname = params['fname']
    else:
        if 'IC' in params and params['IC'][0] == 'r':
                fname = io.make_file_name(params, iterate = False)
                #fname = io.make_file_name(params, iterate = True)
            # don't iterate file names with a unique IC name
        else:
            fname = io.make_file_name(params, iterate = False)
            # set the file name for each simulation
        params['fname'] = fname


# launch a single simulation
def launch_single(  params, comm=None,
                    sim_tasks = ['one_site', 'two_site', 'IPR'],
                    rewrite_states = False,
                    measure_tasks=['s', 'sc', 'mom', 'g', 'm', 'nm', 'stats'],
                    coord_tasks=['xx', 'yy', 'zz'], 
                    nm_tasks=['ND', 'CC', 'Y'], 
                    corrj='L/2',
                    rewrite_measures = False ):
    if comm is None:
        rank = 0
        init_files_single(params)
    else:
       rank = comm.Get_rank()

    t0 = time.time()
    state_res = time_evolve.run_sim(params, 
            sim_tasks=sim_tasks, force_rewrite=rewrite_states)
    t1 = time.time()
    res_size = measures.measure(params, state_res,
            measure_tasks=measure_tasks,
            coord_tasks=coord_tasks,
            nm_tasks=nm_tasks,
            corrj=corrj,
            force_rewrite=rewrite_measures)
    t2 = time.time()
    out_fname = plotting.plot(params)
    t3 = time.time()

    print_string = \
    '='*80                 + '\n'\
    + 'Rank: ' + str(rank) + '\n'\
    + 'Data file:'         + '\n'\
    + params['fname']      + '\n'\
    + 'Plots file:'        + '\n'\
    + out_fname            + '\n'\
    + 'simulating states took {:.2f} s'.format(t1-t0) + '\n'\
    + 'measuring states took  {:.2f} s'.format(t2-t1) + '\n'\
    + 'plotting measures took {:.2f} s'.format(t3-t2) + '\n'\
    + 'data file is {:.2f} MB'.format(res_size/1e6)   + '\n'\
    + '='*80
    return print_string



# initialize communication for parallel simulations
def init_comm():
    comm = MPI.COMM_WORLD
    # get the rank of the processor
    rank = comm.Get_rank()
    # get the number of processsors
    nprocs = comm.Get_size()
    return comm, rank, nprocs


# prepare files for writing in parallel
def init_files_parallel(comm, rank, params_list):
    # use rank 0 to give each simulation a file name
    if rank == 0:
        for params in params_list:
            if 'fname' in params:
                fname = params['fname']
            else:
                if 'IC' in params and params['IC'][0] == 'r':
                        fname = io.make_file_name(params, iterate = False)
                        #fname = io.make_file_name(params, iterate = True)
                    # don't iterate file names with a unique IC name
                else:
                    fname = io.make_file_name(params, iterate = False)
                    # set the file name for each simulation
                params['fname'] = fname
    # boradcast updated params list to each core
    params_list = comm.bcast(params_list, root=0)


# launch severl simulations in parallel
def launch_parallel(params_list, 
                    sim_tasks = ['one_site', 'two_site', 'IPR'],
                    rewrite_states = False,
                    measure_tasks=['s', 'sc', 'mom', 'g', 'm', 'nm', 'stats'],
                    coord_tasks=['xx', 'yy', 'zz'], 
                    nm_tasks=['ND', 'CC', 'Y'], 
                    corrj='L/2',
                    rewrite_measures = False ):
    comm, rank, nprocs = init_comm()
    init_files_parallel(comm, rank, params_list)
    for i, params in enumerate(params_list):
        # each core selects params to simulate without the need for a master
        if i % nprocs == rank:
            print_string = launch_single( params, comm=comm,
                                          sim_tasks = sim_tasks,
                                          rewrite_states = rewrite_states,
                                          measure_tasks=measure_tasks,
                                          coord_tasks=coord_tasks, 
                                          nm_tasks=nm_tasks, 
                                          corrj=corrj ,
                                          rewrite_measures = rewrite_measures )
            print(print_string)

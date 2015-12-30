#!/usr/bin/python3

# =============================================================================
# This script creates a local (3 qubit, 8x8) update operator, U, for the 16
# unitary ECAs and a unitary operator, V, to act on the center site if permitted
# by S.  I introduce a numbering scheme, S in [0, 1, ... 15]. The S relates to
# the original ECA code numer as (S, R):
#
# (0, 204), (1, 201), (2, 198),  (3, 195), (4, 156), (5, 153), (6, 150),
# (7, 147), (8, 108), (9, 105), (10, 102), (11, 99), (12, 60), (13, 157)
# (14, 57), (15, 51)
#
# Results are saved as .hdf5 and include single site reduced density matricies
# with key 'one_site', two site reduced density matrices : 'two_site'. The
# spin moments <Ai>, and <ABij> with A, B \in {X, Y, Z} are saved as symetric
# matrices. A != B has zero trace and A = B has <Ai> stored along the diagonal.
# 
#
# By Logan Hillberry
# =============================================================================

import warnings
import copy 
import time
import h5py

import numpy    as np
import fio      as io
import matrix   as mx
import states   as ss

from os.path import isfile
from collections import OrderedDict
from math import fabs
from cmath import sqrt

# Constructing U
# ==============

# The center site gets updated with V if the neighbors are in a suitable
# configuration. Otherwise, the center site remains unchanged.
def make_V(V, s):
    return s*V + (1-s)*ss.ops['I']


def make_U(S, V, R=False):
    # expand S into 4 digits of binary. 15 possible unitary ECAs 
    # MSB comes first in the list: [s11, s10, s01, s00]
    Sb = np.array(mx.dec_to_bin(S, 4))

    if R is True:
        # first arg interpreted as the usual ECA code number

        # compute swap rule with XOR 204, expand and extract S
        R = np.array(mx.dec_to_bin(204^S, 8))
        S1= np.take(R, [0, 1, 4, 5])
        S2 = np.take(R, [2, 3, 6, 7])
        if not np.array_equal(S1, S2):
            # check if rule number is unitary
            warning_string = 'R = '+str(S)+' is not a unitary rule number'
            warnings.warn(warning_string)
            Sb = S1 
        else:
            # set S
            Sb = S1 

    # Order S into matrix [ [s00, s01], [s10, s11] ]
    S_mat = Sb[::-1].reshape((2,2)) # add .T for old R numbering (pre winter 2015)

    # prepare projectors for looping over in the order  {|0><0|, |1><1|}
    neighbor_projs = [ss.ops['0'], ss.ops['1']] 

    # initialize U's
    U = np.zeros((8,8), dtype=complex)
    Ur = np.zeros((4,4), dtype=complex)
    Ul = np.zeros((4,4), dtype=complex)
    
    # loop through neighborhood configurations, applying V when logic of S permits
    for m, left_proj in enumerate(neighbor_projs):
        for n, right_proj in enumerate(neighbor_projs):
            Vmn = make_V(V, S_mat[m,n])
            U = U + mx.listkron([left_proj, Vmn, right_proj])

    # 2 qubit operator for right-most site, fix right boundary to |0>
    for m, left_proj in enumerate(neighbor_projs):
        n=0
        Vmn = make_V(V, S_mat[m,n])
        Ur = Ur + mx.listkron([left_proj, Vmn])
    
    # 2 qubit operator for left-most site, fix left boundary to |0>
    for n, right_proj in enumerate(neighbor_projs):
        m=0
        Vmn = make_V(V, S_mat[m,n])
        Ul = Ul + mx.listkron([Vmn, right_proj])

    return Ul, U, Ur


# Time evolution
# ==============

# update procedure for fixed BC's
# -------------------------------
def update_site(j, state, Ul, Uj, Ur, L):
    # site 0 has only one neighbor to the right
    if j == 0:
        js = [0,1]
        state = mx.op_on_state(Ul, js, state)

    # site L-1 has only one neighbor to the left
    elif j == L-1:
        js = [L-2, L-1]
        state = mx.op_on_state(Ur, js, state)

    # all other sites have left and right neighbors
    else:
        js = [(j-1), j, (j+1)]
        state = mx.op_on_state(Uj, js, state)
    return state


# check normalization of state
# ----------------------------
def check_norm(state, t, tol):
    # calculate inner product
    ip = (state.conj().dot(state)).real

    # check if ip == 1 within supplied tolorance
    if fabs(ip - 1.0) < tol:
        return state

    else: 
        warnings.warn('Non-normalized state at t = ' + str(t) + \
            ': <psi|psi> =' + str(ip) + ' has been normalized' )
        state = 1.0/sqrt(ip) * state
        return state 


# construct generator for exact time evolved quantum state
# --------------------------------------------------------
def time_evolve(params, tol=1E-10, state=None, norm_check=False):
    # load simulation parameters
    L = params['L'] 
    T = params['T']
    mode = params['mode']
    R = params['R']
    V = mx.listdot([ss.ops[k] for k in params['V']])
    
    # make update operators for left/right boundaries and th bulk
    Ul, Uj, Ur = make_U(R, V, R=True)
    
    # If no state supplied, make from the IC param
    if state is None:
        IC = params['IC']
        state = ss.make_state(L, IC)

    # yield the initial state
    state = np.array(state)
    yield state

    for t in range(T):

        # Sweep ECA 
        if mode=='sweep':
            for j in range(L):
                state = update_site(j, state, Ul, Uj, Ur, L)

            # don't check normalization by default
            if norm_check is True:
                state = check_norm(state, t, tol)
            yield state

        # Block ECA 
        elif mode=='block':
            for k in [0,1,2]:
                for j in range(k, L-1+k, 3):
                    if j!=L:
                        state = update_site(j, state, Ul, Uj, Ur, L)

            # don't check normalization by default
            if norm_check is True:
                state = check_norm(state, t, tol)
            # yield the updated state
            yield state

# import/create simulation results of full quantum state
# ------------------------------------------------------
def run_sim_full_save(params, force_rewrite = False, fname=None):
    #collect params needed for file name
    output_dir = params['output_dir']
    ic_name = io.make_IC_name(params['IC'])

    # make a default file name based on params
    if fname is None:
        fname = io.file_name(output_dir, 'data', io.sim_name(params), '.hdf5')

    # make a unique name for IC's made with a random throw
    if ic_name[0] == 'r' and isfile(fname) and not force_rewrite:
        fname_list = list(fname) 
        fname_list[-5] = str(eval(fname[-5])+1)
        fname = ''.join(fname_list)
    
    # check if file already exists and, if so, if it should be re-written
    if not isfile(fname) or force_rewrite:
        state_gen = time_evolve(params)
        io.write_states(fname, state_gen)

    state_gen = io.read_states(fname)
    return state_gen


# make indices of sites in the smaller half of a bipartite cut
# ------------------------------------------------------------
def bi_partite_inds(L, cut):
    inds = [i for i in range(cut+1)]
    if cut >= int(L/2):
        inds = np.setdiff1d(range(L), inds)
    return inds

# reduced density matrix of the smaller of all bi-partitions of the lattice
# -------------------------------------------------------------------------
def bi_partite_rdm(L, state):
    return np.array([mx.rdms(state, bi_partite_inds(L, cut)) 
            for cut in range(L-1)])

# compute the inverse participation ratio
# ---------------------------------------
def inv_participation_ratio(L, state):
    ipr = 0.0
    for basis_num in range(2**L):
        ipr = ipr + abs(state[basis_num])**4
    if ipr == 0.0:
        return 0.0
    return 1.0 / ipr

# import/create simulation results of one and two site reduced density matrices
# and all one and two point spin averages
# -----------------------------------------------------------------------------
def run_sim(params, force_rewrite = False,
        sim_tasks=['one_site', 'two_site', 'bi_partite']):
    # collect params needed for initialization
    L = params['L']
    T = params['T']
    
    fname = io.file_name(params)
    
    # check if file already exists and, if so, if it should be re-written
    if not isfile(fname) or force_rewrite:
        print('Running simulation...')

        # initialize arrays for data collection
        data = {}
        if 'one_site' in sim_tasks:
            data['one_site'] = np.zeros((T+1, L, 2, 2), dtype = complex)

        if 'two_site' in sim_tasks:
            data['two_site'] = np.zeros((T+1, L, L, 4, 4), dtype = complex)

        if 'bi_partite' in sim_tasks:
            # each cut is stored as a different data set of length T
            cdata = {}
            rdm_dims = [2**len(bi_partite_inds(L, cut)) 
                    for cut in range(L-1)]

            for cut, dim in enumerate(rdm_dims):
                cdata['bi_partite/cut'+str(cut)] = np.zeros((T+1, dim, dim), dtype=complex)

        # loop through quantum states
        for t, state in enumerate(time_evolve(params)):

            # first loop through lattice, make single site matrices
            for j in range(L):
                if 'one_site' in sim_tasks:
                    rtj = mx.rdms(state, [j])
                    data['one_site'][t, j][::] = rtj[::]

                # second loop through lattice
                if 'two_site' in sim_tasks:
                    for k in range(j+1, L):
                        rtjk = mx.rdms(state, [j, k])
                        data['two_site'][t, j, k][::] = rtjk[::]
                        data['two_site'][t, k, j][::] = rtjk[::]

            if 'bi_partite' in sim_tasks:
                for cut in range(L-1):
                    rtc = mx.rdms(state, bi_partite_inds(L, cut))
                    cdata['bi_partite/cut'+str(cut)][t][::] = rtc[::]
                    io.write_hdf5(fname, cdata)
        # write the simulation results to disk
        io.write_hdf5(fname, data, force_rewrite=force_rewrite)
    elif not force_rewrite:
        pass

    return



# Default behavior of this file
# =============================
if __name__ == "__main__":
    import csv
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import plotting as pt
    import processing as pp
    import fitting as ft

    font = {'family':'serif', 'size':10}
    mpl.rc('font',**font)


    # Simulation time scaling with L
    # ------------------------------
    # set up loop to time 1 iteration of evolution for increasing L
    L_list = [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    t_list = []
    for L in L_list:

    # dicts of params specifying the simulation
        params =  {
                        'output_dir' : 'testing/state_saving',
                        'fname' : '../output/testing/state_saving/cuts.hdf5',

                        'L'    : L,
                        'T'    : 1,
                        'mode' : 'sweep',
                        'R'    : 102,
                        'V'    : ['H'],
                        'IC'   : 'c1s0'
                                           }


        # get run time of simulation, append to timing list
        tic = time.time()
        run_sim(params, force_rewrite = True)
        toc = time.time()
        t_list.append(toc-tic)

    # save data to compare as improvements are made
    # NOTE: Change file names after each optimization!!
    data_fname = io.base_name('timing', 'data')+'cutting_nobi.csv'
    plots_fname = io.base_name('timing', 'plots')+'cutting_nobi.pdf'
    with open(data_fname, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(zip(L_list, t_list))

    # check if data imports
    with open(data_fname, 'r') as f:
        reader = csv.reader(f)
        # csv reader imports numbers as strings, convert them to floats
        timing_list = np.asarray([[float(val) for val in row] \
                for row in list(reader)])

    # get data into two lists
    L_list = timing_list[:,0]
    t_list = timing_list[:,1]
    Ls = np.linspace(min(L_list), max(L_list), 150)


    # fit data to an exponential form
    Bs, chi2 = ft.f_fits(ft.fexp, [0.0, 2.0, 0.0], L_list, t_list) 

    # plot the results
    plt.plot(L_list, t_list)
    plt.plot(Ls, ft.fexp(Bs, Ls),
            label = r"${%.6f} \times {%.2f}^L  {%+.3f}$" % (Bs[0], Bs[1], Bs[2]) 
            + "\n  $\chi^2 = {:.2e}$".format(chi2))
    plt.xlabel('L')
    plt.ylabel('time to simulate and save one iteration [s]')
    plt.legend(loc='upper left')
    plt.savefig(plots_fname)
    plt.close()

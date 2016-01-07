#!/usr/bin/python

# =============================================================================
# This script process the data saved by time_evolve
#
# By Logan Hillberry
# =============================================================================

from math import log
import copy
import numpy           as np
import scipy           as sp
import states          as ss
import matrix          as mx
import fio             as io
import h5py
import networkmeasures as nm

# Measures
# ========

# von Neumann entropy of reduced density matrix rho
# -------------------------------------------------
def vn_entropy(rho, tol=1e-14):
    evals = sp.linalg.eigvalsh(rho)
    s = -sum(el*log(el, 2) if el >= tol else 0.0  for el in evals)
    return s

# compute spacetime grid of local von Neumann entropy
# ---------------------------------------------------
def stj_calc(one_site, L, T):
    stj = np.array( [[vn_entropy(one_site[t, j]) 
          for j in range(L)] for t in range(T+1)] )
    return stj

# compute mutual information adjacency matrix at each time step
# -------------------------------------------------------------
def mtjk_calc(stj, two_site, L, T):
        mtjk = np.zeros((T+1, L, L))
        for t in range(T+1):
            for j in range(L):
                for k in range(j, L):
                    stjk = vn_entropy(two_site[t, j, k])
                    if j == k:
                        mtjk[t, j, j] = 0.0
                    else:
                        mtjk[t, j, k ] = mtjk[t, k, j] = \
                                0.5*(stj[t, j] + stj[t, k] - stjk)

        return mtjk

# wrappers for nm calculations
# ----------------------------
def invharmoniclen(net):
    return nm.harmoniclength(nm.distance(net))

def eveccentrality(net):
    return list(nm.eigenvectorcentralitynx0(net).values())

# calculate network measures on state
# -----------------------------------
def nm_calc(mtjk, typ = 'scalar', m_tasks=['ND', 'CC', 'Y']):
    nm_func_dict = { 'scalar' : {
                         'CC'  : nm.clustering,
                         'ND'  : nm.density,
                         'Y'   : nm.disparity,
                         'IHL' : invharmoniclen,
                        },

                'vector'  : { 
                         'CC'  : nm.localclustering,
                         'Y'   : nm.disparitylattice,
                         'EV'  : eveccentrality,
                         }
              }
    nm_dict = {} 
    for key in m_tasks:
        nm_dict[key] = np.array([nm_func_dict[typ][key](mjk) for mjk in mtjk])
    return nm_dict

# pull diagonals from a list of matrices
# --------------------------------------
def get_diag_vecs(mats):
    return np.array([mat.diagonal() for mat in mats])

# zero out the diagonal of list of matrices
# -----------------------------------------
def get_offdiag_mats(mats, diag_fill=0.0):
    L = len(mats[0])
    mats_out = copy.deepcopy(mats)
    for t, mat in enumerate(mats_out): 
        mat[np.arange(L), np.arange(L)] = diag_fill
        mats_out[t][::] = mat[::]
    return mats_out

# pull row of constant j from a list of matrices
# ----------------------------------------------
def get_row_vecs(mats, j=0):
    return np.array([mat[j, ::] for mat in mats])

# extract one and two point correlators
# -------------------------------------
def make_moments(moment, alphas=None, betas=None):
    if (alphas, betas) == (None, None):
        one_moment = get_diag_vecs(moment)
        one_moment_alpha = one_moment[::]
        one_moment_beta = one_moment[::]
        two_moment = get_offdiag_mats(moment, diag_fill=1.0)

    else:
        one_moment_alpha = get_diag_vecs(alphas)
        one_moment_beta = get_diag_vecs(betas)
        two_moment = moment 
    return one_moment_alpha, one_moment_beta, two_moment

# compute the correlator g_jk(t)
# ------------------------------
def gtjk( c2tjk, c1_alphatj, c1_betatk):
    gtjk = c2tjk - c1_alphatj * c1_betatk
    return gtjk 

# make spacetime grid of g_jk(t) with fixed j
# -------------------------------------------
def g_calc(moment_dict, L, T, g_tasks = ['xx', 'yy', 'zz']):
    # initialize dictionary of arrays to hold results 
    g_dict = {}
    for task, moment in moment_dict.items():
        g_dict['g'+task] = np.zeros((T+1, L, L))

        # extract moments
        alpha, beta = task[0], task[1]
        if alpha == beta:
            alphas, betas = None, None

        if alpha != beta:
            alphas = moment_dict[alpha+alpha]
            betas = moment_dict[beta+beta]
        one_moment_alpha, one_moment_beta, two_moment = \
                make_moments(moment, alphas=alphas, betas=betas)

        # fill correlation mtrices
        for t in range(T+1):
            for j in range(L):
                # second loop through the lattice fills g_jk symmetrically
                for k in range(j, L):
                    g_dict['g'+task][t, j, k] = g_dict['g'+task][t, k, j] =\
                          gtjk(two_moment[t, j, k], 
                               one_moment_alpha[t, j],
                               one_moment_beta[t, k] )
    return g_dict

# calculate the marix of spin moments
# -----------------------------------
def moments_calc(one_site, two_site, L, T, moment_tasks=['xx', 'yy', 'zz'] ):
    # construct 2pt ops for second order moment calculations
    twopt_ops = {task : mx.listkron([ss.ops[s.upper()]
        for s in task]) for task in moment_tasks}

    onept_ops = {task : mx.listdot([ss.ops[s.upper()]
        for s in task]) for task in moment_tasks}

    # initialize dictionary of arrays to hold results 
    moment_dict = {}
    for task in moment_tasks: 
        moment_dict[task] = np.zeros((T+1, L, L))

    for t in range(T+1):
        for j in range(L):
            rtj = one_site[t, j]
            
            # second loop through the lattice fills upper triangle
            for k in range(j, L):

                # store one-point moments on diagonals for non-cross correlators
                # knowing the diagonal ought to be one for the 2pt matrix
                if j == k:
                    if 'xx' in moment_tasks:
                        moment_dict['xx'][t, j, j] =\
                                np.trace(rtj.dot(ss.ops['X'])).real
                    if 'yy' in moment_tasks:
                        moment_dict['yy'][t, j, j] =\
                                np.trace(rtj.dot(ss.ops['Y'])).real
                    if 'zz' in moment_tasks:
                        moment_dict['zz'][t, j, j] =\
                                np.trace(rtj.dot(ss.ops['Z'])).real

                # store the diagonal for cross correlations propperly
                    for task in moment_tasks:
                        if task not in ('xx', 'yy', 'zz'):
                            moment_dict[task][t, j, j] =\
                                    np.trace(rtj.dot(onept_ops[task])).real

                # store two-point moments symmetrically on off diagonals
                else:
                    rtjk = two_site[t, j, k]
                    for task in moment_tasks:
                        moment_dict[task][t, j, k] = moment_dict[task][t, k, j] =\
                                np.trace( rtjk.dot(twopt_ops[task]) ).real
    return moment_dict

def stc_calc(bi_partite, L, T):
    stc = np.zeros((T+1, L-1))
    for t in range(T+1):
        for c in range(L-1):
            rct = bi_partite[c][t]
            stc[t, c] = vn_entropy(rct)
    return stc

# measure reduced density matrices. Includes making MI nets and measures
# -------------------------------------------------------------------------
def measure(params, fname, force_rewrite = False,
        moment_tasks = ['xx', 'yy', 'zz'], g_tasks = ['xx', 'yy', 'zz']):

    # detemine if measures have been made for this simulation yet
    f = h5py.File(fname, 'r')
    meas_ran = 'zz' in f
    f.close()

    if force_rewrite or not meas_ran:
        print('Measuring simulation...')

        # ensure zz moment is calculated. That is how the meas_ran is determined
        if 'zz' not in moment_tasks:
            moment_tasks.append('zz')

        L = params['L']
        T = params['T']
        # fname must be populated with one and two site density matricies
        one_site, two_site, bi_partite = \
                io.read_hdf5(fname, ['one_site', 'two_site', 'bi_partite'])

        # calculate local von Neumann entropy at each time step t and site j
        stj = stj_calc(one_site, L, T)

        # calculate local von Neumann entropy of the smaller of each bi-partitioning
        # of the lattice (cut c) at each time step t.
        stc = stc_calc(bi_partite, L, T)

        # calculate mutual information adjacency matices (with site indices j, k) at
        # each time step t.
        mtjk = mtjk_calc(stj, two_site, L, T)

        # calculate moment matrices (with site indices j, k) j=k holds the one point
        # correlator. j != k holds the two point correlator.
        moment_dict = moments_calc(one_site, two_site, L, T, 
                moment_tasks = moment_tasks)

        # calculate two point correlator matricies. g_tasks must be a subset of moment_tasks
        g_dict = g_calc(moment_dict, L, T, 
                g_tasks = g_tasks) 

        # store von Neumann entropy as a dictonary
        stj_dict = {'s' : stj}

        # store cut entropy as a dictonary
        stc_dict = {'sc' : stc}

        # store mutual info matrices as a dictonary
        mtjk_dict = {'m' : mtjk}

        # apply mutual information measures to the matrices
        nm_dict = nm_calc(mtjk)

        # collect all results in a single dictionary
        results = mx.listdicts([moment_dict, g_dict, stj_dict,stc_dict,  mtjk_dict, nm_dict])

        # write the results to the same file
        io.write_hdf5(fname, results, force_rewrite=force_rewrite)
    elif not force_rewrite:
        pass
    return

if __name__ == "__main__":
    import fio as io
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import plotting as pt
    import time_evolve

    font = {'family':'serif', 'size':10}
    mpl.rc('font',**font)

    params =  {
                    'output_dir' : 'testing/state_saving',

                    'L'    : 11,
                    'T'    : 100,
                    'mode' : 'block',
                    'R'    : 150,
                    'V'    : ['H','T'],
                    'IC'   : 'l0'
                                    }

    fname = time_evolve.run_sim(params, force_rewrite=False)

    measure(params, fname, force_rewrite=True, g_tasks=['xx', 'yy', 'zz',
        'xy'], moment_tasks=['xx','yy','zz', 'xy'])

    fig = plt.figure(1)
    ax = fig.add_subplot(111) 
    gzz = io.read_hdf5(fname, 'gzz')
    pt.plot_grid(get_row_vecs(gzz, j=0), ax, title=r'$g_2(X_0 Y_k; t)$', span=[0, 60], xlabel='site')
    plt.show()



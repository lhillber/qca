#!/usr/bin/python3

# =============================================================================
# This script process the data saved by time_evolve
#
# By Logan Hillberry
# =============================================================================

from math import log
import copy
import numpy           as np
import scipy           as sp
import simulation.states          as ss
import simulation.matrix          as mx
import simulation.fio             as io
import h5py
import simulation.networkmeasures as nm
import scipy.fftpack       as spf

# Measures
# ========

# von Neumann entropy of reduced density matrix rho
# -------------------------------------------------
def vn_entropy(rho, tol=1e-14):
    evals = sp.linalg.eigvalsh(rho)
    s = -sum(el*log(el, 2) if el >= tol else 0.0  for el in evals)
    return s

# make Fourier transform of time series data
# ------------------------------------------
def make_ft(time_series, dt=1):
    time_series = np.nan_to_num(time_series)
    Nsteps = len(time_series)
    times = [n*dt for n in range(Nsteps)]

    if Nsteps%2 == 1:
        time_sereis = np.delete(time_series,-1)
        Nsteps = Nsteps - 1

    # dt = 2*pi*dt
    time_series = time_series - np.mean(time_series)
    amps =  (2.0/Nsteps)*np.abs(spf.fft(time_series)[0:Nsteps/2])
    freqs = np.linspace(0.0,1.0/(2.0*dt), Nsteps/2)
    return freqs, amps

# compute spacetime grid of local von Neumann entropy
# ---------------------------------------------------
def stj_calc(results, L, T, tasks=None):
    one_site = results['one_site']
    stj = np.array( [[vn_entropy(one_site[t, j])
          for j in range(L)] for t in range(T+1)] )
    results['s'] = stj
    return stj

# compute mutual information adjacency matrix at each time step
# -------------------------------------------------------------
def mtjk_calc(results, L, T, tasks=None):
        stj = results['s']
        two_site = results ['two_site']
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

        results['m'] = mtjk
        return mtjk

# wrappers for nm calculations
# ----------------------------
def invharmoniclen(net):
    return nm.harmoniclength(nm.distance(net))

def eveccentrality(net):
    return list(nm.eigenvectorcentralitynx0(net).values())

# calculate network measures on state
# -----------------------------------
def nm_calc(results, L, T, tasks = ['ND', 'CC', 'Y']):

    typ = 'scalar'

    mtjk = results['m']
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
    for key in tasks:
        meas = np.array([nm_func_dict[typ][key](mjk) for mjk in mtjk])
        nm_dict[key] = meas[::]
        nm_dict['F'+key] = make_ft(meas)[1]

    results.update(nm_dict)
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
def g_calc(results, L, T, tasks = ['xx', 'yy', 'zz']):
    # initialize dictionary of arrays to hold results
    moment_dict = {task: results[task] for task in tasks}
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
    results.update(g_dict)
    return g_dict

# calculate the marix of spin moments
# -----------------------------------
def moments_calc(results, L, T, tasks=['xx', 'yy', 'zz'] ):
    one_site = results['one_site']
    two_site = results['two_site']
    # construct 2pt ops for second order moment calculations
    twopt_ops = {task : mx.listkron([ss.ops[s.upper()]
        for s in task]) for task in tasks}

    onept_ops = {task : mx.listdot([ss.ops[s.upper()]
        for s in task]) for task in tasks}

    # initialize dictionary of arrays to hold results
    moment_dict = {}
    for task in tasks:
        moment_dict[task] = np.zeros((T+1, L, L))

    for t in range(T+1):
        for j in range(L):
            rtj = one_site[t, j]

            # second loop through the lattice fills upper triangle
            for k in range(j, L):

                # store one-point moments on diagonals for non-cross correlators
                # knowing the diagonal ought to be one for the 2pt matrix
                if j == k:
                    if 'xx' in tasks:
                        moment_dict['xx'][t, j, j] =\
                                np.trace(rtj.dot(ss.ops['X'])).real
                    if 'yy' in tasks:
                        moment_dict['yy'][t, j, j] =\
                                np.trace(rtj.dot(ss.ops['Y'])).real
                    if 'zz' in tasks:
                        moment_dict['zz'][t, j, j] =\
                                np.trace(rtj.dot(ss.ops['Z'])).real

                # store the diagonal for cross correlations propperly
                    for task in tasks:
                        if task not in ('xx', 'yy', 'zz'):
                            moment_dict[task][t, j, j] =\
                                    np.trace(rtj.dot(onept_ops[task])).real

                # store two-point moments symmetrically on off diagonals
                else:
                    rtjk = two_site[t, j, k]
                    for task in tasks:
                        moment_dict[task][t, j, k] = moment_dict[task][t, k, j] =\
                                np.trace( rtjk.dot(twopt_ops[task]) ).real
    results.update(moment_dict)
    return moment_dict

def stc_calc(results, L, T, tasks=None):
    stc = np.zeros((T+1, L-1))
    for t in range(T+1):
        for c in range(L-1):
            rct = results['cut'+str(c)][t]
            stc[t, c] = vn_entropy(rct)
    results['sc'] = stc
    return stc

# measure reduced density matrices. Includes making MI nets and measures
# -------------------------------------------------------------------------
def measure(params, results, force_rewrite = False,
        measure_tasks=['s', 'sc', 'mom', 'g', 'm', 'nm'],
        coord_tasks=['xx', 'yy', 'zz'], nm_tasks=['ND', 'CC', 'Y']):

    if 'cut0' not in results:
        if 'sc' in measure_tasks:
            measure_tasks.remove('sc')
            print('can not measure sc (cut entropies) without bi_partite')


    fname = params['fname']
    # detemine if measures have been made for this simulation yet
    f = h5py.File(fname, 'r')
    meas_ran = 'zz' in f
    f.close()

    if force_rewrite or not meas_ran:
        print('Measuring simulation...')
        # ensure zz moment is calculated. That is how the meas_ran is determined
        if 'zz' not in coord_tasks:
            moment_tasks.append('zz')

        L = params['L']
        T = params['T']

        meas_map = {
            's' : stj_calc,
            'sc' :stc_calc,
            'm' : mtjk_calc,
            'nm' : nm_calc,
            'mom' : moments_calc,
            'g' : g_calc}

        for meas_task in measure_tasks:
            if meas_task in ('g', 'mom'):
                tasks = coord_tasks
            elif meas_task is 'nm':
                tasks = nm_tasks
            else:
                tasks = None
            meas_map[meas_task](results, L, T, tasks = tasks)

        # write the simulation results to disk
        io.write_hdf5(fname, results, force_rewrite=force_rewrite)
        print("data saved to: ", fname)
    elif not force_rewrite:
        with h5py.File(fname, 'r') as f:
            keys = [key for key in f.keys()]
        results = io.read_hdf5(fname, keys)
    return results

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



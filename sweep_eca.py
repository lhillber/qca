#!/usr/bin/python

# =============================================================================
# This script generalizes ECA to QECA with irreversible, asynchronus updating. 
# The state vector is renormalized at after each complete update since
# irreversibilities in ECA rules will not conserve probability. An update is
# complete once each state has been updated by the local update operator, which
# acts on one Moore neighborhood at a time. In otherwords, the update is
# asynchonus because the local update operator is swept across the lattice
# sequentially. 
#
# By Logan Hillberry
# =============================================================================




# Required libraries
# ==================

# standard libraries
# ------------------
import time
import copy
import json

from itertools import product, cycle
from os.path   import isfile
from cmath     import sqrt
from math      import pi, log
from os        import makedirs, environ

# additional libraries
# --------------------
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.sparse      as sps
import matplotlib        as mpl

from matplotlib.backends.backend_pdf import PdfPages
from scipy.linalg                    import kron
from mpi4py                          import MPI

# custom scripts
# --------------
import networkmeasures as nm
import matrix as mx
import states as ss
# default plot font to bold and size 18
# -------------------------------------
font = {'family':'normal', 'weight':'bold', 'size':16}
mpl.rc('font',**font)




# File I/O functions
# ==================

# string describing initial condition (IC)
# ----------------------------------------
def IC_name(IC):
    return '-'.join(['{:0.3f}{}'.format(val.real, name) \
                for (name, val) in IC])

# string describing simulation parameters
# ---------------------------------------
def sim_name(R, IC, L, tmax):
    return 'R{}_IC{}_L{}_tmax{}'.format( \
                R, IC_name(IC), L, tmax)

# make an output directory
# ------------------------
def base_name(output_name, output_type):
    bn = environ['HOME']+'/Documents/qca/output/' + output_name + '/' + output_type 
    makedirs(bn, exist_ok=True)
    return bn

# full path to a file to be opened
# --------------------------------
def file_name(output_name, output_type, name, ext):
    return base_name(output_name, output_type) + '/' + name + ext

# save simulation results
# -----------------------
def write_results(results, params):
    output_name, R, IC, L, tmax = params 
    results = np.asarray(results).tolist()
    with open(file_name(output_name,'data', sim_name(R, IC, L, tmax), '.res'), 'w') as outfile:
        json.dump(results, outfile)
    return

# load simulation results
# -----------------------
def read_results(params):
    input_name, R, IC, L, tmax = params 
    with open(file_name(input_name, 'data', sim_name(R, IC, L, tmax), '.res'), 'r') as infile:
       results =  json.load(infile)
    return results

# save multi page pdfs of plots
# -----------------------------
def multipage(fname, figs=None, clf=True, dpi=300): 
    pp = PdfPages(fname) 
    if figs is None:
        figs = [plt.figure(fignum) for fignum in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
        
        if clf==True:
            fig.clf()
    pp.close()
    return




# Measures
# ========

# probability of finding state at site j in state proj
# --------------------------------------------------
def prob(state, j, proj = '1'):
    L = int(log(len(state), 2)) 
    proj = ss.brhos[proj]
    op_state = mx.op_on_state(proj, [j], state) 
    return state.conj().dot(op_state).real 

# von Neumann entropy of reduced density matrix keeping klist from state
# ----------------------------------------------------------------------
def entropy(state, klist):
    rdm = mx.rdms(state, klist)
    L = int(log(len(rdm),2)) 
    evals = sp.linalg.eigvalsh(rdm)
    s = -sum(el*log(el,2) if el > 1e-14 else 0.  for el in evals)
    return s

# entropy of the smaller of all bi-partitions of the lattice
# ----------------------------------------------------------
def entropy_of_cut(state):
    L = int(log(len(state),2))
    klist = [ [i for i in range(mx)] if mx <= round(L/2) 
            else np.setdiff1d(np.arange(L), [i for i in range(mx)]).tolist() 
            for mx in range(1,L)]
    return [entropy(state, ks) for ks in klist ]

# compute mutual information network for state rho given a list of all
# single-site entropies (ss_entropy) ordered 0 to L
# --------------------------------------------------------------------
def MInetwork(state, ss_entropy):
    L = int(log(len(state),2)) 
    MInet = np.zeros((L,L))
    for i in range(L):
        MI = 0.
        MInet[i][i] = MI
        for j in range(i,L):
            if i != j:
                MI = ss_entropy[i] + ss_entropy[j] - entropy(state, [i,j])
                if MI > 1e-14:
                    MInet[i][j] = MI
                    MInet[j][i] = MI
                if MI<= 1e-14:
                    MInet[i][j] = 1e-14
                    MInet[j][i] = 1e-14
    return MInet


# calculate network measures on state state
# -----------------------------------------
def MIcalc(state, ss_entropy):
    MInet = MInetwork(state, ss_entropy)
    MICC = nm.clustering(MInet)
    MIdensity = nm.density(MInet)
    MIdisparity = nm.disparity(MInet)
    MIharmoniclen = nm.harmoniclength(nm.distance(MInet))
    MIeveccentrality = nm.eigenvectorcentralitynx0(MInet)
    return { 'net' : MInet.tolist(),
             'CC'  : MICC,
             'ND'  : MIdensity,
             'Y'   : MIdisparity,
             'IHL' : MIharmoniclen,
             'EV'  : list(MIeveccentrality.values()) }




# Plotting utilities
# ==================

# plot space time grids
# ---------------------
def board_plot(board_res, cmap, norm):
    vmin, vmax = 0.0, 1.0
    if np.max(board_res) > 1.0:
        vmax = np.max(board_res)
    if np.min(board_res) < 0.0:
        vmin = np.min(board_res)
    plt.imshow(board_res,
                    vmin = vmin,
                    vmax = vmax,
                    cmap = cmap,
                    norm = norm,
                    interpolation = 'none',
                    aspect = 'auto',
                    rasterized = True)

# plot probability, discretized probability, and single-site entropy
# ------------------------------------------------------------------
def board_plots(results, params, fignum=1, axs=(131, 132, 133)):
    output_name, R, IC, L, tmax = params 
    xtick_locs = range(0,L,2)
    xtick_lbls =xtick_locs
     
    prob_res      = results['prob'] 
    sse_res       = results['sse'] 
    disc_prob_res = [ [0 if p < 0.5 else 1 for p in p_vec] for p_vec in prob_res]
    
    disc_cmap = mpl.colors.ListedColormap([plt.cm.jet(0.), plt.cm.jet(1.)])
    bounds = [0,0.5,1]
    disc_norm = mpl.colors.BoundaryNorm(bounds, disc_cmap.N)
    
    titles = [' probability', ' projected prob.', ' s.s. entropy']
    results = [prob_res, disc_prob_res, sse_res]
    mycmaps = [plt.cm.jet, disc_cmap, plt.cm.jet]
    norms = [None, disc_norm, None]
    fig = plt.figure(fignum)
    for title, res, cmap, norm, ax in zip(titles, results, mycmaps, norms, axs):
        fig.add_subplot(ax)
        board_plot(res, cmap, norm)
        plt.title(title)
        plt.xlabel('site number')
        plt.xticks(xtick_locs, xtick_lbls)
        plt.ylabel('time')
        plt.colorbar()
    plt.suptitle('R ' + str(R)) 
    plt.tight_layout() 

def entropy_of_cut_plots(res, params, fignum=1, axs=(121, 122)):
    output_name, R, IC, L, tmax = params 

    xtick_locs = range(0,L-1,2)
    xtick_lbls = xtick_locs
     
    EC_res         = res['ec']
    center_cut_res = (np.array(EC_res).transpose()[int(L/2)]).transpose()

    fig =  plt.figure(fignum)

    fig.add_subplot(axs[0]) 
    board_plot(EC_res, plt.cm.jet, None)
    plt.title('Entropy of cut')
    plt.xlabel('cut number')
    plt.xticks(xtick_locs, xtick_lbls)
    plt.ylabel('time')
    
    fig.add_subplot(axs[1]) 
    plt.plot(range(tmax), center_cut_res)
    plt.title('Entropy of center cut')
    plt.xlabel('time')
    plt.ylabel('Entropy')
    plt.suptitle('R ' + str(R)) 
    plt.tight_layout() 

# plot network measures
# ---------------------
def nm_plot(res, params, fignum=1, ax=111):
    output_name, R, IC, L, tmax = params
    nm_keys = ['ND', 'CC', 'Y', 'IHL'] 

    fig = plt.figure(fignum)
    fig.add_subplot(ax)
    for key in nm_keys: 
        measure = res[key]
        plt.plot(range(tmax), measure, label = key)
    plt.title('R '+str (R) + ' Network measures')
    plt.legend()
    plt.tight_layout() 

# plot eigenvector centrality
# ---------------------------
def evec_centrality_plot(results, params, fignum=1, ax=111):
    output_name, R, IC, L, tmax = params
    fig = plt.figure(fignum) 
    fig.add_subplot(ax)
    board_plot(results['EV'], plt.cm.jet, None)
    plt.title('R ' + str(R) + 'Eigenvector Centrality')
    plt.xlabel('site number')
    plt.ylabel('time')
    plt.tight_layout()




# Sweep updating ECA
# ==================

def local_update_op(R, S):
    sxR = 204^R                                 # calculate swap rule s.t. 0 -> I, 1 -> sx
    sxR = mx.dec_to_bin(sxR, 2**S)[::-1]        # reverse so rule element 0 comes first
    
    op = np.zeros((2**S, 2**S), dtype=complex)
    for Rel_num, sxR_el in enumerate(sxR):      # Rel_num -> sxR_el:  000 -> 1,
                                                # 001 -> 0, etc
        op_sub_el_list = [] 
        for sub_Rel_num, proj_label in enumerate(mx.dec_to_bin(Rel_num, S)[::-1]):
            if sub_Rel_num == 1:                # sub_rel_num == 1 is center site
                op_sub_el = \
                        ss.pauli[str(sxR_el)].dot(ss.brhos[str(proj_label)]) 
            else:
                op_sub_el = ss.brhos[str(proj_label)]  # leave neighbors alone
            op_sub_el_list.append(op_sub_el)           # make the 3-site update op
        op = op + mx.listkron(op_sub_el_list) 
    return op

def time_evolve(R, IC, L, tmax):
    Tj = local_update_op(R, 3) 
    state = ss.make_state(L, IC)
    yield state 
    for t in np.arange(tmax):
        for j in range(L):
            js = [(j-1)%L, j, (j+1)%L]
            state = mx.op_on_state(Tj, js, state)
        ip = (state.conj().dot(state))
        if ip == 0.0:
            yield state
        else: 
            yield  1.0/sqrt(ip) * state
# measure state at step to generate 

# probability-board, single-site entropy board, and networkmeasures' time series
# saving reults to /data.
# ------------------------------------------------------------------------------
def measure_sim(params): 
    output_name, R, IC, L, tmax = params
    measures = [0]*(tmax+1)
    for t, state in enumerate(time_evolve(R, IC, L, tmax)): 
        ss_entropy = [entropy(state, [j]) for j in range(L)]
        measure = MIcalc(state, ss_entropy)
        measure['ec'] = entropy_of_cut(state)
        measure['sse'] = ss_entropy
        measure['prob'] = [prob(state, j) for j in range(L)]    
        measure['t'] = t
        measures[t] = measure
    results = {}
    for key in measure.keys(): 
        results[key] = [measures[t][key] for t in range(tmax)]
    write_results(results, params)
    
    return results

# import/create measurement results and plot them
# -----------------------------------------------
def run_sim(params, force_rewrite = False, name=None):
    output_name, R, IC, L, tmax = params
    if name is None:
        name = sim_name(R, IC, L, tmax)
    else:
        name = name
    
    if not isfile(file_name(output_name, 'data', sim_name(R, IC, L, tmax), '.res' )) \
        or force_rewrite:
        results = measure_sim(params)
    else:
        print('Importing results...')
        results = read_results(params)
 
    board_plots (results, params)
    nm_plot     (results, params, fignum=2) 
    evec_centrality_plot(results, params, fignum=3) 
    entropy_of_cut_plots(results, params, fignum=4) 
    multipage(file_name(output_name, 'plots', name, '.pdf'))    
    return



# Execute simulations
# ===================
# TODO:  make a one-shot function

# lists of parameters to simulate
# -------------------------------

output_name = 'sweep/classical_inspired_rules'

R_list = [102,90,91,60,153,165,195]

IC_list = [ [ ('c2l1', 1.0/sqrt(2)), ('c2l2',1.0/sqrt(2)) ] ]
L_list = [21]
tmax_list = [21]

params_list = [ (output_name, R, IC, L, tmax) \
        for R   in R_list     \
        for IC   in IC_list   \
        for L    in L_list    \
        for tmax in tmax_list ]

# run independent simulations in parallel
# ---------------------------------------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()
for i, params in enumerate(params_list):
    if i % nprocs == rank: 
        run_sim(params, force_rewrite=True)




# END
# ===

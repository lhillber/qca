#!/usr/bin/python3

# =============================================================================
# This script generalizes ECA rule number to a set of local unitary operators
# on a QCA lattice, simulates the dynamics, and plots the results.
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
from math      import pi, sqrt, log
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
import qca_util as util
import networkmeasures as nm




# Dictionaries of useful states and operators
# ===========================================

# default plot font to bold and size 18
# -------------------------------------
font = {'family':'normal', 'weight':'bold', 'size':18}
mpl.rc('font',**font)

# local 2-D Hilbert space (es = equal superposition)
# --------------------------------------------------
local_basis = { 'dead'  : np.array([[1.,0.]]).transpose(),
                'alive' : np.array([[0.,1.]]).transpose(),
                'es'    : np.array([[1./sqrt(2), 1./sqrt(2)]]).transpose() }

# convert pairs of rule elements with the 
# same neighbors to operator elements.
# ---------------------------------------
uabi = { '00' : (np.array([[1,0],[0,0]]), np.array([[0,1],[0,0]])), 
         '01' : (np.array([[0,1],[1,0]]), np.array([[0,0],[0,0]])), 
         '10' : (np.array([[1,0],[0,1]]), np.array([[0,0],[0,0]])),
         '11' : (np.array([[0,0],[0,1]]), np.array([[0,0],[1,0]]))}

# pure density matrix of local basis
# ----------------------------------
local_rhos = { 0 : local_basis['dead'].dot(local_basis['dead'].transpose()), 
               1 : local_basis['alive'].dot(local_basis['alive'].transpose()) }

# possible configurations of sites j-1 and j+1 
# ordered (|00><00|, |01><01|, |10><10|, |11><11|)
# ------------------------------------------------
neighbor_confs = [ list(map(lambda local_ket: local_rhos[local_ket], conf)) \
        for conf in product((0,1), repeat=2)] 



# File I/O functions
# ==================

# string describing initial condition (IC)
# ----------------------------------------
def IC_name(IC):
    return '-'.join(['{:0.3f}{}'.format(val, name) \
                for (name, val) in IC])

# string describing mixture of ECA rules
def Rs_name(Rs):
    return '-'.join([str(R) for R in Rs])
 
# string describing simulation parameters
# ---------------------------------------
def sim_name(Rs, IC, L, tmax):
    return 'R{}_IC{}_L{}_tmax{}'.format( \
                Rs_name(Rs), IC_name(IC), L, tmax)

# make an output directory
# ------------------------
def base_name(output_name, output_type):
    bn = environ['HOME']+'/Documents/eca/output/' + output_name + '/' + output_type 
    makedirs(bn, exist_ok=True)
    return bn

# full path to a file to be opened
# --------------------------------
def file_name(output_name, output_type, name, ext):
    return base_name(output_name, output_type) + '/' + name + ext

# save simulation results
# -----------------------
def write_results(results, params):
    output_name, Rs, IC, L, tmax = params 
    results = np.asarray(results).tolist()
    with open(file_name(output_name,'data', sim_name(Rs, IC, L, tmax), '.res'), 'w') as outfile:
        json.dump(results, outfile)
    return

# load simulation results
# -----------------------
def read_results(params):
    input_name, Rs, IC, L, tmax = params 
    with open(file_name(input_name, 'data', sim_name(Rs, IC, L, tmax), '.res'), 'r') as infile:
       results =  json.load(infile)
    return results




# Useful functions
# ===============

# reduced density matrix starting from density matrix
# NOTE: klist is list of site numbers to keep, indexed from 0
# -----------------------------------------------------------
# TODO: improve method!
def rdmr(rho, klist):
    L = int(log(rho.get_shape()[0], 2)) 
    rest = np.setdiff1d(np.arange(L), klist)
    local_bi_list = map(lambda bi: np.array([bi]).transpose(), np.eye(2))
    bi_list = product(local_bi_list, repeat=len(rest)) 
    RDM = sps.csr_matrix([[0]*(2**len(klist)) for i in range(2**len(klist))])
    for bi in bi_list:
        L_prod_list = [np.eye(2)]*L
        R_prod_list = [np.eye(2)]*L
        for i, loc in enumerate(rest):
            L_prod_list[loc] = bi[i].transpose() 
            R_prod_list[loc] = bi[i]
        RDM += util.spmatkron(L_prod_list)*rho*util.spmatkron(R_prod_list)
    
    return RDM

# reduced density matrix starting from state vector
# NOTE: klist is list of site numbers to keep, indexed from 0
# -----------------------------------------------------------
def rdms(state, klist):
    L = int(log(len(state), 2))
    n = len(klist)
    rest = np.setdiff1d(np.arange(L), klist)
    ordering = []
    ordering = klist+list(rest)
    block = state.reshape(([2]*L))
    block = block.transpose(ordering)
    block = block.reshape(2**n,2**(L-n))
    RDM = np.zeros((2**n,2**n), dtype=complex)
    
    tot = complex(0,0)
    for i in range(2**n-1):
        Rii = sum(np.multiply(block[i,:], np.conj(block[i,:])))
        tot = tot+Rii
        RDM[i][i] = Rii
        for j in range(i,2**n):
            if i != j:
                Rij = np.inner(block[i,:], np.conj(block[j,:]))
                RDM[i][j] = Rij
                RDM[j][i] = Rij
    RDM[2**n-1,2**n-1] = complex(1,0)-tot
    return RDM

# hermitian conjugate of matrix
# -----------------------------
def dagger(mat):
    return mat.conj().transpose()

# probability of finding rho at site j in state proj
# --------------------------------------------------
def prob(rho, j, proj = 'alive'):
    L = int(log(rho.get_shape()[0], 2)) 
    I = np.eye(2) 
    proj = local_basis[proj]
    p = proj.dot(proj.transpose())
    P = [I]*L
    P[j] = p
    P = sps.csr_matrix(util.matkron(P))
    return sum((rho*P).diagonal())

# von Neumann entropy of reduced density matrix described by rho and klist
# ------------------------------------------------------------------------
def entropy(rho, klist):
    rdm = rdmr(rho, klist)
    L = int(log(rdm.get_shape()[0],2)) 
    evals = sp.linalg.eigvalsh(rdm.todense())
    np.append(evals, 1 - sum(evals))
    s = -sum(el*log(el,2) if el > 1e-14 else 0.  for el in evals)
    return s

# convert base ten number (dec) to binary padded to length L
# ----------------------------------------------------------
def dec_to_bin(dec, L):
    return '{0:0b}'.format(dec).rjust(L, '0')

# compute mutual information network for state rho given a list of all
# single-site entropies (ss_entropy) ordered 0 to L
# --------------------------------------------------------------------
def MInetwork(rho, ss_entropy):
    L = int(log(rho.get_shape()[0],2)) 
    MInet = np.zeros((L,L))
    for i in range(L):
        for j in range(i,L):
            if i != j:
                MI = ss_entropy[i] + ss_entropy[j] - entropy(rho, [i,j])
                if MI > 1e-14:       # mutual information cutoff
                    MInet[i][j] = MI
                    MInet[j][i] = MI
    return MInet

# calculate network measures (CC, ND, Y, and IHL) on state rho
# ------------------------------------------------------------
def MIcalc(rho, ss_entropy):
    MInet = MInetwork(rho, ss_entropy)
    MICC = nm.clustering(MInet)
    MIdensity = nm.density(MInet)
    MIdisparity = nm.disparity(MInet)
    MIharmoniclen = nm.harmoniclength(nm.distance(MInet))
    return { 'Y'   : MIdisparity,   \
             'CC'  : MICC,          \
             'ND'  : MIdensity,     \
             'IHL' : MIharmoniclen, \
             'net' : MInet.tolist() }




# Plotting utilities
# ==================
def board_plot(board_res, cmap, norm):
    plt.imshow(board_res,
                    vmin = 0.,
                    vmax = 1.,
                    cmap = cmap,
                    norm = norm,
                    interpolation = 'none',
                    aspect = 'auto',
                    rasterized = True)

# plot probability board (prob_res)
# ---------------------------------
def board_plots(results, params, fignum=1, axs=(131, 132, 133)):
    output_name, Rs, IC, L, tmax = params 
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
        plt.xlabel('Site Number')
        plt.xticks(xtick_locs, xtick_lbls)
        plt.ylabel('Time')
    plt.suptitle('R ' + Rs_name(Rs)) 
    plt.tight_layout() 

# plot network measures
# ---------------------
def nm_plot(res, params, fignum=1, ax=111):
    output_name, Rs, IC, L, tmax = params
    nm_keys = ['ND', 'CC', 'Y', 'IHL'] 

    fig = plt.figure(fignum)
    fig.add_subplot(ax)
    for key in nm_keys: 
        measure = res[key]
        plt.plot(range(tmax), measure, label = key)
    plt.title('R'+Rs_name(Rs) + ' Network measures')
    plt.legend()
    plt.tight_layout() 





# Functions for simulations
# =========================

# shape the rule number by type of neighbor configuration
# same order as neighbor_confs at top of file
# -------------------------------------------------------
def shape_R(R):
    R = list(dec_to_bin(R, 8))
    R = np.concatenate((R[::2], R[1::2])).reshape((4,2)) 
    R[[1,2]] = R[[2,1]]
    R = R[::-1]
    R = map(lambda x: ''.join(x), R)
    return R 

# build operator elements for site j for ECA rule number R and L sites
# assuming concatenation of lattice state to itself (update first, read second)
# -----------------------------------------------------------------------------
def build_local_R(R, j, L):
    R = shape_R(R)
    I = np.eye(2)
    F0j, F1j = sps.csr_matrix([[0]*2**(2*L) for i in range(2**(2*L))]), \
               sps.csr_matrix([[0]*2**(2*L) for i in range(2**(2*L))])

    for local_conf, uab in zip(neighbor_confs, R):
        F0_list            = [I]*(2*L)
        F0_list[j]         = uabi [uab][0]
        F0_list[L+(j-1)%L] = local_conf[0]
        F0_list[L+(j+1)%L] = local_conf[1]
        
        F1_list            = [I]*(2*L)
        F1_list[j]         = uabi[uab][1]
        F1_list[L+(j-1)%L] = local_conf[0]
        F1_list[L+(j+1)%L] = local_conf[1]

        F0j += util.spmatkron(F0_list)
        F1j += util.spmatkron(F1_list)

    return F0j, F1j

# time evolve an initial lattice state with operator-sum formalism
# collecting lattice state (rho) at each time step into a list (rho_list)
# -----------------------------------------------------------------------
def time_evolve(Rs, IC, L, tmax):
    init_state = util.make_state(L, IC)
    rho = sps.csr_matrix(init_state.dot(init_state.transpose()))
    rho_list    = [rho]*tmax
    timing_list = [0.0]*(tmax-1)
    
    for t in range(1, tmax):
        tic = time.time() 
        rho_copy = copy.deepcopy(rho)
        rho_Rs = sps.csr_matrix([ [0]*2**(L) for i in range(2**(L)) ]) 
        for R in Rs:
            xsi = sps.kron(rho_copy, rho_copy)
            for j in range(L):
                F0j, F1j = build_local_R(R, j, L)
                xsi = F0j * xsi * dagger(F0j) + \
                      F1j * xsi * dagger(F1j)
            rho_el = rdmr(xsi, range(L))
            rho_Rs = rho_Rs + rho_el
        rho = rho_Rs.multiply(1.0/len(Rs))
        rho_list[t] = rho
        toc = time.time()
        timing = toc-tic
        print('t = ',t, ' of ', tmax, ' took {:0.3f}'.format(timing), ' s')
        timing_list[t-1] = timing
    timing_stats = (np.mean(timing_list), np.std(timing_list))
    return rho_list, timing_stats 

# measure rho at step to generate 
# probability-board, single-site entropy board, and networkmeasures' time series
# saving reults to /data.
# ------------------------------------------------------------------------------
def measure_sim(params): 
    tic = time.time() 
    output_name, Rs, IC, L, tmax = params
    rho_list, gen_timing_stats = time_evolve(Rs, IC, L, tmax)
    measures = [0]*tmax 
    toc = time.time()
    for t, rho in enumerate(rho_list): 
        ss_entropy = [entropy(rho, [j]) for j in range(L)]
        measure = MIcalc(rho, ss_entropy)
        measure['sse'] = ss_entropy
        measure['prob'] = [prob(rho, j) for j in range(L)]    
        measure['t'] = t
        measures[t] = measure
    results = {}
    for key in measure.keys(): 
        results[key] = [measures[t][key] for t in range(tmax)]
    write_results(results, params)
    tuc = time.time()
    
    print('All generations took', toc - tic, ' s')
    print('All measurements took', tuc - toc, ' s')
    return results, gen_timing_stats

# import/create measurement results and plot them
# -----------------------------------------------
def run_sim(params, force_rewrite = False, name=None):
    output_name, Rs, IC, L, tmax = params
    if name is None:
        name = sim_name(Rs, IC, L, tmax)
    else:
        name = name
    
    if not isfile(file_name(output_name, 'data', sim_name(Rs, IC, L, tmax), '.res' )) \
        or force_rewrite:
        results, timing_stats = measure_sim(params)
        print('timing per generation (mean +- std): {:0.3f}+-{:0.3f}'.format( \
               *timing_stats), ' s')
    else:
        print('Importing results...')
        results = read_results(params)
 
    board_plots (results, params)
    nm_plot     (results, params, fignum=2) 
    multipage(file_name(output_name, 'plots', name, '.pdf'))    
    return

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



# Execute simulations
# ===================
# TODO: put into parallel function and make a one-shot function

# lists of parameters to simulate
# -------------------------------
IC_list = [ [('c1d1', 1.0)], [('c1l1', 1.0)], \
       [('c2d3', 1.0)], [('c2l3', 1.0)], [('c2E0_1', 1.0)], \
       [('c3d5', 1.0)], [('c3l5', 1.0)], [('c3E0_2', 1.0)] ] 


output_name = '110'
Rs_list = [[110],[110, 212], [110, 206, 220], [110,106]]
IC_list = [ [('c1d1',1.0)] ]
L_list = [5]
tmax_list = [10]

params_list = [ (output_name, Rs, IC, L, tmax) \
        for Rs   in Rs_list   \
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
        run_sim(params, force_rewrite=False)



# END
# ===

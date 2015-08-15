#!/usr/bin/python

# =============================================================================
# This script generates classical elementary cellular automata with sweep
# updating
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
import information as im
import measures as ms
import plotting as pt
import fio as io
# default plot font to bold and size 18
# -------------------------------------
font = {'family':'normal', 'weight':'bold', 'size':16}
mpl.rc('font',**font)




# File I/O functions
# ==================

# string describing initial condition (IC)
# ----------------------------------------
def IC_name(IC):
    return '-'.join(['{:0.3f}{}'.format(val, name) \
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

def time_evolve(rule_dict, state, L, tmax):
    state_copy = copy.copy(state) 
    state_list = [state_copy]*tmax
    for t in range(1, tmax):
        for j in range(L):
            Nj = [(j-1)%L, j, (j+1)%L]
            local_state = state_copy[Nj]
            state[j] = rule_dict[tuple(local_state)]
        state_copy = copy.copy(state)
        state_list[t] = state_copy
    return state_list

def sweep_time_evolve(rule_dict, IC, L, tmax):
    state = copy.copy(IC)
    state_list = [IC]*tmax
    for t in range(1, tmax):
        for j in range(L):
            Nj = [(j-1)%L, j, (j+1)%L]
            local_state = state[Nj]
            state[j] = rule_dict[tuple(local_state)]
        state_list[t]  = copy.copy(state)
    return state_list



def run_sim(R, IC, L, tmax, mode = 'sweep'):
    IC = np.array(IC)
    Rb = mx.dec_to_bin(R, 8)[::-1]
    
    neighborhood_basis = (i_Nj for i_Nj in [mx.dec_to_bin(d, 3) 
                               for d in range(8)])
    
    rule_dict = {tuple(Nj) : Rb[d] for d, Nj in 
            zip(np.arange(8), neighborhood_basis)}
    if   mode is 'ECA': 
         state_list = time_evolve(rule_dict, IC, L, tmax)
    elif mode is 'sweep':
         state_list = sweep_time_evolve(rule_dict, IC, L, tmax)
    return state_list


def plot_all(L, IC, tmax, fname):
    for R in range(256):
        fignum = R
        fig = plt.figure(fignum) 
        
        fig.add_subplot(121)
        state_list = run_sim(R, IC, L, tmax, mode = 'ECA') 
        board_plot(state_list, plt.cm.jet, None)
        plt.xlabel('site number')
        plt.ylabel('time')
        #plt.show()
        plt.title('ECA R ' + str(R)) 
        
        
        fig.add_subplot(122)
        state_list = run_sim(R, IC, L, tmax, mode = 'sweep') 
        board_plot(state_list, plt.cm.jet, None)
        plt.xlabel('site number')
        plt.ylabel('time')
        #plt.show()
        plt.title('Sweep ECA R ' + str(R)) 
        plt.tight_layout()
    multipage(file_name('classical', 'plots',  fname, '.pdf')) 


def local_probabilities(boards, w, t, i, j):
    if len(boards)!=len(w):
        print( "Number of Weights must be equal to number of boards!" )
    else:
        nb  = len(w)
        pi  = np.zeros((2,2))
        pj  = np.zeros((2,2))
        pij = np.zeros((4,4))
        boards = np.asarray(boards)
        for b in range(nb):
            if boards[b,t,i] == 0:
                pi[0,0] += w[b]
                if boards[b,t,j] == 0:
                    pj[0,0]  += w[b]
                    pij[0,0] += w[b]
                else:
                    pj[1,1]  += w[b]
                    pij[1,1] += w[b]
            else:
                pi[1,1] += w[b]
                if boards[b,t,j] == 0:
                    pj[0,0]  += w [b]
                    pij[2,2] += w[b]
                else:
                    pj[1,1]  += w[b]
                    pij[3,3] += w[b]
    return { 'pi' : pi, 'pj' : pj, 'pij' : pij }

def plogp(p):
    #If probability ~ 0 we adopt the standard convention
    #that plogp = 0.  This is justified by the fact that
    #lim_{x->0} x*logx = 0.
    if p<1e-14:
        return 0.0
    else:
        return p*log(p, 2)

def shannon_entropy(probabilities):
    probabilities = probabilities.diagonal() 
    s = 0.0
    for p in probabilities:
        s -= plogp(p)
    return s 

def local_mi(boards, w, t, i, j):
    p_dict = local_probabilities(boards, w, t, i, j)
    si  = shannon_entropy(p_dict['pi'])
    sj  = shannon_entropy(p_dict['pj'])
    sij = shannon_entropy(p_dict['pij'])
    return 0.5 * (si + sj - sij)

def spatialnetworkC(boards,w):
    if len(boards)!=len(w):
        print( "Number of Weights must be equal to number of boards!" )
    else:
        L = len(boards[0][0])
        tmax = len(boards[0])
        mi_nets=np.zeros((tmax,L,L))
        for t in range(tmax):
            for j in range(L):
                for k in range(j): # ignore diagonal elements
                    mi = local_mi(boards, w, t, j, k)
                    if mi>1e-14:
                        mi_nets[t,j,k] = mi_nets[t,k,j] = mi
                    else:
                        mi_nets[t,j,k] = mi_nets[t,k,j] = 1e-14
        return mi_nets


def index(L, config):
    state = [0]*L
    js = map(int, config.split('_'))
    for j in js:
        state[j] = 1
    return state

def center(L, config):
    len_cent = int(config[0])
    len_back = L - len_cent
    len_L = int(len_back/2)
    len_R = len_back - len_L
    cent_IC = [(config[1:], 1.0)]
    left = [0]*len_L
    cent = make_states_to_mix(len_cent, cent_IC)[0][0]
    right = [0]*len_R
    if len_back == 0:
        return cent
    elif len_back == 1:
        return cent + right
    else:
        return left + cent + right

def all_zero(L, config):
    return [0]*L

def all_one(L, config):
    return [1]*L

def make_states_to_mix(L, IC):
    state_map = { 'i' : index,
                  'z' : all_zero,
                  'o' : all_one,
                  'c' : center }
    
    states_to_mix = []
    for s in IC:
        name = s[0][0]
        config = s[0][1:]
        w = s[1]
        state = state_map[name](L, config)
        pair = (state, w)
        states_to_mix.append(pair)
    return states_to_mix

def measure_networks(nets, tasks):
    measures = {} 
    for task in tasks:
        measures[task] = [ms.NMcalc(net, tasks=tasks)[task] for net in nets]
    return measures

def plot_time_series(data, label, title, loc='lower right', fignum=1, ax=111):
    fig = plt.figure(fignum)
    fig.add_subplot(ax)
    plt.plot(range(len(data)), data, label=label)
    plt.title(title)
    plt.legend(loc=loc)
    plt.tight_layout() 

def plot_spacetime_grid(data, title, cmap=plt.cm.jet, norm=None, fignum=1, ax=111):
    vmin, vmax = 0.0, 1.0
    if np.max(data) > 1.0:
        vmax = np.max(board_res)
    if np.min(data) < 0.0:
        vmin = np.min(board_res)
    
    fig = plt.figure(fignum)
    fig.add_subplot(ax)
    plt.imshow(data,
                    vmin = vmin,
                    vmax = vmax,
                    cmap = cmap,
                    norm = norm,
                    interpolation = 'none',
                    aspect = 'auto',
                    rasterized = True)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout() 

def network_plots(data, tasks = ['ND', 'CC','Y', 'IHL'], fignum=10):
    title = 'Network Measures on classical mi'
    for i, task in enumerate(tasks):
        dat = np.array(data[task]) + 0.0
        label = task
        plot_time_series(dat, label, title, fignum=fignum)

def probability_plots(data, fignum=1, ax=131):
    board_list, w_list = data
    row_col = board_list[0].shape
    mixed_board = np.zeros((row_col))
    for i, (board, w) in enumerate(zip(board_list, w_list)):
        plot_spacetime_grid(board, 'IC'+str(i) + ' w = ' + str(w),  
                ax=ax, fignum=fignum)
        mixed_board += w * board 
        ax += 1
    plot_spacetime_grid(mixed_board, 'mixed', ax=ax, fignum=fignum)

#Parrallelize?
def gen_classical_mixture(paramsC):
    output_name, R, IC, L, tmax = paramsC
    data_list = [0]*len(IC)
    w_list = [0]*len(IC)
    for count, (state, w) in enumerate(make_states_to_mix(L, IC)): 
        data_list[count] = run_sim(R, state, L, tmax)
        w_list[count] = w
    return np.asarray(data_list), w_list

output_name = 'comp'

R_list  = [ 91 ]

IC_listC = [[('z', 0.5), ('c2i0_1', 0.5)], [('c2i0', 0.5), ('c2i1', 0.5)], [('z', 0.5), ('c2i1', 0.5)] ]

L_list    = [ 10 ]

tmax_list = [ 35 ]

params_listC = [ (output_name, R, IC, L, tmax) \
        for L    in L_list    \
        for tmax in tmax_list \
        for R    in R_list    \
        for IC   in IC_listC   ]

def plot_main(params, name=None):
    output_name, R, IC, L, tmax = params
    if name is None:
        name = io.sim_name(R, IC, L, tmax)
    else:
        name = name
    mix_data = gen_classical_mixture(params)
    probability_plots(mix_data, fignum=1) 
    mi_nets = spatialnetworkC(*mix_data)
    mi_measures = measure_networks(mi_nets, ['ND', 'CC','Y','IHL'])
    network_plots(mi_measures, fignum=2)
    io.multipage(io.file_name(output_name, 'plots', 'C'+name, '.pdf'))    

for params in params_listC:
    plot_main(params)




#plot_all(L, IC, tmax,  'many_periods')

# END
# ===

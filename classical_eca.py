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





def time_evolve(rule_dict, init_state, L, tmax):
    state_copy = init_state.copy() 
    state_list = np.zeros((tmax, L))
    state_list[0] = init_state
    for t in range(1, tmax):
        state = state_list[t]
        for j in range(L):
            Nj = [(j-1)%L, j, (j+1)%L]
            local_state = state_copy.take(Nj)
            state[j] = rule_dict[tuple(local_state)]
        state_copy = state.copy()
        state_list[t] = state
    return state_list

def sweep_time_evolve(rule_dict, init_state, L, tmax):
    state_copy = init_state.copy() 
    state_list = np.zeros((tmax, L))
    state_list[0] = state_copy
    for t in range(1, tmax):
        for j in range(L):
            Nj = [(j-1)%L, j, (j+1)%L]
            local_state = state_copy.take(Nj)
            state_copy[j] = rule_dict[tuple(local_state)]
        state_list[t]  = state_copy
    return state_list

def sweep_time_evolve_between(rule_dict, init_state, L, tmax):
    state_copy = init_state.copy() 
    state_list = np.zeros((tmax*L, L))
    state_list[0] = state_copy
    it = 0
    for t in range(1, tmax):
        for j in range(L):
            Nj = [(j-1)%L, j, (j+1)%L]
            local_state = state_copy.take(Nj)
            state_copy[j] = rule_dict[tuple(local_state)]
            state_list[it]  = state_copy
            it += 1
    return state_list

def run_sim(R, IC, L, tmax, mode = 'sweep'):
    Rb = mx.dec_to_bin(R, 8)[::-1]
    neighborhood_basis = (i_Nj for i_Nj in [mx.dec_to_bin(d, 3) 
                               for d in range(8)])
    
    rule_dict = {tuple(Nj) : Rb[d] for d, Nj in 
                 enumerate(neighborhood_basis) }
    if   mode is 'ECA': 
         state_list = time_evolve(rule_dict, IC, L, tmax)
         return state_list

    elif mode is 'sweep':
         state_list = sweep_time_evolve(rule_dict, IC, L, tmax)
         return state_list

    elif mode is 'between':
         state_list = sweep_time_evolve_between(rule_dict, IC, L, tmax)
         return state_list

# Initial States
# ==============
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

def binary(L, config):
    return mx.dec_to_bin(int(config[0:]), L)

def all_states(L, config):
    IC = [('b'+str(i), 1.0 / 2.0**float(L)) for i in range(2**L)]
    return make_states_to_mix(l, ic) 

def make_states_to_mix(L, IC):
    state_map = { 'i' : index,
                  'z' : all_zero,
                  'o' : all_one,
                  'c' : center, 
                  'b' : binary,
                  'a' : all_states }
    
    states_to_mix = []
    for s in IC:
        name = s[0][0]
        config = s[0][1:]
        if name == 'a':
            return state_map[name](L, config)
        w = s[1]
        state = state_map[name](L, config)
        pair = (state, w)
        states_to_mix.append(pair)
    return states_to_mix



# Mixing IC's
# ===========
def gen_classical_mixture(params):
    output_name, R, IC, L, tmax = params
    data_list = {}
    w_list = []
    for count, (state, w) in enumerate(make_states_to_mix(L, IC)): 
        state = np.array(state) 
        data_list[count] = run_sim(R, state, L, tmax)
        w_list.append(w)
    return data_list, w_list

def measure_sim(params): 
    output_name, R, IC, L, tmax = params
    data_list, w_list = gen_classical_mixture(params)
    mi_nets = spatialnetworkC(data_list, w_list)
    return {'b' : data_list, 'w' : w_list, 'mi' : mi_nets} 

def run_mixture(params, force_rewrite = False):
    output_name, R, IC, L, tmax = params
    if not isfile(io.file_name(output_name, 'data', io.sim_name(R, IC, L, tmax), '.res' )) \
        or force_rewrite:
        results = measure_sim(params)
    else:
        results = io.read_results(params, typ='C')
    io.write_results(results, params, typ='C')
    return results



# calculations on a mixture of IC's
# =================================

def local_probabilities(boards, w, t, i, j):
    if len(boards)!=len(w):
        print( "Number of Weights must be equal to number of boards!" )
    else:
        nb  = len(w)
        pi  = np.zeros((2,2))
        pj  = np.zeros((2,2))
        pij = np.zeros((4,4))
        for b in range(nb):
            if boards[b][t,i] == 0:
                pi[0,0] += w[b]
                if boards[b][t,j] == 0:
                    pj[0,0]  += w[b]
                    pij[0,0] += w[b]
                else:
                    pj[1,1]  += w[b]
                    pij[1,1] += w[b]
            else:
                pi[1,1] += w[b]
                if boards[b][t,j] == 0:
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
    if p > 1e-14:
        return p*log(p, 2)
    else:
        return 0.0

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

def spatialnetworkC(boards, w, tol=1e-14):
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
                    if mi > tol:
                        mi_nets[t,j,k] = mi_nets[t,k,j] = mi
                    else:
                        mi_nets[t,j,k] = mi_nets[t,k,j] = tol
        return mi_nets

def measure_networks(nets, tasks=['Y','CC'], typ='avg'):
    measures = {} 
    for task in tasks:
        measures[task] = np.asarray([ms.NMcalc(net, typ=typ,
                                    tasks=tasks)[task] for net in nets])
    return measures



# plotting
# ========
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
        vmax = np.max(data)
    if np.min(data) < 0.0:
        vmin = np.min(data)
    
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

def avg_nm_plots(avg_data, tasks = ['ND', 'CC','Y'], fignum=1):
    title = 'Network Measures on classical mi'
    for i, task in enumerate(tasks):
        dat = avg_data[task]
        label = task
        plot_time_series(dat, label, title, fignum=fignum)

def st_nm_plots(st_data, tasks = ['EV', 'CC', 'Y'], fignum=1):
    ax = 131 
    for i, task in enumerate(tasks):
        title = task
        dat = st_data[task]
        label = task
        plot_spacetime_grid(dat, title, fignum=fignum, ax=ax)
        plt.ylabel('Time')
        plt.xlabel('Site number')
        ax += 1

def probability_plots(data_list, w_list, fignum=1, ax=131):
    row_col = data_list[0].shape
    mixed_board = np.zeros(row_col)
    for i in range(len(w_list)):
        board = data_list[i]
        w = w_list[i]
        title = 'IC {} w = {:0.3f}'.format(i, w)
        if i<3: 
            plot_spacetime_grid(board, title,  
                    ax=ax, fignum=fignum)
            plt.ylabel('Time')
            plt.xlabel('Site number')
            ax += 1
        mixed_board += w * board 
    plot_spacetime_grid(mixed_board, 'mixed', ax=ax-1, fignum=fignum)
    plt.ylabel('Time')
    plt.xlabel('Site number')

def plot_main(params, name=None):
    output_name, R, IC, L, tmax = params
    if name is None:
        name = io.sim_name(R, IC, L, tmax)
    else:
        name = name
    
    output_name, R, IC, L, tmax = params
    
    results   = io.read_results(params, typ='C')
    
    data_list = results['b']
    w_list    = results['w']
    mi_nets   = results['mi'] 
    
    st_mi_measures  = measure_networks(mi_nets, 
                                       tasks=['EV', 'CC', 'Y'], typ='st' )
    avg_mi_measures = measure_networks(mi_nets, 
                                       tasks=['ND', 'CC', 'Y'], typ='avg')
   
    probability_plots(data_list, w_list, fignum=0) 
    avg_nm_plots(avg_mi_measures, fignum=1)
    st_nm_plots(  st_mi_measures, fignum=2)
    
    io.multipage(io.file_name(output_name, 'plots', 'C'+name, '.pdf'))    

def plot_all(R_list, L, lc, tmax, fname):
    fignum = 0
    for R in R_list:
        ic_list = gen_ics(L, lc)
        print('R: ', R)
        for IC in ic_list:
            print(IC)
            fignum += 1
            '''
            state_list = run_sim(R, IC, L, tmax, mode = 'ECA')
            plot_spacetime_grid(state_list, 'ECA R ' + str(R), fignum=fignum, ax=121)
            plt.xlabel('site number')
            plt.ylabel('time')
            #plt.show()
            '''
            state_list = run_sim(R, IC, L, tmax, mode = 'between')
            plot_spacetime_grid(state_list, 'Sweep between ECA R ' + str(R), fignum=fignum, ax=111)
            plt.xlabel('site number')
            plt.ylabel('time')
            #plt.show()

            plt.tight_layout()

        io.multipage(io.file_name('classical', 'plots',  'R'+str(R)+fname, '.pdf'), dpi=100) 
        plt.close('all')

def gen_ics(L, lc):
    for d in range(2**lc): 
        c  = mx.dec_to_bin(d, lc)
        IC = [0]*int((L-lc)/2) + c + [0]*int((L-lc)/2)
        IC = np.array(IC)
        yield IC

if __name__ == '__main__':
    
    unitary_Rs = [ 51,  54,  57,  60, \
                   99, 102, 105, 108, \
                  147, 150, 153, 156, \
                  195, 198, 201, 204  ]
    
    unitary_Rs = [2]
    L = 13
    tmax = 15
    lc = 1
    
    
    plot_all(unitary_Rs, L, lc, tmax,  '_lc'+str(lc))
    
    ''' 
    output_name = 'comp'

    IC_listC = [[('z', 0.5), ('c2i0_1', 0.5)] ]

    R_list = [91]

    L_list = [10]

    tmax_list = [20]

    params_listC = [ (output_name, R, IC, L, tmax) \
            for L    in L_list    \
            for tmax in tmax_list \
            for R    in R_list    \
            for IC   in IC_listC   ]


    for params in params_listC:
        run_mixture(params, force_rewrite = True)
        plot_main(params)

    '''
# END
# ===

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


def run_sim(L, R, tmax, IC, mode = 'sweep'):
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
        
    board_plot(state_list, plt.cm.jet, None)
    plt.xlabel('site number')
    plt.ylabel('time')
    #plt.show()

def plot_all(IC, fname):
    for R in range(256):
        fignum = R
        tmax = 4*int((L-1)/2)
        fig = plt.figure(fignum) 
        fig.add_subplot(121)
        run_sim(L, R, tmax, IC, mode = 'ECA') 
        plt.title('ECA R ' + str(R)) 
        fig.add_subplot(122)
        run_sim(L, R, tmax, IC, mode = 'sweep') 
        plt.title('Sweep ECA R ' + str(R)) 
        plt.tight_layout()
    multipage(file_name('classical', 'plots',  fname, '.pdf')) 


L = 101

fname = 'all_1' 
IC = [1]*L
plot_all(IC, fname) 

fname = 'all_0' 
IC = [0]*L
plot_all(IC, fname) 


if __name__ == '__main__':
    L = 51
    R = 60
    tmax = int((L-1)/2)
    IC = [0]*int((L-1)/2) + [1] + [0]*int((L-1)/2)
    run_sim(L, R, tmax, IC, mode = 'sweep') 



# END
# ===

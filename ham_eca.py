#!/usr/bin/python

# =============================================================================
#
# By Logan Hillberry
# =============================================================================



from itertools import product, cycle
from os.path   import isfile
from cmath     import sqrt
from math import sin, cos, pi
import copy
import time

import numpy    as np
import scipy.sparse as sp
import fio      as io
import matrix   as mx
import states   as ss
import measures as ms

# updating op for ECA
# ===================
def local_update_op(R):
    sxR = 204^R                                 # calculate swap rule s.t. 0 -> I, 1 -> sx
    sxR = mx.dec_to_bin(sxR, 2**3)[::-1]        # reverse so rule element 0 comes first
    op = np.zeros((2**3, 2**3), dtype=complex)
    for Rel_num, sxR_el in enumerate(sxR):      # Rel_num -> sxR_el:  000 -> 1,
                                                # 001 -> 0, etc
        op_sub_el_list = [] 
        for sub_Rel_num, proj_label in enumerate(mx.dec_to_bin(Rel_num, 3)[::-1]):
            if sub_Rel_num == 1:                # sub_rel_num == 1 is center site
                op_sub_el = \
                        ss.pauli[str(sxR_el)].dot(ss.brhos[str(proj_label)]) 
            else:
                op_sub_el = ss.brhos[str(proj_label)]  # leave neighbors alone
            op_sub_el_list.append(op_sub_el)           # make the 3-site update op
        op = op + mx.listkron(op_sub_el_list) 
    return op

def comp_basis_vec(bin_num_as_list):
    bin_num_as_str = map(str, bin_num_as_list)
    vec_list =[]
    for bin_str in bin_num_as_str:
        vec_list.append(ss.bvecs[bin_str])
    return mx.listkron(vec_list)

def general_local_update_op(R, th=pi/2.0):
    R = mx.dec_to_bin(R, 2**3)[::-1]        # reverse so rule element 0 comes first
    T = np.zeros((2**3, 2**3), dtype=complex)
    
    for d, r in  enumerate(R):     
        d_b = mx.dec_to_bin(d, 3)[::-1]
        r_b = copy.copy(d_b)
        
        if r_b[1] == r:
            thr = 0.0
        else:
            thr = th
            r_b[1] = r

        self = comp_basis_vec(d_b)
        flip = comp_basis_vec(r_b)
        
        ket = (cos(thr) * self + sin(thr) * flip)
       
        ket_bra = np.outer(ket, self)
        T = T + ket_bra
    return mx.edit_small_vals(T)


# construct generator for sweep time evolved states
# -------------------------------------------------
def time_evolve(g, R, IC, L, tmax, dt):
    J = -1.0
    Tj = local_update_op(R)
    Isingj = mx.listkron([ss.pauli['0'], ss.pauli['3'], ss.pauli['3']])
    Hj = J*Isingj + g*Tj
    Uj = sp.linalg.expm(-1j*Hj*dt)
    print(R)
    print(Uj)
    state = ss.make_state(L, IC)
    yield state 

    for t in np.arange(tmax):
        for j in range(L):
            js = [(j-1)%L, j, (j+1)%L]
            state = mx.op_on_state(Uj, js, state)
    
        ip = (state.conj().dot(state))
        if ip == 0.0:
            yield state
        
        else: 
            yield  1.0/sqrt(ip) * state


# import/create measurement results and plot them
# -----------------------------------------------
def run_sim(params, force_rewrite = False):
    output_name, R, IC, L, tmax = params
    if not isfile(io.file_name(output_name, 'data', io.sim_name(R, IC, L, tmax), '.res' )) \
        or force_rewrite:
        results = ms.measure_sim(params, time_evolve(R, IC, L, tmax))
        io.write_results(results, params, typ='Q')
    return


if __name__ == "__main__":
    import plotting as pt
    import matplotlib.pyplot as plt
    
    IC = [('d1',1.0)]
    L = 15
    tmax = 100
    g_list = [0.1, 1.0, 2.0] 
    R_list = range(256) 
    
    nz = ss.brhos['1']
    fignum = 1

    for R in R_list:
        print(R)
        for ng, g in enumerate(g_list):
            dt = 1.0/g
            state_gen = time_evolve(g, R, IC, L, tmax, dt)
            nz_board = np.zeros((tmax+1, L))

            for t, state in enumerate(state_gen):
                for j in range(L):
                    print(j,t )
                    rj = mx.rdms(state, [j])
                    nzj = np.trace( rj.dot(nz) ).real
                    nz_board[t, j] = nzj 
                    
            pt.plot_spacetime_grid(nz_board, r'$\langle n_j \rangle g = $'+str(g), 
                    cmap=plt.cm.jet, norm=None, fignum=fignum, ax=131+ng)
        
        plt.suptitle(r'R = ' + str(R))
        
        fignum+=1
    io.multipage(io.file_name('ising_eca', 'plots', 'all_rules_g01_1_2', '.pdf'))    
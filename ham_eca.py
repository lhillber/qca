#!/usr/bin/python

# =============================================================================
#
# By Logan Hillberry
# =============================================================================



from itertools import product, cycle
from os.path   import isfile
from cmath     import sqrt
from math import sin, cos, pi, fabs
from collections import OrderedDict

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

def local_update_op2(R, return_TX=False):
    sxR = 204^R                                 # calculate swap rule s.t. 0 -> I, 1 -> sx
    sxR = mx.dec_to_bin(sxR, 2**3)[::-1]        # reverse so rule element 0 comes first
    
    IXI = mx.listkron([ss.pauli['0'], ss.pauli['1'], ss.pauli['0']])
    
    TX = np.zeros((2**3, 2**3), dtype=complex)
    TI = np.zeros((2**3, 2**3), dtype=complex)
    for m, r in enumerate(sxR):
        m_2 = mx.dec_to_bin(m, 3)[::-1]
        m_ket = comp_basis_vec(m_2)
        m_proj = np.outer(m_ket, m_ket)
        TX = TX + r * IXI.dot(m_proj)
        TI = TI + (1 - r) * m_proj
    
    T = TI + TX
    
    if return_TX == True:
        return TX
   
    else:
        return T


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
def time_evolve(params, tol=1E-10):
    g    = params[ 'g'   ]
    R    = params[ 'R'   ]
    L    = params[ 'L'   ]
    IC   = params[ 'IC'  ]
    dt   = params[ 'dt'  ] 
    tmax = params[ 'tmax']

    J = -0.0
    
    TX = local_update_op2(R, return_TX=True)
    Isingj = mx.listkron([ss.pauli['0'], ss.pauli['3'], ss.pauli['3']])
    Hj = J*Isingj + g*TX
    Uj = sp.linalg.expm(-1j*Hj*dt)
    
    state = ss.make_state(L, IC)
    yield state 

    for t in np.arange(tmax):
        for j in range(L):
            js = [(j-1)%L, j, (j+1)%L]
            state = mx.op_on_state(Uj, js, state)
    
        ip = (state.conj().dot(state)).real
        if fabs(ip - 1.0) < tol:
            yield state
        
        else: 
            print(ip)
            state = 1.0/sqrt(ip) * state
            yield state

# import/create measurement results and plot them
# -----------------------------------------------
def run_sim(params, force_rewrite = False):
    output_name = params['output_name']
    if not isfile(io.file_name(output_name, 'data', io.sim_name(params), '.res' )) \
        or force_rewrite:
        results = ms.measure_sim(params, time_evolve(params))
        io.write_results(results, params, typ='Q')
    return results


if __name__ == "__main__":

    import plotting as pt
    import matplotlib.pyplot as plt
    
    output_name = 'testing/ham_eca'
    IC = [('c1d1',1.0)]
    L = 15 
    tmax = 100
    dt = 0.099
    g_list = [2.0]
    R_list = [ 51,  54,  57,  60,
               99, 102, 105, 108,
              147, 150, 153, 156,
              195, 198, 201, 204 ]

    R_list = [150] 

    for R in R_list:
        for ng, g in enumerate(g_list):
            params = OrderedDict(
                    [
                        ('output_name', output_name), 
                        ('g', g),
                        ('R', R), 
                        ('IC', IC),
                        ('L', L), 
                        ('tmax', tmax), 
                        ('dt', dt)
                    ]
                                )

            run_sim(params)
            pt.plot_main(params)

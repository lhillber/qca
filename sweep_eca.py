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



from itertools import product, cycle
from os.path   import isfile
from cmath     import sqrt

import numpy    as np

import fio      as io
import matrix   as mx
import states   as ss
import measures as ms

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

# construct generator of time evolved states
# ------------------------------------------
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

# import/create measurement results and plot them
# -----------------------------------------------
def run_sim(params, force_rewrite = False):
    output_name, R, IC, L, tmax = params
    if not isfile(io.file_name(output_name, 'data', io.sim_name(R, IC, L, tmax), '.res' )) \
        or force_rewrite:
        results = ms.measure_sim(params, time_evolve(R, IC, L, tmax))
        io.write_results(results, params, typ='Q')
    return




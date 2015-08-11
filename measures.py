#!/usr/bin/python

from math import log

import numpy           as np
import scipy           as sp
import states          as ss
import matrix          as mx
import networkmeasures as nm

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
        MI = 0.0
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
    L = int( log(len(state), 2) )
    MInet = MInetwork(state, ss_entropy)
    MICC = nm.clustering(MInet)
    MIdensity = nm.density(MInet)
    MIdisparity = nm.disparity(MInet)
    MIharmoniclen = nm.harmoniclength(nm.distance(MInet))
    #MIeveccentrality = nm.eigenvectorcentralitynx0(MInet)
    MIeveccentrality = [0.0]*L
    return { 'net' : MInet.tolist(),
             'CC'  : MICC,
             'ND'  : MIdensity,
             'Y'   : MIdisparity,
             'IHL' : MIharmoniclen,
             #'EV'  : list(MIeveccentrality.values()) }
             'EV'  : MIeveccentrality }

# Measure each state of the time evolution
# ----------------------------------------
def measure_sim(params, state_gen): 
    output_name, R, IC, L, tmax = params
    measures = [0]*(tmax+1)
    for t, state in enumerate(state_gen): 
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
    return results

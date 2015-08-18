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
def n_j(state, j, proj = '1'):
    L = int(log(len(state), 2)) 
    proj = ss.brhos[proj]
    op_state = mx.op_on_state(proj, [j], state) 
    return state.conj().dot(op_state).real 

def n_in_j(state, i, j, proj_list = ['1', '1']):
    L = int( log(len(state), 2) ) 
    proj = mx.listkron( [ss.brhos[proj] for proj in proj_list] )
    op_state = mx.op_on_state(proj, [i, j], state) 
    return state.conj().dot(op_state).real 

# von Neumann entropy of reduced density matrix keeping klist from state
# ----------------------------------------------------------------------
def vn_entropy(rdm):
    evals = sp.linalg.eigvalsh(rdm)
    s = -sum(el*log(el,2) if el >= 1e-14 else 0.0  for el in evals)
    return s
        

# entropy of the smaller of all bi-partitions of the lattice
# ----------------------------------------------------------
def entropy_of_cut(state):
    L = int(log(len(state),2))
    js = [ [i for i in range(mx)] if mx <= round(L/2) 
            else np.setdiff1d(np.arange(L), [i for i in range(mx)]).tolist() 
            for mx in range(1,L)]
    return [vn_entropy(mx.rdms(state, ks)) for ks in js ]

# compute mutual information network for state rho given a list of all
# single-site entropies (ss_entropy) ordered 0 to L
# --------------------------------------------------------------------
'''

def MInetwork(state, ss_entropy):
    L = int(log(len(state),2)) 
    MInet = np.zeros((L,L))
    for i in range(L):
        MI = 0.0
        MInet[i][i] = MI
        for j in range(i,L):
            if i != j:
                MI = 0.5 * (ss_entropy[i] + ss_entropy[j] - vn_entropy(state, [i,j]))
                if MI > 1e-14:
                    MInet[i][j] = MI
                    MInet[j][i] = MI
                if MI<= 1e-14:
                    MInet[i][j] = 1e-14
                    MInet[j][i] = 1e-14
    return MInet
'''

# calculate network measures on state state
# -----------------------------------------
def invharmoniclen(net):
    return nm.harmoniclength(nm.distance(net))

def eveccentrality(net):
    return list(nm.eigenvectorcentralitynx0(net).values())



def NMcalc(net, typ = 'avg', tasks=['CC', 'ND']):
    
    NM_dict = { 'avg' : {
                         'CC'  : nm.clustering,
                         'ND'  : nm.density,
                         'Y'   : nm.disparity,
                         'IHL' : invharmoniclen,
                        },
                'st'  : { 
                         'CC'  : nm.localclustering,
                         'Y'   : nm.disparitylattice,
                         'EV'  : eveccentrality,
                         
                         }
              }
    
    measures = {} 
    for key in tasks:
        measures[key] = NM_dict[typ][key](net)
    return measures
    

# Measure each state of the time evolution
# ----------------------------------------
def measure_sim(params, state_gen, tol=1e-14): 
    output_name, R, IC, L, tmax = params
    nz = ss.brhos['1']
    nx = 0.5 * (ss.pauli['0'] + ss.pauli['1'])
    measures = [0]*(tmax+1)
    
    for t, state in enumerate(state_gen): 
        sr_mat = np.zeros((L,L)) 
        nz_mat = np.zeros((L,L))
        nx_mat = np.zeros((L,L))
        mi_mat = np.zeros((L,L))
        
        for j in range(L):
            rj  = mx.rdms(state, [j])
            sj = vn_entropy(rj)
            nzj = np.trace( rj.dot(nz) )
            nxj = np.trace( rj.dot(nx) )
            mij  = 0.0
            sr_mat[j,j] = sj.real
            nz_mat[j,j] = nzj.real
            nx_mat[j,j] = nxj.real
            mi_mat[j,j] = 0.0 
            for k in range(j+1, L):
                rjk  = mx.rdms(state, [j, k])
                rk   = mx.rdms(state, [k])

                sjk  = vn_entropy(rjk)
                sk   = vn_entropy(rk)
                nzjk = np.trace( rjk.dot( np.kron(nz, nz) ) )
                nxjk = np.trace( rjk.dot( np.kron(nx, nx) ) )

                sr_mat[j,k] = sr_mat[k,j] = sjk.real
                nz_mat[j,k] = nz_mat[k,j] = nzjk.real
                nx_mat[j,k] = nx_mat[k,j] = nxjk.real
                mi_mat[j,k] = mi_mat[k,j] = 0.5 * (sj + sk - sjk).real

        # set small elemetns to tol 
        mi_mat = mx.edit_small_vals(mi_mat, tol=tol, replacement=tol)
        s_cuts = mx.edit_small_vals( entropy_of_cut(state), tol=tol )
        sr_mat = mx.edit_small_vals( sr_mat, tol=tol )
        nz_mat = mx.edit_small_vals( nz_mat, tol=tol, replacement=tol )
        nx_mat = mx.edit_small_vals( nx_mat, tol=tol, replacement=tol )
        mi_mat[np.arange(L), np.arange(L)] = 0.0   #set diagonals of mi to 0
        measure = {} 
        measure['ec'] = s_cuts
        measure['sr'] = sr_mat
        measure['nz'] = nz_mat
        measure['nx'] = nx_mat
        measure['mi'] = mi_mat
        measure['t' ] = t
        measures[t] = measure
    
    results = {}
    for key in measure.keys(): 
        results[key] = np.array([measures[t][key] for t in range(tmax)])
    return results




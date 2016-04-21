#!/usr/bin/python3

# ==============================================================
# Functions for matrix manipulations of quantum states/operators
#
# by Logan Hillberry
# ===============================================================


from math import log
from functools import reduce
import numpy as np
import scipy.sparse as sps
import time
import matplotlib.pyplot as plt
from os import environ
import simulation.fio as io

import matplotlib as mpl
font = {'size':12, 'weight' : 'normal'}
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rc('font',**font)
# concatinate two dictionaries (second arg replaces first if keys in common)
# --------------------------------------------------------------------------
def concat_dicts(d1, d2):
    d = d1.copy()
    d.update(d2)
    return d

def listdicts(dictlist):
    return reduce(lambda d1, d2: concat_dicts(d1, d2), dictlist)

# Kroeneker product list of matrices
# ----------------------------------
def listkron(matlist):
    return reduce(lambda A,B: np.kron(A, B), matlist)


# Kroeneker product list of sparse matrices
# -----------------------------------------
def spmatkron(matlist):
    return sps.csc_matrix(reduce(lambda A, B: sps.kron(A,B,'csc'),matlist)) 

# dot product list of matrices
# ----------------------------
def listdot(matlist):
    return reduce(lambda A, B: np.dot(A, B), matlist)

# replace small elements in an array
# ----------------------------------
def edit_small_vals(mat, tol=1e-14, replacement=0.0):
    if not type(mat) is np.ndarray:
        mat = np.asarray(mat)
    mat[mat<=tol] = replacement
    return mat

# sparse matrix tensor product (custom)
# -------------------------------------
def tensor(A, B):
    a_nrows, a_ncols = A.shape
    b_nrows, b_ncols = B.shape
    m_nrows, m_ncols = a_nrows*b_nrows, a_ncols*b_ncols
    
    b = list(zip(B.row, B.col, B.data))
    a = list(zip(A.row, A.col, A.data))

    M = np.zeros((m_nrows, m_ncols))
    for a_row, a_col, a_val in a:
        for b_row, b_col, b_val in b:
            row = a_row * b_nrows + b_row
            col = a_col * a_ncols + b_col
            M[row, col] = a_val * b_val
    return M

# Hermitian conjugate
# -------------------
def dagger (mat):
    return mat.conj().transpose()


# apply k-qubit op to a list of k sites (js) of state-vector state. ds
# is a list of local dimensions for each site of state, assumed to be a listtice
# of qubits if not provided.
# -------------------------------------------------------------------------------
def op_on_state(meso_op, js, state, ds = None):
    if ds is None:
        L = int( log(len(state), 2) )
        ds = [2]*L
    else:
        L = len(ds)

    dn = np.prod(np.array(ds).take(js))
    dL = np.prod(ds)
    rest = np.setdiff1d(np.arange(L), js)
    ordering = list(rest) + list(js)

    state = state.reshape(ds).transpose(ordering)\
            .reshape(dL/dn, dn).dot(meso_op).reshape(ds)\
            .transpose(np.argsort(ordering)).reshape(dL)
    return state

# partial trace of a state vector, js are the site indicies kept
# --------------------------------------------------------------
def rdms(state, js, ds=None):
    js = np.array(js)
    if ds is None:
        L = int( log(len(state), 2) )
        ds = [2]*L
    else:
        L = len(ds)

    rest = np.setdiff1d(np.arange(L), js)
    ordering = np.concatenate((js, rest))
    dL = np.prod(ds)
    djs = np.prod(np.array(ds).take(js))
    drest = np.prod(np.array(ds).take(rest))

    block = state.reshape(ds).transpose(ordering).reshape(djs, drest)

    RDM = np.zeros((djs, djs), dtype=complex)
    tot = complex(0,0)
    for i in range(djs):
        for j in range(djs):
            Rij = np.inner(block[i,:], np.conj(block[j,:]))
            RDM[i, j] = Rij
    return RDM

# partial trace of a  density matrix
# ----------------------------------
def rdmr(rho, klist):
    L = int(log(len(rho), 2))
    d = 2*L
    n = len(klist)

    kin = list(klist)
    kout = [k+L for k in kin] 

    klist = kin + kout

    ordering = klist+list(rest)

    block = rho.reshape(([2]*(d)))
    block = block.transpose(ordering)
    block = block.reshape(2**n, 2**(d-n))

    RDM = np.zeros((2**n,2**n), dtype=complex)
    tot = 0+0j

    for i in range(2**n - 1):
        Rii = sum(np.multiply(block[i,:], np.conj(block[i,:])))
        tot = tot+Rii
        RDM[i][i] = Rii

        for j in range(i, 2**n):
            if i != j:
                Rij = np.inner(block[i,:], np.conj(block[j,:]))
                RDM[i][j] = Rij
                RDM[j][i] = np.conj(Rij)
    RDM[2**n-1,2**n-1] = 1+0j - tot
    return RDM


# convert base-10 to base-2
# -------------------------
def dec_to_bin(n, count):
     return [(n >> y) & 1 for y in range(count-1, -1, -1)]

# arithmatic index generation
# NOTE: Generalize local dim by converting 
# to base-localdim instead of base-2
# ---------------------------------------
def inds_gen2(js, L):
    d = len(js)
    keep = js
    rest = np.setdiff1d(np.arange(L), keep)
    env_vals = [2**x for x in rest][::-1]
    sys_vals = [2**x for x in keep][::-1]
    for env_count in np.arange(2**len(env_vals)):
        env = np.array(dec_to_bin(env_count, len(env_vals))).dot(np.array(env_vals))
        sys_list = []
        for sys_count in np.arange(2**len(sys_vals)):
            sys = np.array(dec_to_bin(sys_count, len(sys_vals))).dot(np.array(sys_vals))
            sys = env + sys 
            sys_list.append(sys) 
        yield sys_list 

# enumerate indicies of environment
# ---------------------------------
def spread_env(js, env_count):
    env_ind = env_count
    for j in js:
        msb = env_ind >> j << j
        lsb = env_ind ^ msb
        env_ind = msb << 1 ^ lsb
    return env_ind

# enumerate indicies of system
# ----------------------------
def spread_js(js):
    for sys_count in range(2**len(js)):
        sys_ind = 0
        for j in js:
            level = sys_count % 2
            sys_ind += level * 2**j
            sys_count = sys_count >> 1
        yield sys_ind

# generate indicies with bit-wise ops
# -----------------------------------
def inds_gen(js, L):
    js = [(L-1)-j for j in js]
    for env_count in range(2**(L-len(js))):
        yield np.array([spread_env(js, env_count) + sys for sys in spread_js(js)])

# older version (V1)
def op_on_state2(meso_op, js, state):
    L = int( log(len(state), 2) )
    js = ((L - 1) - np.array(js))[::-1]
    js = list(js)
    new_state = np.array([0.0]*(2**L), dtype=complex)
    for inds in inds_gen2(js, L):
        new_state[inds] = meso_op.dot(state.take(inds))
    return new_state


# mememory intensive method (oldest version, V0)
# ----------------------------------------------
def big_mat(local_op_list, js, state):
    L = int( log(len(state), 2) )
    I_list = [np.eye(2.0, dtype=complex)]*L
    for j, local_op in zip(js, local_op_list): 
        I_list[j] = local_op
    big_op = listkron(I_list)
    return big_op.dot(state)

# compare timing of various methods
# ---------------------------------

def comp_plot():
    from math import sqrt
    j=0
    LMAX = 28
    LMax = 18
    Lmax = 13
    n_flips_list = [3]

    for color, n_flips in zip(['r', 'g', 'b', 'k', 'c','w'], n_flips_list):
        js = range(j, j+n_flips)
        Llist = range(j + n_flips, LMAX)
        lop = ss.ops['X']
        lops = [lop]*n_flips
        mop = listkron(lops)

        ta_list = np.array([])
        tb_list = np.array([])
        tc_list = np.array([])
        td_list = np.array([])
        te_list = np.array([])
        tf_list = np.array([])
        mem_list = np.array([])

        for L in Llist:
            print('L=',L)
            init_state = ss.make_state(L, 'f0')
            nbyts = init_state.nbytes
            print(nbyts/1e9)
            mem_list = np.append(mem_list, nbyts)

            ta_list = np.append(ta_list, time.time())
            op_on_state(mop, js, init_state)
            tb_list = np.append(tb_list, time.time())

            if L < LMax:
                tc_list = np.append(tc_list, time.time())
                op_on_state2(mop, js, init_state)
                td_list = np.append(td_list, time.time())
                #print('V2=V1:', np.array_equal(state2, state1))

            if L < Lmax: 
                te_list = np.append(te_list, time.time())
                big_mat(lops, js, init_state)
                tf_list = np.append(tf_list, time.time())
                #print('V2=V0:', np.array_equal(state2, state0))
                #print('V1=V0:', np.array_equal(state1, state0))
                #print()

            del init_state

        fig = plt.figure(1, (7, 2.5))
        t_ax = fig.add_subplot(111)
        #m_ax = fig.add_subplot(212, sharex=t_ax)
        t_ax.plot(Llist, tb_list - ta_list, '-o', 
                 color = color, label='V2')
        t_ax.plot(range(j+n_flips, LMax), td_list - tc_list, '-s', 
                 color = color, label='V1')
        t_ax.plot(range(j+n_flips, Lmax), tf_list - te_list, '-^', 
                 color = color, label='V0')

        #m_ax.plot(Llist, mem_list)
    t_ax.set_yscale('log')
    t_ax.set_xlabel('number of sites [L]')
    t_ax.set_ylabel('computation time [s]')
    t_ax.set_title('Application of 3-site operator')
    t_ax.grid('on')
    t_ax.legend(loc = 'upper left')
    t_ax.set_ylim([0.5e-4, 5])
    t_ax.set_xlim([2, 26])

    '''
    m_ax.set_yscale('log')
    m_ax.set_xlabel('number of sites [L]')
    m_ax.set_ylabel('memory required [bytes]')
    m_ax.set_title('Size of state vector')
    m_ax.grid('on')
    '''
    plt.tight_layout()
    plots_fname = io.base_name('timing', 'plots')+'3-site_op_timing2.pdf'
    io.multipage(plots_fname)


def rdm_plot():
    Llist = range(4, 20)
    for L in Llist:
        ta_list = np.array([])
        tb_list = np.array([])
        mem_list = np.array([])
        trace_list = range(1, int(L/2))
        for n_trace in trace_list:
            js = range(n_trace)
            print('L=',L)
            init_state = ss.make_state(L, 'G')

            ta_list = np.append(ta_list, time.time())
            rdm = rdms(init_state, js)
            tb_list = np.append(tb_list, time.time())
            nbyts = rdm.nbytes
            print(nbyts/1e9)
            mem_list = np.append(mem_list, nbyts)
            del rdm
            del init_state

        fig = plt.figure(1)
        t_ax = fig.add_subplot(211)
        m_ax = fig.add_subplot(212, sharex=t_ax)
        t_ax.plot(trace_list, tb_list - ta_list, '-o')
        m_ax.plot(trace_list, mem_list)

    t_ax.set_yscale('log')
    t_ax.set_xlabel('size of cut [n]')
    t_ax.set_ylabel('computation time [s]')
    t_ax.set_title('Trace out L-n qubits')
    t_ax.grid('on')
    #t_ax.legend(loc = 'upper left')

    m_ax.set_yscale('log')
    m_ax.set_xlabel('size of cut [n]')
    m_ax.set_ylabel('memory required [bytes]')
    m_ax.set_title('Size of rdm')
    m_ax.grid('on')
    plt.tight_layout()
    #plt.show()
    plt.savefig(environ['HOME'] +
            '/documents/research/cellular_automata/qeca/qca_notebook/notebook_figs/'+'trace_timing_numba'+'.pdf', 
            format='pdf', dpi=300, bbox_inches='tight')



if __name__ == '__main__':

    import simulation.states as ss
    import simulation.measures as ms
    L = 7
    IC = 'f0'

    js = [0,3,2]
    op = listkron( [ss.ops['X']]*(len(js)-1) + [ss.ops['H']] ) 

    print()
    print('op = XXH,', 'js = ', str(js)+',', 'IC = ', IC)
    print()

    init_state3 = ss.make_state(L, IC)
    init_rj = [rdms(init_state3, [j]) for j in range(L)]
    init_Z_exp = [round(np.trace(r.dot(ss.ops['Z'])).real) for r in init_rj]
    print('initl Z exp vals:', init_Z_exp) 

    final_state = op_on_state(op, js, init_state3)

    final_rj = [rdms(final_state, [j]) for j in range(L)]
    final_Z_exp = [round(np.trace(r.dot(ss.ops['Z'])).real) for r in final_rj]
    print('final Z exp vals:', final_Z_exp) 

    #rdm_plot()
    comp_plot()

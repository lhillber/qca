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


# Kroeneker product list of matrices
# ----------------------------------
def listkron (matlist):
    return reduce(lambda A,B: np.kron(A,B), matlist)


# Kroeneker product list of sparse matrices
# -----------------------------------------
def spmatkron (matlist):
    return sps.csc_matrix(reduce(lambda A,B: sps.kron(A,B,'csc'),matlist))

# dot product list of matrices
# ----------------------------
def listdot (matlist):
    return reduce(lambda A,B: np.dot(A,B), matlist)


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

# partial trace of a state vector
# -------------------------------
def rdms(state, klist):
    L = int(log(len(state), 2))
    n = len(klist)
    rest = np.setdiff1d(np.arange(L), klist)
    ordering = []
    ordering = list(klist)+list(rest)
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

# partial trace of a  density matrix
# ----------------------------------
def rdmr(rho, klist):
    L = int(log(len(rho), 2))
    d = 2*L    
    n = len(klist)
    kin = list(klist)
    kout = [k+L for k in kin]
    klist = kin+kout
    rest = np.setdiff1d(np.arange(d), klist)
    ordering = klist+list(rest)
    block = rho.reshape(([2]*(d)))
    block = block.transpose(ordering)
    block = block.reshape((2**(n), 2**((d-n)))) 
    
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
    keep = np.arange(j, j-d, -1)
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
    for j in sorted(js):
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

# indicies generated by bit-wise operation
# ----------------------------------------
def op_on_state(meso_op, js, state):
    L = int( log(len(state), 2) )
    d = int( log(len(meso_op), 2) )
    new_state = np.array([0.0]*(2**L), dtype=complex)
    for inds in inds_gen(js, L):
        new_state[inds] = meso_op.dot(state[inds])
    return new_state

# indicies generated by arithmatic
# --------------------------------
def meso_op_on_global_state2(meso_op, j, state):
    L = int( log(len(state), 2) )
    d = int( log(len(meso_op), 2) )
    js = np.arange(j,j+d) 
    new_state = np.array([0.0]*(2**L), dtype=complex)
    for inds in inds_gen2(js, L):
        new_state[inds] = meso_op.dot(state[inds])
    return new_state

# mememory intensive method
# ------------------------
def big_mat(local_op_list, js, state):
    L = int( log(len(state), 2) )
    I_list = [np.eye(2.0, dtype=complex)]*L
    for j, local_op in zip(js, local_op_list): 
        I_list[j] = local_op
    big_op = util.matkron(I_list)
    return big_op.dot(state)

# compare timing of various methods
# ---------------------------------
def comp_plot():
    dead = np.array([1.0 + 0j, 0.0 + 0.0j])
    alive = np.array([0.0 + 0.0j + 1.0j+0.0])
    for color, n_flips in zip(['r', 'g', 'b', 'k', 'c'], range(1, 10, 2)):
        Llist = range(n_flips+3, 21)
        lop = np.array([[0.,1.],[1.,0.]])
        lops = [lop]*n_flips
        
        ta_list = np.array([])
        tb_list = np.array([])
        tc_list = np.array([])
        tm1_list = np.array([])
        tm2_list = np.array([])
        for L in Llist:
            j = 1
            init_state = util.matkron([dead]*L)
            
            ta_list = np.append(ta_list, time.time())
            mop = util.matkron(lops)
            state1 = op_on_state(mop, j, init_state )
            tb_list = np.append(tb_list, time.time())
            state2 = meso_op_on_global_state(mop, j, init_state )
            tc_list = np.append(tc_list, time.time()) 
            if L<1:
                tm1_list = np.append(tm1_list, time.time())
                state3 = big_mat(lops, j, init_state)
                tm2_list = np.append(tm2_list, time.time())
            print( np.array_equal(state2.transpose(), state1.transpose()))
           #     print( np.array_equal(state2.transpose(), state1.transpose()))
                 
            #print('G', state1.transpose())
            #print('M', state1.transpose())
           # print( np.array_equal(state2.transpose(), state1.transpose()))
        plt.plot(Llist, tb_list - ta_list, '-o', 
                 color = color, label='bit:' + str(n_flips))
        plt.plot(Llist, tc_list - tb_list, '-^', 
                 color = color, label='calc:' + str(n_flips))

        #plt.plot(np.arange(n_flips+3, 14), tm2_list - tm1_list,'-s', 
        #         color = color, label='global op: ' + str(n_flips))
    plt.yscale('log')
    plt.xlabel('number of sites [L]')
    plt.ylabel('computation time [s]')
    
    plt.title('bit flips')
    plt.legend(loc = 'upper left')
    plt.show()


if __name__ == '__main__':
    comp_plot()

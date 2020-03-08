from math import log
from cmath import sqrt, sin, cos, exp, pi
from functools import reduce
import numpy as np

# from scipy.sparse import sparse as sps
from scipy.linalg import eigvals
from numba import njit, jit

ops = {
    "H": 1.0 / sqrt(2.0) * (np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex)),
    "I": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex),
    "O": np.array([[0.0, 0.0], [0.0, 0.0]], dtype=complex),
    "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    "Y": np.array([[0.0, (0.0 - 1j)], [1j, 0.0]], dtype=complex),
    "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
    "S": np.array([[1.0, 0.0], [0.0, 1j]], dtype=complex),
    "T": np.array([[1.0, 0.0], [0.0, exp(1j * pi / 4.0)]], dtype=complex),
    "0": np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex),
    "1": np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex),
    "D": 1.0 / sqrt(2) * np.array([[1.0, -1j], [-1j, 1.0]], dtype=complex),
}

# works but doesn't help mem leak


def op_on_state_inplace(op, js, state, ds=None):
    if ds is None:
        L = int(np.log2(len(state)))
        ds = [2] * L
    else:
        L = len(ds)
    dn = np.prod(np.array(ds).take(js))
    dL = np.prod(ds)
    rest = np.setdiff1d(np.arange(L), js)
    ordering = list(rest) + list(js)
    state = state.reshape(ds).transpose(ordering).reshape(np.int(dL / dn), dn)
    state = (
        state.dot(op, out=np.ascontiguousarray(state))
        .reshape(ds)
        .transpose(np.argsort(ordering))
        .reshape(dL)
    )
    return state


def op_on_state(op, js, state, ds=None):
    if ds is None:
        L = int(np.log2(len(state)))
        ds = [2] * L
    else:
        L = len(ds)
    dn = np.prod(np.array(ds).take(js))
    dL = np.prod(ds)
    rest = np.setdiff1d(np.arange(L), js)
    ordering = list(rest) + list(js)
    state = (
        state.reshape(ds)
        .transpose(ordering)
        .reshape(np.int(dL / dn), dn)
        .dot(op)
        .reshape(ds)
        .transpose(np.argsort(ordering))
        .reshape(dL)
    )
    return state


def op_on_state_2(local_op_list, js, state):
    L = np.int(log(len(state), 2))
    I_list = [np.eye(2.0, dtype=complex)] * L
    for j, local_op in zip(js, local_op_list):
        I_list[j] = local_op

    big_op = listkron(I_list)
    return big_op.dot(state)


# @jit
def rdms(state, js, ds=None, out=None):
    js = np.array(js)
    if ds is None:
        L = int(np.log2(len(state)))
        ds = tuple([2] * L)
    else:
        L = len(ds)
    rest = np.setdiff1d(np.arange(L), js)
    ordering = tuple(np.concatenate((js, rest)))
    djs = np.prod(np.array(ds).take(js))
    drest = np.prod(np.array(ds).take(rest))
    block = state.reshape(ds).transpose(ordering).reshape(djs, drest)
    return rdms_njit(block, ds, ordering, djs, drest, out=out)


def rdms2(state, js):
    L = log(len(state), 2)
    mask = np.ones(L, dtype=np.bool)
    mask[js] = False
    trce = list(np.array(list(range(L)))[mask])
    if(js is None):
        mask = np.ones((L), dtype=np.bool)
        mask[trce] = False
        js = list(np.array(list(range(L)))[mask])
    state = np.reshape(state, [2] * L)
    rho = np.tensordot(state, np.conj(state), (trce, trce))
    return rho


# @njit
def rdms_njit(block, ds, ordering, djs, drest, out=None):
    if out is None:
        RDM = np.zeros((djs, djs), dtype=np.complex128)
    else:
        RDM = out
    for i in range(djs):
        for j in range(i, djs):
            Rij = np.dot(block[i, :], np.conj(block[j, :]))
            RDM[(i, j)] = Rij
            RDM[(j, i)] = np.conjugate(Rij)

    return RDM


def rdmr(rho, js):
    print("WARNING: this function is NOT verified")
    L = np.int(np.log(len(rho), 2))
    d = 2 * L
    n = len(js)
    js = list(js)
    rest = np.setdiff1d(np.arange(L), js)
    ordering = js + list(rest)
    orderingout = [o + L for o in ordering]
    ordering = ordering + orderingout
    block = (
        rho.reshape([2] * d)
        .transpose(ordering)
        .reshape(2 ** (d - 2 * n), 2 ** n, 2 ** n)
    )
    print(block)
    RDM = np.trace(block)
    return RDM / np.trace(RDM)


# @jit
def traceout_outer_njit(rho, dim, mask):
    rho_out = np.zeros((dim, dim), dtype=rho.dtype)
    for kk in range(dim):
        for jj in range(dim):
            rho_out[(kk, jj)] = rho[(kk, jj, mask)].sum()

    return rho_out


# @jit
def traceout_first(rho, ld=2):
    L = int(np.log2(len(rho)))
    dim = ld ** (L - 1)
    rho = np.reshape(
        np.transpose(np.reshape(rho, (ld, dim, ld, dim)), (1, 3, 0, 2)),
        (dim, dim, ld ** 2),
    )
    mask = np.cumsum((ld + 1) * (np.ones(ld, dtype=np.int))) - (ld + 1)
    return traceout_outer_njit(rho, dim, mask)


# @jit
def traceout_last(rho, ld=2):
    L = int(np.log2(len(rho)))
    dim = ld ** (L - 1)
    rho = np.reshape(
        np.transpose(np.reshape(rho, (dim, ld, dim, ld)), (0, 2, 1, 3)),
        (dim, dim, ld ** 2),
    )
    mask = np.cumsum((ld + 1) * (np.ones(ld, dtype=np.int))) - (ld + 1)
    return traceout_outer_njit(rho, dim, mask)


def make_U2(V):
    if type(V) == np.ndarray:
        return V
    Vs_angs = V.split("_")
    if len(Vs_angs) == 2:
        Vs, angs = Vs_angs
        angs = angs.split("-")
    else:
        if len(Vs_angs) == 1:
            Vs, angs = Vs_angs[0], []
    ang_inds = [i for i, v in enumerate(Vs) if v in ("P", "R", "p")]
    if len(angs) != len(ang_inds):
        raise ValueError(
            "improper V configuration {}:                need one angle for every P, R, and p".format(
                V
            )
        )
    ang_id = 0
    Vmat = np.eye(2, dtype=complex)
    for v in Vs:
        if v == "P":
            ang = angs[ang_id]
            ang_in_rad = string_deg2float_rad(ang)
            Pmat = make_Pmat(ang_in_rad)
            Vmat = Vmat.dot(Pmat)
            ang_id += 1
        elif v == "R":
            ang = angs[ang_id]
            ang_in_rad = string_deg2float_rad(ang)
            Rmat = make_Rmat(ang_in_rad)
            Vmat = Vmat.dot(Rmat)
            ang_id += 1
        elif v == "p":
            ang = angs[ang_id]
            ang_in_rad = string_deg2float_rad(ang)
            global_phase = make_global_phase(ang_in_rad)
            Vmat = global_phase * Vmat
            ang_id += 1
        else:
            try:
                Vmat = Vmat.dot(ops[v])
            except:
                raise ValueError("string op {} not understood".format(v))

    return Vmat


def make_Pmat(ang_in_rad):
    return np.array([[1.0, 0.0], [0.0, exp(1j * ang_in_rad)]], dtype=complex)


def make_Rmat(ang_in_rad):
    return np.array(
        [
            [cos(ang_in_rad / 2), -sin(ang_in_rad / 2)],
            [sin(ang_in_rad / 2), cos(ang_in_rad / 2)],
        ],
        dtype=complex,
    )


def make_global_phase(ang_in_rad):
    return exp(1j * ang_in_rad)


def string_deg2float_rad(string_deg):
    float_rad = eval(string_deg) * pi / 180.0
    return float_rad


def dagger(rho):
    return np.transpose(np.conj(rho))


def commute(A, B):
    return A.dot(B) - B.dot(A)


def isU(U):
    m, n = U.shape
    Ud = np.conjugate(np.transpose(U))
    UdU = np.dot(Ud, U)
    UUd = np.dot(U, Ud)
    I = np.eye(n, dtype=complex)
    if np.allclose(UdU, I):
        if np.allclose(UUd, I):
            return True
    else:
        return False


def isherm(rho):
    if np.allclose(rho, dagger(rho)):
        return True
    return False


def issymm(mat):
    matT = mat.T
    if np.allclose(mat, matT):
        return True
    return False


def istrace_one(mat):
    tr = np.trace(mat)
    if tr == 1.0:
        return True
    return False


def ispos(mat):
    evals = eigvals(mat)
    if np.all(evals >= 0.0):
        return True
    return False


def listkron(matlist):
    return reduce(lambda A, B: np.kron(A, B), matlist)


# def spmatkron(matlist):
#    return sps.csc_matrix(reduce(lambda A, B: sps.kron(A, B, 'csc'), matlist))


def listdot(matlist):
    return reduce(lambda A, B: np.dot(A, B), matlist)


def concat_dicts(d1, d2):
    d = d1.copy()
    d.update(d2)
    return d


def listdicts(dictlist):
    return reduce(lambda d1, d2: concat_dicts(d1, d2), dictlist)


def dec_to_bin(n, count):
    return [n >> y & 1 for y in range(count - 1, -1, -1)]


def bin_to_dec(n):
    return int("".join(list(map(str, n))), 2)


def tensor(A, B):
    a_nrows, a_ncols = A.shape
    b_nrows, b_ncols = B.shape
    m_nrows, m_ncols = a_nrows * b_nrows, a_ncols * b_ncols
    b = list(zip(B.row, B.col, B.data))
    a = list(zip(A.row, A.col, A.data))
    M = np.zeros((m_nrows, m_ncols))
    for a_row, a_col, a_val in a:
        for b_row, b_col, b_val in b:
            row = a_row * b_nrows + b_row
            col = a_col * a_ncols + b_col
            M[(row, col)] = a_val * b_val

    return M


if __name__ == "__main__":
    import states as ss

    L = 7
    IC = "f0-3-4_t90-p90"
    js = [0, 2, 3]
    op = listkron([ops["X"]] * (len(js) - 1) + [ops["H"]])
    print()
    print("op = XXH,", "js = ", str(js) + ", ", "IC = ", IC)
    print()
    init_state3 = ss.make_state(L, IC)
    init_rj = [rdms(init_state3, [j]) for j in range(L)]
    init_Z_exp = [np.trace(r.dot(ops["Z"]).real) for r in init_rj]
    init_Y_exp = [np.trace(r.dot(ops["Y"]).real) for r in init_rj]
    print("initial Z exp vals:", init_Z_exp)
    print("initial Y exp vals:", init_Y_exp)
    final_state = op_on_state(op, js, init_state3)
    final_rj = [rdms(final_state, [j]) for j in range(L)]
    final_Z_exp = [np.trace(r.dot(ops["Z"])).real for r in final_rj]
    final_Y_exp = [np.trace(r.dot(ops["Y"])).real for r in final_rj]
    print("final Z exp vals:", final_Z_exp)
    print("final Y exp vals:", final_Y_exp)
# okay decompiling matrix.cpython-37.pyc

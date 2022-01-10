from math import log
from cmath import sqrt, sin, cos, exp, pi
from functools import reduce
import numpy as np

# from scipy.sparse import sparse as sps
from scipy.linalg import eigvals
from numba import njit, jit
from warnings import simplefilter
from numba.core.errors import NumbaPerformanceWarning
simplefilter('ignore', category=NumbaPerformanceWarning)


def haar(n):
    z = np.random.randn(n, n) + 1j*np.random.randn(n, n) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    Lambda = d / np.abs(d)
    return np.multiply(q, Lambda, q)

ops = {
    "H": 1.0 / sqrt(2.0) * (np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex)),
    "h": np.array([[0.85355339+0.14644661j, 0.35355339-0.35355339j],
                [0.35355339-0.35355339j, 0.14644661+0.85355339j]], dtype=complex),
    "I": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex),
    "O": np.array([[0.0, 0.0], [0.0, 0.0]], dtype=complex),
    "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    "Y": np.array([[0.0, -1j], [1j, 0.0]], dtype=complex),
    "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
    "S": np.array([[1.0, 0.0], [0.0, 1j]], dtype=complex),
    "T": np.array([[1.0, 0.0], [0.0, exp(1j * pi / 4.0)]], dtype=complex),
    "0": np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex),
    "1": np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex),
    "D": 1.0 / sqrt(2) * np.array([[1.0, -1j], [-1j, 1.0]], dtype=complex),
    "d": np.array([[ 9.23879533e-01, -3.82683432e-01j],
                  [ -3.82683432e-01j,  9.23879533e-01]], dtype=complex),
}


def op_at(opstrs, js, L, base=None, BC=None):
    if type(js) == int:
        js = [js]
    if base is None:
        base = ["I"]*L
    for j, opstr in zip(js, opstrs):
        if BC == "0":
            j = j%L
        base[j] = opstr
    return listkron([ops[opstr] for opstr in base])

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

def random_psi(L):
    "Haar-random state of L qubits"
    Re, Im = np.random.randn(2, 2**L)
    psi = Re + 1j * Im
    return psi / np.sqrt(psi @ np.conj(psi))


def ptranspose(state, js, outshape):
    """ Partial transpose of a state vector or density matrix.

    Parameters:
        state (array): 1D interpreted as state vector, 2D interpreted as density matrix

        js (array-like): Qubit indicies to be transposed in front of remaining indicies

        outshape (tuple): Shape of the returned array

    Returns:
        array: state wth js transposed to be the first indicies

    """
    js = np.array(js)
    D = len(state.shape)
    L = int(np.log2(len(state)))
    ds = [2] * (L * D)
    mask = np.ones(L, dtype=bool)
    mask[js] = False
    rest = np.arange(L)[mask]
    if D == 1:  # state vector
        ordering = list(js) + list(rest)
    elif D == 2:  # density matrix
        ordering = list(js) + list(js+L) + list(rest) + list(rest + L)
    return state.reshape(ds).transpose(ordering).reshape(outshape)


@njit
def _rdms_njit(state):
    """ Partial trace helper for numba implementation"""
    djs, drest = state.shape
    RDM = np.zeros((djs, djs), dtype=np.complex128)
    for i in range(djs):
        for j in range(i, djs):
            Rij = 0
            for k in range(drest):
                Rij += state[i, k] * np.conj(state[j, k])
            RDM[i, j] = Rij
            if i != j:
                RDM[j, i] = np.conj(Rij)
    return RDM



def rdms_numba(state, js):
    """ Partial trace (numba implementation)

     Parameters:
        state (array): 1D interpreted as state vector, 2D interpreted as density matrix

        js (array-like): Qubit indicies to keep

    Returns:
        array: state wth js kept and the remaining qubits traced-out."""
    L = int(np.log2(len(state)))
    js = np.array(js)
    dims =np.array([2]*L)
    mask = np.ones(L, dtype=bool)
    mask[js] = False
    rest = np.arange(L)[mask]
    ordering = np.concatenate((js, rest))
    djs = np.prod(dims.take(js))
    drest = np.prod(dims.take(rest))
    state = ptranspose(state, js, outshape=(djs, drest))
    return _rdms_njit(state)


def rdms_tensordot(state, js):
    """ Partial trace (tensordot implementation)

     Parameters:
        state (array): 1D interpreted as state vector, 2D interpreted as density matrix
        js (array-like): Qubit indicies to keep

    Returns:
        array: state wth js kept and the remaining qubits traced-out."""
    L = int(np.log2(len(state)))
    js = np.array(js)
    dims = np.array([2]*L)
    mask = np.ones(L, dtype=bool)
    mask[js] = False
    rest = np.arange(L)[mask]
    state = state.reshape(dims)
    djs = np.prod(dims.take(js))
    return np.tensordot(state, state.conj(), (rest, rest)).reshape((djs, djs))


def rdms_einsum(state, js):
    """ Partial trace (einsum implementation)

     Parameters:
        state (array): 1D interpreted as state vector, 2D interpreted as density matrix

        js (array-like): Qubit indicies to keep

    Returns:
        array: state wth js kept and the remaining qubits traced-out."""
    L = int(np.log2(len(state)))
    js = np.array(js)
    dims = np.array([2]*L)
    djs = np.prod(dims.take(js))
    idx1 = [i for i in range(L)]
    idx2 = [L+i if i in js else i for i in np.arange(L)]
    state = state.reshape(dims)
    rho = np.einsum(state, idx1, state.conj(), idx2, optimize=True)
    return rho.reshape(djs, djs)


def rdmr(rho, js):
    js = np.array(js)
    L = np.int(np.log2(len(rho)))
    rest = np.setdiff1d(np.arange(L), js)
    shape = [2] * 2 * L
    d1 = 2**len(js)
    d2 = 2**len(rest)
    perm = list(js) + list(js+L) + list(rest) + list(rest + L)
    block = ptranspose(rho, js, outshape=(d1, d1, d2**2))
    mask = np.cumsum((d2 + 1) * np.ones(d2), dtype=np.int32) - (d2 + 1)
    rho = np.sum(block[:, :, mask], axis=2)
    return rho

def rdm(state, js):
    if len(state.shape) == 1:
        if len(js) <= 3:
            return rdms_numba(state, js)
        else:
            return rdms_tensordot(state, js)
    elif len(state.shape) == 2:
        return rdmr(state, js)


@njit
def traceout_outer_njit(rho, dim, mask):
    rho_out = np.zeros((dim, dim), dtype=rho.dtype)
    for kk in range(dim):
        for jj in range(dim):
            Rij = 0
            for ll in mask:
                Rij += rho[kk, jj, ll]
            rho_out[kk, jj] = Rij
    return rho_out


def traceout_first(rho, ld=2):
    L = int(np.log2(len(rho)))
    dim = ld ** (L - 1)
    rho = np.reshape(
        np.transpose(np.reshape(rho, (ld, dim, ld, dim)), (1, 3, 0, 2)),
        (dim, dim, ld ** 2),
    )
    mask = np.cumsum((ld + 1) * (np.ones(ld, dtype=np.int))) - (ld + 1)
    return traceout_outer_njit(rho, dim, mask)


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
    elif len(Vs_angs) == 1:
            Vs, angs = Vs_angs[0], []
    else:
        raise ValueError(
            "improper V configuration {}: need one angle for every P, R, and p".format(
                V
            )
        )

    ang_inds = [i for i, v in enumerate(Vs) if v in ("P", "R", "p")]
    if len(angs) != len(ang_inds):
        raise ValueError(
            "improper V configuration {}: need one angle for every P, R, and p".format(
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
            except KeyError:
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
    Ud = dagger(U)
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
    IC = "f0-3"
    js = [2, 0, 3]
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

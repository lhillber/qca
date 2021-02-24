import numpy as np
from numpy import log2, sqrt
from numpy.linalg import matrix_power, multi_dot
from scipy.linalg import fractional_matrix_power, logm
import scipy as sp
import scipy.linalg
import matrix as mx
from math import pi
from copy import copy
from joblib import Parallel, delayed

def spectrum(rho):
    spec = sp.linalg.eigvalsh(rho).real
    return spec


def vn_entropy_from_spectrum(spec, tol=1e-14):
    return np.real(-np.sum(((el * log2(el) if el >= tol else 0.0) for el in spec)))


def vn_entropy(rho, tol=1e-14):
    spec = spectrum(rho)
    return vn_entropy_from_spectrum(spec)


def renyi_entropy_from_spectrum(spec, order=2, tol=1e-14):
    if order == 0:
        return sum(spec>tol)
    elif order == 1:
        return vn_entropy_from_spectrum(spec, tol=tol)
    else:
        denom = 1.0 - order
        return np.real(log2(np.sum(spec**order)) / denom)


def renyi_entropy(rho, order=2, tol=1e-14):
    if order == 0:
        return sum(spectrum(rho)>tol)
    elif order == 1:
        return vn_entropy(rho, tol=tol)
    else:
        denom = 1.0 - order
        return np.real(log2(np.trace(matrix_power(rho, order))) / denom)


def expectation(state, A):
    if len(state.shape) == 2:
        return np.real(np.trace(state.dot(A)))
    elif len(state.shape) == 1:
        return np.real(np.conjugate(state).dot(A.dot(state)))


def network_density(mat):
    ll = len(mat)
    lsq = ll * (ll - 1)
    return sum(sum(mat)) / lsq


def network_clustering(mat):
    matsq = matrix_power(mat, 2)
    matcube = matrix_power(mat, 3)
    for i in range(len(matsq)):
        matsq[i][i] = 0

    denominator = sum(sum(matsq))
    numerator = np.trace(matcube)
    if numerator == 0.0:
        return 0.0
    return numerator / denominator


def network_disparity(mat, eps=1e-17j):
    numerator = np.sum(mat ** 2, axis=1)
    denominator = (np.sum(mat, axis=1)) ** 2
    return (1 / len(mat) * sum(numerator / (denominator + eps))).real


def network_pathlength(mat, tol=1e-10):
    import networkx as nx
    try :
        M = copy(mat)
        med = np.percentile(M[M>1e-10], 40)
        #med = np.mean(M[M>1e-6])
        #med = 1e-6

        M[M<=med] = 0
        M[M>med] = 1
        G = nx.from_numpy_matrix(M)
        return nx.average_shortest_path_length(G)
    except: #networkX Error
        return np.nan


def get_rhoj(state):
    L = int(log2(len(state)))
    rhoj = np.asarray([mx.rdm(state, [j]) for j in range(L)])
    return rhoj


def get_rhojk(state):
    L = int(log2(len(state)))
    rhojk = np.asarray(
        [mx.rdm(state, [j, k]) for j in range(L) for k in range(j)]
    )
    return rhojk


def get_bisect(state):
    L = int(log2(len(state)))
    center_rho = mx.rdm(state, list(range(int(L / 2))))
    return center_rho


def get_ebisect(state):
    center_rho = get_bisect(state)
    return spectrum(center_rho)


def get_bipart(state):
    N = int(log2(len(state))) - 1
    c = int(N / 2)
    left_rdms = [0] * c
    left_rdm = mx.rdm(state, range(c))
    left_rdms[-1] = left_rdm
    right_rdms = [0] * (N - c)
    right_rdm = mx.rdm(state, range(c + 1, N + 1))
    right_rdms[0] = right_rdm
    for j in range(c - 1):
        left_rdm = mx.traceout_last(left_rdm)
        left_rdms[c - j - 2] = left_rdm
        right_rdm = mx.traceout_first(right_rdm)
        right_rdms[j + 1] = right_rdm

    if N % 2 != 0:
        right_rdm = mx.traceout_first(right_rdm)
        right_rdms[-1] = right_rdm
    return left_rdms + right_rdms


def get_ebipart(state):
    rs = get_bipart(state)
    return [spectrum(r) for r in rs]


def get_state(state):
    return state


def get_bitstring(state):
    return (np.conjugate(state) * state).real


def init_rhoj(L, M):
    return np.zeros((M, L, 2, 2), dtype=complex)


def init_rhojk(L, M):
    return np.zeros((M, int(L*(L-1)/2), 4, 4), dtype=complex)


def init_bipart(L, M):
    N = L - 1
    c = int(N / 2)
    left_dims = [2 ** (l + 1) for l in range(c)]
    right_dims = left_dims
    if N % 2 != 0:
        right_dims = np.append(right_dims, 2 ** (c + 1))
    dims = np.append(left_dims, right_dims[::-1])
    init_shape = [[np.zeros((d, d), dtype=complex) for d in dims] for _ in range(M)]
    return init_shape


def init_ebipart(L, M):
    N = L - 1
    c = int(N / 2)
    left_dims = [2 ** (l + 1) for l in range(c)]
    right_dims = left_dims
    if N % 2 != 0:
        right_dims = np.append(right_dims, 2 ** (c + 1))
    dims = np.append(left_dims, right_dims[::-1])
    init_shape = [[np.zeros(d, dtype=float) for d in dims] for _ in range(M)]
    return init_shape


def init_bisect(L, M):
    return np.zeros((M, 2 ** int(L / 2), 2 ** int(L / 2)), dtype=complex)


def init_ebisect(L, M):
    return np.zeros((M, 2 ** int(L / 2)), dtype=float)


def init_state(L, M):
    return np.zeros((M, 2**L), dtype=complex)


def init_bitstring(L, M):
    return np.zeros((M, 2**L), dtype=float)


measures = {"rhoj": {"init": init_rhoj, "get": get_rhoj},
            "rhojk": {"init": init_rhojk, "get": get_rhojk},
            "bipart": {"init": init_bipart, "get": get_bipart},
            "ebipart": {"init": init_ebipart, "get": get_ebipart},
            "bisect": {"init": init_bisect, "get": get_bisect},
            "ebisect": {"init": init_ebisect, "get": get_ebisect},
            "state": {"init": init_state, "get": get_state},
            "bitstring": {"init": init_bitstring, "get": get_bitstring}
            }


def select_jk(rhojk, j, k):
    if j == k:
        raise ValueError(
            "[{}, {}] not valid two site indicies (cannot be the same)".format(
                j, k)
        )
    row = max(j, k)
    col = min(j, k)
    ind = sum(range(row)) + col
    return rhojk[ind]


def symm_mat_from_vec(vec):
    N = len(vec)
    L = int((1 + sqrt(1 + 8 * N)) / 2)
    mat = np.zeros((L, L))
    for j in range(L):
        for k in range(L):
            if j != k:
                mat[j, k] = select_jk(vec, j, k)
    return mat


def get_spectrum(rhos):
    return np.array([spectrum(rho) for rho in rhos])


def get_entropy(rhos, order):
    return np.array([renyi_entropy(rho, order) for rho in rhos])


def get_expectation(rhos, A):
    return np.array([expectation(rho, A) for rho in rhos])


def get_expectation2(rhos, A, B):
    AB = np.kron(A, B)
    BA = np.kron(B, A)
    exp2AB = get_expectation(rhos, AB)
    exp2BA = get_expectation(rhos, BA)
    N = len(rhos)
    L = int((1 + sqrt(1 + 8 * N)) / 2)
    mat = np.zeros((L, L))
    for j in range(L):
        for k in range(L):
            if j > k:
                mat[j, k] = select_jk(exp2AB, j, k)
            elif j < k:
                mat[j, k] = select_jk(exp2BA, j, k)
    return mat


def get_entropy_from_spectrum(specs, order):
    return np.array([renyi_entropy_from_spectrum(spec, order) for spec in specs])


def get_entropy2(rhos, order):
    s2_vals = np.array([renyi_entropy(rho, order) for rho in rhos])
    return symm_mat_from_vec(s2_vals)


def get_bitstring_entropy(bitstring):
    return sum(-p * np.log2(p) for p in bitstring if p > 0)


def get_bitstring_crossentropy(bitstringp, bitstringq, tol=1e-14):
    return sum(-p * np.log2(q) if q>=tol else -p*np.log2(tol)
        for p, q in zip(bitstringp, bitstringq))


def get_bitstring_fidelity(pmeasured, pexpected):
    pincoherent = np.ones(pmeasured.shape, dtype=float)
    pincoherent /= np.sum(pincoherent)
    S_inc_exp = get_bitstring_crossentropy(pincoherent, pexpected)
    S_meas_exp = get_bitstring_crossentropy(pmeasured, pexpected)
    S_exp = get_bitstring_entropy(pexpected)
    return (S_inc_exp - S_meas_exp) / (S_inc_exp - S_exp)


def get_clustering_fidelity(Cmeasured, Cexpected, L):
    Cincoherent = 2.856 * np.e**(-L/1.46) # calibrated elsewhere
    return (Cmeasured - Cincoherent ) / (Cexpected - Cincoherent)

def get_MI(s, s2):
    L = len(s)
    MI = np.zeros((L, L))
    for j in range(L):
        for k in range(L):
            if j != k:
                MI[(j, k)] = np.abs(s[j] + s[k] - s2[(j, k)]) / 2.0
    return MI


def get_MI_from_state(state, order):
    rhoj = get_rhoj(state)
    rhojk = get_rhojk(state)
    s1 = get_entropy(rhoj, order)
    s2 = get_entropy2(rhojk, order)
    return get_MI(s1, s2)

#def get_classicla_MI(pm, pmn):


def get_g2(exp12, exp1, exp2):
    L = len(exp1)
    g2 = np.zeros((L, L))
    for j in range(L):
        for k in range(L):
            if j != k:
                g2[(j, k)] =  exp12[(j, k)] - exp1[j] * exp2[k]
    return g2


def get_g2_from_state(state, A, B):
    rhoj = get_rhoj(state)
    rhojk = get_rhojk(state)
    exp1 = get_expectation(rhoj, A)
    exp2 = get_expectation(rhoj, B)
    exp12 = get_expectation2(rhojk, A, B)
    return get_g2(exp12, exp1, exp2)


def msqrt(mat):
    return fractional_matrix_power(mat, 0.5)


def get_fidelity(rho, sigma):
    if np.allclose(rho, sigma):
        return 1
    sqrt_rho = msqrt(rho)
    return (np.trace(msqrt(multi_dot([sqrt_rho, sigma, sqrt_rho]))).real)**2


def get_relative_entropy(rho, sigma):
    if np.allclose(rho, sigma):
        return 0
    return np.trace(rho.dot(logm(rho) - logm(sigma))).real


def KL_divergence(ps, qs, tol=1e-6):
    assert np.all(ps[qs<=tol] <= tol)
    return np.sum([p * np.log2(p/q) for p, q in zip(ps/sum(ps), qs/sum(qs))])


def autocorr(x, h=1):
    N = len(x)
    mu = np.mean(x)
    acorr = sum(((x[j] - mu) * (x[j + h] - mu) for j in range(N - h)))
    denom = sum(((x[j] - mu) ** 2 for j in range(N)))
    if denom > 1e-14:
        acorr = acorr / denom
    else:
        print("auto correlation denom less than", 1e-14)
    return acorr


def fourier(sig, dt=1, h=1):
    sig = np.nan_to_num(sig)
    # remove transient. TODO: check heuristic
    if len(sig) > 300:
        sig = sig[300:]
    sig = sig - np.mean(sig)
    n = sig.size
    ps = np.absolute(np.fft.rfft(sig) / n) ** 2
    fs = np.fft.rfftfreq(n, dt)
    a1 = autocorr(sig, h=1)
    a = a1
    rn = 1 - a ** 2
    rn = rn / (1 - 2 * a * np.cos(2 * pi * fs / dt) + a ** 2)
    rn = rn * sum(ps) / sum(rn)
    return np.asarray([fs, ps[: n // 2 + 1], rn])


def fourier2D(vec, dt=1, dx=1):
    vec = np.nan_to_num(vec)
    T, L = vec.shape
    if T > 300:
        vec = vec[300:, :]
        T = T - 300
    vec = vec - np.mean(vec)
    ps = np.absolute(np.fft.fft2(vec) / (L * T)) ** 2
    ps = ps[: T // 2 + 1, : L // 2 + 1]
    ws = np.fft.rfftfreq(T, d=dt)
    ks = np.fft.rfftfreq(L, d=dx)
    ret = np.zeros((len(ws) + 1, len(ks) + 1))
    ret[0, 1:] = ks
    ret[1:, 0] = ws
    ret[1:, 1:] = ps
    return ret

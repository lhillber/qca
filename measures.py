import numpy as np
from numpy import log2, sqrt
from numpy.linalg import matrix_power, multi_dot
from scipy.linalg import fractional_matrix_power, logm
import scipy as sp
import scipy.linalg
import matrix as mx
from math import pi


def spectrum(rho):
    spec = sp.linalg.eigvalsh(rho)
    return spec


def vn_entropy(rho, tol=1e-14):
    spec = spectrum(rho)
    s = -np.sum(((el * log2(el) if el >= tol else 0.0) for el in spec))
    return s


def renyi_entropy(rho, order=2, tol=1e-14):
    if order == 1:
        return vn_entropy(rho, tol=tol)
    else:
        denom = 1.0 - order
        # spec = spectrum(rho)
        # s = np.real(log2(np.sum(spec**order)) / denom)
        s = np.real(log2(np.trace(matrix_power(rho, order))) / denom)
    return s

def expectation(state, A):
    if len(state.shape) == 2:
        exp_val = np.real(np.trace(state.dot(A)))
    else:
        if len(state.shape) == 1:
            exp_val = np.real(np.conjugate(state).dot(A.dot(state)))
        else:
            raise ValueError("Input state not understood")
    return exp_val


def network_density(mat):
    l = len(mat)
    lsq = l * (l - 1)
    return sum(sum(mat)) / lsq


def network_clustering(mat):
    l = len(mat)
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


def get_rhoj(state):
    L = int(log2(len(state)))
    rhoj = np.asarray([mx.rdms(state, [j]) for j in range(L)])
    return rhoj


def get_rhojk(state):
    L = int(log2(len(state)))
    rhojk = np.asarray(
        [mx.rdms(state, [j, k]) for j in range(L) for k in iter(range(j))]
    )
    return rhojk


def get_bisect(state, out=None):
    L = int(log2(len(state)))
    center_rho = mx.rdms(state, list(range(int(L / 2))), out=out)
    return center_rho


def get_bipart(state):
    N = int(log2(len(state))) - 1
    c = int(N / 2)
    left_rdms = [0] * c
    left_rdm = mx.rdms(state, range(c))
    left_rdms[-1] = left_rdm
    right_rdms = [0] * (N - c)
    right_rdm = mx.rdms(state, range(c + 1, N + 1))
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


def init_bisect(L, M):
    return np.zeros((M, 2 ** int(L / 2), 2 ** int(L / 2)), dtype=complex)


def init_state(L, M):
    return np.zeros((M, 2**L), dtype=complex)


def init_bitstring(L, M):
    return np.zeros((M, 2**L), dtype=float)


measures = {"rhoj": {"init": init_rhoj, "get": get_rhoj},
            "rhojk": {"init": init_rhojk, "get": get_rhojk},
            "bipart": {"init": init_bipart, "get": get_bipart},
            "bisect": {"init": init_bisect, "get": get_bisect},
            "state": {"init": init_state, "get": get_state},
            "bitstring": {"init": init_bitstring, "get": get_bitstring}
            }


def select_rhojk(rhojk, j, k):
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
                mat[(j, k)] = select_rhojk(vec, j, k)
    return mat


def get_expectation(rhos, A):
    return np.asarray([expectation(rho, A) for rho in rhos])


def get_expectation2(rhos, A, B):
    exp2_vals = np.asarray([expectation(rho, np.kron(A, B)) for rho in rhos])
    exp2 = symm_mat_from_vec(exp2_vals)
    return exp2


def get_entropy(rhos, order):
    return np.asarray([renyi_entropy(rho, order) for rho in rhos])


def get_entropy2(rhos, order):
    s2_vals = np.asarray([renyi_entropy(rho, order) for rho in rhos])
    return symm_mat_from_vec(s2_vals)


def get_bitstring_entropy(bitstring):
    return sum(-p * np.log(p) for p in bitstring if p > 0)


def get_bitstring_crossentropy(bitstringp, bitstringq):
    return sum(-p * np.log(q) for p , q in zip(bitstringp, bitstringq) if p > 0)


def get_MI(s, s2):
    L = len(s)
    MI = np.zeros((L, L))
    for j in range(L):
        for k in range(L):
            if j != k:
                MI[(j, k)] = np.abs(s[j] + s[k] - s2[(j, k)]) / 2.0
    return MI


def get_g2(expAB, expA, expB):
    L = len(expA)
    g2 = np.zeros((L, L))
    for j in range(L):
        for k in range(j):
            g2[(j, k)] = expAB[(j, k)] - expA[j] * expB[k]
            g2[(k, j)] = g2[(j, k)]

    return g2


def msqrt(mat):
    return fractional_matrix_power(mat, 0.5)


def fidelity(rho, sigma):
    if np.allclose(rho, sigma):
        return 1
    sqrt_rho = msqrt(rho)
    return (np.trace(msqrt(multi_dot([sqrt_rho, sigma, sqrt_rho]))).real)**2


def relative_entropy(rho, sigma):
    if np.allclose(rho, sigma):
        return 0
    return np.trace(rho.dot(logm(rho) - logm(sigma))).real


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

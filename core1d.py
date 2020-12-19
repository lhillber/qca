import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matrix as mx
import measures as ms
from copy import copy
from states import make_state
from numpy.linalg import matrix_power
from scipy.linalg import expm, fractional_matrix_power
from itertools import permutations
from itertools import product, cycle, zip_longest
from hashlib import sha1
from copy import deepcopy
from json import dumps


def rule_element(V, Rel, hood, hamiltonian=False, lslice=None, rslice=None):
    """
    Operator for neighborhood hood with activation V if Rel=1
    or `option` if Rel=0.
    """
    N = len(hood)
    r = N // 2
    if hamiltonian:
        option = "O"  # zero matrix
    else:  # unitaty
        option = "I"  # identity matrix
    if lslice is None:
        lslice = slice(0, r)
    if rslice is None:
        rslice = slice(r, N + 1)
    Vmat = matrix_power(mx.make_U2(V), Rel)
    mx.ops["V"] = Vmat
    ops = (
        [str(el) for el in hood[lslice]]
        + ["V" if Rel else option]
        + [str(el) for el in hood[rslice]]
    )
    OP = mx.listkron([mx.ops[op] for op in ops])
    return OP


def rule_op(V, R, r, totalistic=False, hamiltonian=False):
    """
    Operator for rule R, activation V, and neighborhood radius r.
    totalistic flag for rule numbering schemes and hamiltonian flag
    for simultaion type (hamiltonian=analog, unitary=digital)
    """
    N = 2 * r
    OP = np.zeros((2 ** (N + 1), 2 ** (N + 1)), dtype=complex)
    if totalistic:
        R2 = mx.dec_to_bin(R, N + 1)[::-1]
        for elnum, Rel in enumerate(R2):
            K = elnum * [1] + (N - elnum) * [0]
            hoods = list(set([perm for perm in permutations(K, N)]))
            hoods = map(list, hoods)
            for hood in hoods:
                OP += rule_element(V, Rel, hood, hamiltonian=hamiltonian)

    else:  # non-totalistic
        R2 = mx.dec_to_bin(R, 2 ** N)[::-1]
        for elnum, Rel in enumerate(R2):
            hood = mx.dec_to_bin(elnum, N)
            OP += rule_element(V, Rel, hood, hamiltonian=hamiltonian)

    if hamiltonian:
        assert mx.isherm(OP)
    else:  # unitaty
        assert mx.isU(OP)
    return OP


def boundary_rule_ops(V, R, r, BC_conf, totalistic=False, hamiltonian=False):
    """
    Special operators for boundaries (of which there are 2r).
    BC_conf is a string "b0b1...br...b2r" where each bj is
    either 0 or 1. Visiually, BC_conf represents the fixed boundaries
    from left to right: |b0>|b1>...|br> |psi>|b2r-r>|b2r-r+1>...|b2r>.
    """
    # split BC configuration into left and reverse-right boundaries
    BC_conf = [BC_conf[:r], BC_conf[r::][::-1]]
    N = 2 * r
    OPs = []
    for j, BC in enumerate(BC_conf):
        for e in range(r):
            dim = r + 1 + e
            if j == 0:  # left boundary
                lslice = slice(r - e, r)
                rslice = slice(r, N)
                cslice = slice(0, r - e)
            elif j == 1:  # right boundary
                lslice = slice(0, r)
                rslice = slice(r, r + e)
                cslice = slice(r + e, N)
            OP = np.zeros((2 ** dim, 2 ** dim), dtype=complex)
            if totalistic:
                R2 = mx.dec_to_bin(R, N + 1)[::-1]
                for elnum, Rel in enumerate(R2):
                    K = elnum * [1] + (N - elnum) * [0]
                    hoods = list(set([perm for perm in permutations(K, N)]))
                    hoods = map(list, hoods)
                    for hood in hoods:
                        if BC[e:r] == hood[cslice]:
                            OP += rule_element(
                                V,
                                Rel,
                                hood,
                                lslice=lslice,
                                rslice=rslice,
                                hamiltonian=hamiltonian,
                            )

            else:  # non-totalistic
                R2 = mx.dec_to_bin(R, 2 ** N)[::-1]
                for elnum, Rel in enumerate(R2):
                    hood = mx.dec_to_bin(elnum, N)
                    if BC[e:r] == hood[cslice]:
                        OP += rule_element(
                            V,
                            Rel,
                            hood,
                            lslice=lslice,
                            rslice=rslice,
                            hamiltonian=hamiltonian,
                        )
            if hamiltonian:
                assert mx.isherm(OP)
            else:  # unitaty
                assert mx.isU(OP)
            OPs.append(OP)
    return OPs[:r], OPs[r:][::-1]


def rule_hamiltonian(V, R, r, BC, L=None, totalistic=False):
    BC_type, *BC_conf = BC.split("-")
    BC_conf = "".join(BC_conf)
    if BC_type == "1":
        BC_conf = [int(bc) for bc in BC_conf]

    bulk = rule_op(V, R, r, totalistic=totalistic, hamiltonian=True)
    lUs, rUs = boundary_rule_ops(
        V, R, r, BC_conf, totalistic=totalistic, hamiltonian=True
    )

    if L is None:
        L = 2 * r + 1
        return bulk

    else:  # pad identities
        H = np.zeros((2**L, 2**L), dtype=complex)
        for j in range(r, L - r):
            ln = j - r
            rn = L - 2 * r - 1 - ln
            left = np.eye(2 ** ln)
            right = np.eye(2 ** rn)
            H += mx.listkron([left, bulk, right])
        # boundaries
        for j, (lU, rU) in enumerate(zip(lUs, rUs[::-1])):
            end = np.eye(2**(L - r - 1 - j))
            H += mx.listkron([end, rU])
            H += mx.listkron([lU, end])
        return H


def rule_unitaries(V, R, r, BC, L, dt,
                   totalistic=False, hamiltonian=False, trotter=True):
    """
    Calculate qca unitiary activation V, rule R, radius r, bounary condition BC,
    size L, and time step dt.
    """
    BC_type, *BC_conf = BC.split("-")
    BC_conf = "".join(BC_conf)
    if BC_type == "1":
        BC_conf = [int(bc) for bc in BC_conf]
    elif BC_type == "0":
        BC_conf = [0]*2*r
    if L is None:
        L = 2 * r + 1
    bulk = rule_op(V, R, r, totalistic=totalistic, hamiltonian=hamiltonian)
    lUs, rUs = boundary_rule_ops(
        V, R, r, BC_conf, totalistic=totalistic, hamiltonian=hamiltonian
    )
    if hamiltonian:
        if trotter:
            bulk = expm(-1j * bulk * dt)
            rUs = [expm(-1j * H * dt) for H in rUs]
            lUs = [expm(-1j * H * dt) for H in lUs]
        else:  # not trotter:
            H = np.zeros((2**L, 2**L), dtype=complex)
            for j in range(r, L - r):
                ln = j - r
                rn = L - 2 * r - 1 - ln
                left = np.eye(2 ** ln)
                right = np.eye(2 ** rn)
                H += mx.listkron([left, bulk, right])
            # boundaries
            for j, (lU, rU) in enumerate(zip(lUs, rUs[::-1])):
                end = np.eye(2**(L - r - 1 - j))
                H += mx.listkron([end, rU])
                H += mx.listkron([lU, end])
            U = expm(-1j * H * dt)
            assert mx.isU(U)
            return U

    if BC_type == "0":
        return bulk
    else:  # BC_type == "1"
        return lUs, bulk, rUs


def get_Ufunc(Us, r, L, BC):
    """
    Define neighborhood and associated update operators for
    any qubit j
    """
    BC_type, *BC_conf = BC.split("-")
    if BC_type == "1":
        lUs, U, rUs = Us

        def get_U(j):
            if j < r:
                Nj = range(0, j + r + 1)
                u = lUs[j]
            elif j >= L - r:
                Nj = range(j - r, L)
                u = rUs[-L + j]
            elif r <= j < L - r:
                Nj = range(j - r, j + r + 1)
                u = U
            else:
                raise ValueError
            Nj = list(Nj)
            return Nj, u

    elif BC_type == "0":
        u = Us

        def get_U(j):
            Nj = [k % L for k in range(j - r, j + r + 1)]
            return Nj, u

    return get_U


def depolarize(state, Nj, E):
    """
    Depolarization noise of error rate E applied to state
    """
    if E == 0.0:
        return state
    np.random.seed(None)
    rnd = np.random.rand()
    if rnd < E: # E is single qubit error rate per neighborhood-sized gate
        # random site in neighborhood
        q = np.random.choice(Nj)
        # random Pauli op
        op = mx.ops[np.random.choice(["X", "Y", "Z"])]
        state = mx.op_on_state(op, [q], state)
    return state


def evolve(L, T, dt, R, r, V, IC, BC, E=0,
           totalistic=False, hamiltonian=False,
           symmetric=False, trotter=True, initstate=None, **kwargs):
    """
    Generator of qca dynamics yields state at each time step
    """
    Us = rule_unitaries(V, R, r, BC, L, dt, totalistic=totalistic,
                        hamiltonian=hamiltonian, trotter=trotter)
    ts = np.arange(dt, T + dt, dt)
    if initstate is None:
        initstate = make_state(L, IC)
    yield initstate
    state = initstate

    if not trotter:
        u = Us
        for t in ts:
            state = u.dot(state)
            yield state

    else:  # trotter
        get_U = get_Ufunc(Us, r, L, BC)
        if symmetric:
            if BC == "0":
                sqrtU = fractional_matrix_power(Us, 0.5)
                get_sqrtU = get_Ufunc(sqrtU, r, L, BC)
            else:
                sqrtUs = [[fractional_matrix_power(u, 0.5)
                           for u in us] for us in (Us[0], Us[2])]
                sqrtUs = (sqrtUs[0], fractional_matrix_power(
                    Us[1], 0.5), sqrtUs[1])
                get_sqrtU = get_Ufunc(sqrtUs, r, L, BC)

            for t in ts:
                # forward
                for k in range(r):
                    for j in range(k, L, r + 1):
                        Nj, u = get_sqrtU(j)
                        state = mx.op_on_state(u, Nj, state)
                        state = depolarize(state, Nj, E)

                # center
                for j in range(r, L, r + 1):
                    Nj, u = get_U(j)
                    state = mx.op_on_state(u, Nj, state)
                    state = depolarize(state, Nj, E)

                # backward
                for k in range(r - 1, -1, -1):
                    for j in range(k, L, r + 1):
                        Nj, u = get_sqrtU(j)
                        state = mx.op_on_state(u, Nj, state)
                        state = depolarize(state, Nj, E)
                yield state

        else:  # not symmetric
            for t in ts:
                for k in range(r + 1):
                    for j in range(k, L, r + 1):
                        Nj, u = get_U(j)
                        state = mx.op_on_state(u, Nj, state)
                        state = depolarize(state, Nj, E)
                yield state



def hash_state(d, keep_keys=None, reject_keys=None):
    """
    Create a unique ID for a dict based on the values
    associated with uid_keys.
    """
    if keep_keys is None:
        keep_keys = d.keys()
    if reject_keys is None:
        reject_keys = []
    name_dict = {}
    dc = deepcopy(d)
    for k, v in dc.items():
        if k in keep_keys and k not in reject_keys:
            name_dict[k] = v
    dict_el_array2list(name_dict)
    dict_el_int2float(name_dict)
    dict_key_to_string(name_dict)
    uid = sha1(dumps(name_dict, sort_keys=True).encode(
        "utf-8")).hexdigest()
    return uid


def dict_el_array2list(d):
    """
    Convert dict values to lists if they are arrays.
    """
    for k, v in d.items():
        if type(v) == np.ndarray:
            d[k] = list(v)
        if type(v) == dict:
            dict_el_array2list(v)
        if type(v) == list:
            for i, vel in enumerate(v):
                if type(vel) == dict:
                    dict_el_array2list(vel)
                if type(vel) == np.ndarray:
                    v[i] = list(vel)


def dict_el_int2float(d):
    """
    Convert dict values to floats if they are ints.
    """
    for k, v in d.items():
        if type(v) in (int, np.int64):
            d[k] = float(v)
        if type(v) == dict:
            dict_el_int2float(v)
        if type(v) == list:
            for i, vel in enumerate(v):
                if type(vel) == dict:
                    dict_el_int2float(vel)
                if type(vel) == int:
                    v[i] = float(vel)


def dict_key_to_string(d):
    """
    Convert dict keys to strings.
    """
    for k, v in d.items():
        d[str(k)] = v
        if type(k) != str:
            del d[k]
        if type(v) == dict:
            dict_key_to_string(v)
        if type(v) == list:
            for vel in v:
                if type(vel) == dict:
                    dict_key_to_string(vel)


def save_dict_hdf5(dic, h5file):
    """Save a dictionary to hdf5 file"""
    recurs_save_dict_hdf5(h5file, "/", dic)


def recurs_save_dict_hdf5(h5file, path, dic_):
    """Recursive traversal for saving dictonary to hdf5 file"""
    for key, item in dic_.items():
        if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes)):
            if path + key in h5file.keys():
                h5file[path + key][:] = item
            else:
                h5file[path + key] = item
        elif isinstance(item, dict):
            recurs_save_dict_hdf5(h5file, path + key + "/", item)
        elif isinstance(item, list):
            item_T = [
                [item[j][i] for j in range(len(item))] for i in range(len(item[0]))
            ]
            for k, el in enumerate(item_T):
                if path + key + "/l" + str(k) in h5file.keys():
                    h5file[path + key + "/l" + str(k)][:] = el
                else:
                    h5file[path + key + "/l" + str(k)] = el

        else:
            raise ValueError("Cannot save %s type" % item)


def record(params, tasks):
    """Record tasks from qca time evolution defined by params into a
       dictionary"""
    ts = np.arange(0, params["T"] + params["dt"], params["dt"])
    keys = [task+"data" if task in ("ebipart", "ebisect") else task for task in tasks]
    rec = {key: ms.measures[task]["init"](
        params["L"], len(ts)) for key, task in zip(keys,tasks)}
    rec.update({"ts": ts})
    # average of reduced density matricies
    # TODO: logic for entanglement spectrum if N>1
    for n in range(params["N"]):
        for ti, state in enumerate(evolve(**params)):
            for task in tasks:
                key = task
                if task in ("ebipart", "ebisect"):
                    key += "data"
                if task in ("bipart", "ebipart"):
                    data = ms.measures[task]["get"](state)
                    for l in range(params["L"] - 1):
                        rec[key][ti][l] += data[l] / params["N"]
                else:
                    rec[key][ti] += ms.measures[task]["get"](
                        state) / params["N"]
    return rec


def make_params_dict(params, L, Lx, T, dt, R, r, V, IC, BC, E, N):
    """ Explicit conversion of parameters to dictionary. Updates
        a base dictonary 'params' """
    p = copy(params)
    p.update(
            {"L": L, "Lx":Lx, "T": T, "dt": dt, "R": R, "r": r,
             "V": V, "IC": IC, "BC": BC, "E": E, "N": N})
    return p


def product_params_list(params, *args):
    """ Product set of lists of params """
    return [make_params_dict(params, *p) for p in product(*args)]


def cycle_params_list(params, *args):
    """ Cycle shorter lists  of params """
    lens = [l for l in map(len, args)]
    ind = np.argmax(lens)
    to_zip = [el for el in map(cycle, args)]
    to_zip[ind] = args[ind]
    return [make_params_dict(params, *p) for p in zip(*to_zip)]


def repeat_params_list(params, *args):
    """ Repeat las element of shorter lists of params """
    lens = np.array([l for l in map(len, args)])
    ind = np.argmax(lens)
    longest = lens[ind]
    pads = longest - lens
    to_zip = [arg + [arg[-1]] * pad for arg, pad in zip(args, pads)]
    return [make_params_dict(params, *p) for p in zip_longest(*to_zip)]


# collect parameter list constructors into
# a dictonary of functions
params_list_map = {"product": product_params_list,
                   "cycle": cycle_params_list,
                   "repeat": repeat_params_list}

if __name__ == "__main__":
    L, Lx = 17, 1
    Ly = int(L/Lx)
    T = 60
    dt = 1
    Rs = [4]
    r = 2
    fig, axs = plt.subplots(2, 2)
    Zgrids = np.zeros((len(Rs), T+1, Ly))
    MI_meas = np.zeros((len(Rs), 2, T+1))
    ims = []
    for ll, R in enumerate(Rs):
        lj, lk = np.unravel_index(ll, (2, 2))
        evo = evolve(L, T, dt, R, r, "H","c3_f0-2", "1-0000", totalistic=True, hamiltonian=True, symmetric=False)
        for t, state in enumerate(evo):
            rhoj = ms.get_rhoj(state)
            # rhojk = ms.get_rhojk(state).reshape(Ly, Lx, 2, 2)
            Zgrids[ll, t] = ms.get_expectation(rhoj, mx.ops["Z"])
        plt.clf()
        plt.imshow(Zgrids[ll,:,:], origin="lower")
        plt.show()
        im = axs[lj, lk].imshow(Zgrids[ll, 0], vmin=-1, vmax=1, cmap="inferno")
        axs[lj, lk].axis("off")
        axs[lj, lk].set_title(f"R={R}")
        ims.append(im)


import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matrix as mx
import measures as ms
import matplotlib.animation as animation
from copy import copy
from states import make_state
from numpy.linalg import matrix_power
from scipy.linalg import expm, fractional_matrix_power
from itertools import permutations
from itertools import product, cycle, zip_longest
from hashlib import sha1
from copy import deepcopy
from json import dumps


tri_dict = {1: [0, 2], 2: [1, 2], 3: [1, 3], 4: [0, 3]}

tri_comp_dict = {1: [1, 3], 2: [0, 3], 3: [0, 2], 4: [1, 2]}

rod_dict = {1: [1, 2, 3], 2: [0, 2, 3], 3: [0, 1, 3], 4: [0, 1, 2]}

rod_comp_dict = {1: [0], 2: [1], 3: [2], 4: [3]}


def make_mask(j, k, Ly, Lx, r, d, BC_type, tri=0, rod=0):
    mask = np.array([True] * 2 * r * d)
    if BC_type == "1":
        if j - r < 0:
            for i in range(r - j):
                mask[i] = False
        if j + r > Ly - 1:
            for i in range(r, 2 * r + j - Ly + 1):
                mask[i] = False
        if d == 2:
            if k - r < 0:
                for i in range(2 * r, 3 * r - k):
                    mask[i] = False

            if k + r > Lx - 1:
                for i in range(3 * r, 4 * r + k - Lx + 1):
                    mask[i] = False

    if tri in [1, 2, 3, 4]:
        mask[tri_dict[tri]] = False

    if rod in [1, 2, 3, 4]:
        mask[rod_dict[rod]] = False

    return mask


def central(j, k, Ly, Lx):
    return np.ravel_multi_index([j, k], (Ly, Lx))


def neighbors_index_arr(j, k, r, d):
    # U, D, L, R
    if d == 2:
        return np.vstack(
            (
                np.r_[
                    np.arange(j - r, j),
                    np.arange(j + 1, j + r + 1)[::-1],
                    np.ones(2 * r) * j,
                ],
                np.r_[
                    np.ones(2 * r) * k,
                    np.arange(k - r, k),
                    np.arange(k + 1, k + r + 1)[::-1],
                ],
            )
        ).astype(int)
    elif d == 1:
        return np.vstack(
            (
                np.r_[np.arange(j - r, j), np.arange(j + 1, j + r + 1)[::-1]],
                np.r_[np.zeros(2 * r)],
            )
        ).astype(int)


def neighbors(j, k, Ly, Lx, r, d, BC_type, tri=0, rod=0):
    mask = make_mask(j, k, Ly, Lx, r, d, BC_type, tri=tri, rod=rod)
    index = neighbors_index_arr(j, k, r, d)
    Njk = np.ravel_multi_index(index, (Ly, Lx), mode="wrap")
    return Njk[mask][::-1]


def neighborhood(j, k, Ly, Lx, r, d, BC_type, tri=0, rod=0):
    return np.r_[
        central(j, k, Ly, Lx), neighbors(
            j, k, Ly, Lx, r, d, BC_type, tri=tri, rod=rod)
    ]


def rule_element(V, Rel, hood, hamiltonian=False):
    """
    Operator for neighborhood bitstring hood with activation V if Rel=1
    or `option` if Rel=0.
    """
    Vmat = mx.make_U2(V)
    if hamiltonian:
        Vmat = Vmat * Rel
    else:  # unitaty
        Vmat = matrix_power(Vmat, Rel)
    ops = [Vmat] + [mx.ops[str(el)] for el in hood]
    OP = mx.listkron(ops)
    return OP


def rule_op(V, R, r, d, totalistic=False, hamiltonian=False, triangles=False, rods=False):
    """
    Operator for rule R, activation V, and neighborhood radius r, and dimension d.
    hamiltonian flag for simultaion type (hamiltonian=analog, unitary=digital)
    """
    if triangles:
        factor = 1
    elif rods:
        factor = 1 / 2
    else:
        factor = d
    N = int(2 * r * factor)
    OP = np.zeros((2 ** (N + 1), 2 ** (N + 1)), dtype=complex)
    if totalistic:
        R2 = mx.dec_to_bin(R, N + 1)[::-1]
    else:
        R2 = mx.dec_to_bin(R, 2 ** N)[::-1]
    for elnum, Rel in enumerate(R2):
        if totalistic:
            K = elnum * [1] + (N - elnum) * [0]
            hoods = list(set([perm for perm in permutations(K, N)]))
            hoods = map(list, hoods)
        else:
            hoods = [mx.dec_to_bin(elnum, N)]
        for hood in hoods:
            OP += rule_element(V, Rel, hood, hamiltonian=hamiltonian)
    if hamiltonian:
        assert mx.isherm(OP)
    else:  # unitaty
        assert mx.isU(OP)
    return OP


def boundary_rule_ops(V, R, r, d, BC_conf, totalistic=False, hamiltonian=False):
    """
    Special operators for boundaries (of which there are 2r).
    BC_conf is a string "b0b1...br...b2r" where each bj is
    either 0 or 1. Visiually, BC_conf represents the fixed boundaries
    from left to right: |b0>|b1>...|br> |psi>|b2r-r>|b2r-r+1>...|b2r>.
    """
    BC_conf = np.array([int(s) for s in BC_conf])
    OPs = []
    N = 2 * r * d
    if totalistic:
        R2 = mx.dec_to_bin(R, N + 1)[::-1]
    else:
        R2 = mx.dec_to_bin(R, 2 ** N)[::-1]

    indsj = [np.arange(r), np.array([r]), np.arange(r + 1, 2 * r + 1)]
    dj = 2 * r + 1
    if d == 2:
        indsk = indsj
        dk = 2 * r + 1
    elif d == 1:
        indsk = [[0]]
        dk = 1
    OPs = []
    for J, js in enumerate(indsj):
        for K, ks in enumerate(indsk):
            OPs_region = []
            for j in js:
                OPs_row = []
                for k in ks:
                    mask = make_mask(j, k, dj, dk, r, d, BC_type="1", tri=0)
                    clip = np.logical_not(mask)
                    if np.sum(clip) > 0:
                        dim = np.sum(mask) + 1
                        OP = np.zeros((2 ** dim, 2 ** dim), dtype=complex)
                        for elnum, Rel in enumerate(R2):
                            if totalistic:
                                tot = elnum * [1] + (N - elnum) * [0]
                                hoods = list(
                                    set([perm for perm in permutations(tot, N)])
                                )
                                hoods = map(np.array, hoods)
                            else:
                                hoods = np.array([mx.dec_to_bin(elnum, N)])
                            for hood in hoods:
                                if np.all(BC_conf[clip] == hood[clip]):
                                    OP += rule_element(
                                        V, Rel, hood[mask], hamiltonian=hamiltonian,
                                    )
                        if hamiltonian:
                            assert mx.isherm(OP)
                        else:  # unitaty
                            assert mx.isU(OP)
                        OPs_row.append(OP)
                OPs_region.append(OPs_row)
            if len(OPs_region[0]) > 0:
                OPs.append(OPs_region)
    return OPs


def boundary_rule_ops_triangles(
    V, R, r, d, BC_conf, totalistic=False, hamiltonian=False
):
    assert d == 2
    assert r == 1
    assert BC_conf == [0, 0]
    assert totalistic == True
    N = 2
    R2 = mx.dec_to_bin(R, N + 1)[::-1]
    BC_conf = np.array([int(s) for s in BC_conf])
    OPs = []
    for tri in [2, 3]:
        mask = make_mask(0, 0, 2, 2, r, d, BC_type="1", tri=tri)
        mask = mask[tri_comp_dict[tri]]
        clip = np.logical_not(mask)
        dim = np.sum(mask) + 1
        OP = np.zeros((2 ** dim, 2 ** dim), dtype=complex)
        for elnum, Rel in enumerate(R2):
            tot = elnum * [1] + (N - elnum) * [0]
            hoods = list(set([perm for perm in permutations(tot, N)]))
            hoods = map(np.array, hoods)
            for hood in hoods:
                if np.all(BC_conf[clip] == hood[clip]):
                    OP += rule_element(V, Rel,
                                       hood[mask], hamiltonian=hamiltonian,)
        if hamiltonian:
            assert mx.isherm(OP)
        else:  # unitaty
            assert mx.isU(OP)
        OPs.append(OP)
    return OPs


def boundary_rule_ops_rods(V, R, r, d, BC_conf, totalistic=False, hamiltonian=False):
    assert d == 2
    assert r == 1
    assert BC_conf == [0]
    assert totalistic == True
    N = 1
    R2 = mx.dec_to_bin(R, N + 1)[::-1]
    BC_conf = np.array([int(s) for s in BC_conf])
    OPs = []
    mask = make_mask(0, 0, 2, 2, r, d, BC_type="1", rod=1)
    mask = mask[rod_comp_dict[1]]
    clip = np.logical_not(mask)
    dim = np.sum(mask) + 1
    OP = np.zeros((2 ** dim, 2 ** dim), dtype=complex)
    for elnum, Rel in enumerate(R2):
        tot = elnum * [1] + (N - elnum) * [0]
        hoods = list(set([perm for perm in permutations(tot, N)]))
        hoods = map(np.array, hoods)
        for hood in hoods:
            if np.all(BC_conf[clip] == hood[clip]):
                OP += rule_element(V, Rel,
                                   hood[mask], hamiltonian=hamiltonian,)
    if hamiltonian:
        assert mx.isherm(OP)
    else:  # unitaty
        assert mx.isU(OP)
    OPs.append(OP)
    return OPs


def rule_unitaries(
    V, R, r, d, BC, dt, totalistic=False, hamiltonian=False, triangles=False, rods=False
):
    """
    Calculate qca unitiary activation V, rule R, radius r, bounary condition BC,
    size L, and time step dt.
    """
    BC_type, *BC_conf = BC.split("-")
    BC_conf = "".join(BC_conf)
    if rods:
        factor = 1
    if triangles:
        factor = 1
    else:
        factor = d
    if BC_type == "1":
        BC_conf = [int(bc) for bc in BC_conf]
    else:
        BC_conf = [0] * 2 * r * factor
    bulk = rule_op(
        V, R, r, d, totalistic=totalistic, hamiltonian=hamiltonian, triangles=triangles, rods=rods
    )

    if triangles:
        bounds = boundary_rule_ops_triangles(
            V, R, r, d, BC_conf, totalistic=totalistic, hamiltonian=hamiltonian
        )

    elif rods:
        bounds = boundary_rule_ops_rods(
            V, R, r, d, BC_conf, totalistic=totalistic, hamiltonian=hamiltonian
        )
    else:
        bounds = boundary_rule_ops(
            V, R, r, d, BC_conf, totalistic=totalistic, hamiltonian=hamiltonian
        )

    if hamiltonian:
        bulk = expm(-1j * bulk * dt)
        if triangles:
            bounds = [expm(-1j * H * dt) for H in bounds]
        else:
            bounds = [
                [[expm(-1j * H * dt) for H in row] for row in region]
                for region in bounds
            ]

    if BC_type == "0":
        return bulk

    else:  # BC_type == "1"
        return bulk, bounds


def get_Ufunc(Us, Ly, Lx, r, d, BC, triangles, rods):
    """
    Define neighborhood and associated update operators for
    any qubit j,k
    """
    BC_type, *BC_conf = BC.split("-")
    if BC_type == "1":
        U, bUs = Us
        if d == 2:
            if triangles:

                def get_U(j, k, tri):
                    Nj = neighborhood(j, k, Ly, Lx, r, d, BC_type, tri=tri)
                    if len(Nj) == 3:
                        u = U
                    elif len(Nj) == 2:
                        u = bUs[0]
                    elif len(Nj) == 1:
                        u = bUs[1]
                    else:
                        print(f"error, j,k, {len(Nj)}")
                    return Nj, u

            elif rods:

                def get_U(j, k, rod):
                    Nj = neighborhood(j, k, Ly, Lx, r, d, BC_type, rod=rod)
                    if len(Nj) == 2:
                        u = U
                    elif len(Nj) == 1:
                        u = bUs[0]
                    else:
                        print(f"error, j, k, {len(Nj)}")
                    return Nj, u

            else:

                def get_U(j, k, tri=0):
                    if 0 <= j < r and 0 <= k < r:
                        u = bUs[0][j][k]
                        dir = "UL"

                    elif 0 <= j < r and r <= k < Lx - r:
                        u = bUs[1][j][0]
                        dir = "U"

                    elif 0 <= j < r and Lx - r <= k < Lx:
                        u = bUs[2][j][-Lx + k]
                        dir = "UR"

                    elif r <= j < Ly - r and 0 <= k < r:
                        dir = "L"
                        u = bUs[3][0][k]

                    elif r <= j < Ly - r and r <= k < Lx - r:
                        u = U
                        dir = "C"

                    elif r <= j < Ly - r and Lx - r <= k < Lx:
                        u = bUs[4][0][-Lx + k]
                        dir = "R"

                    elif Ly - r <= j < Ly and 0 <= k < r:
                        dir = "DL"
                        u = bUs[5][-Ly + j][k]

                    elif Ly - r <= j < Ly and r <= k < Lx - r:
                        dir = "D"
                        u = bUs[6][-Ly + j][0]

                    elif Ly - r <= j < Ly and Lx - r <= k < Lx:
                        dir = "DR"
                        u = bUs[7][-Ly + j][-Lx + k]
                    else:
                        print("error", j, k)

                    Nj = neighborhood(j, k, Ly, Lx, r, d, BC_type, tri=tri)
                    return Nj, u

        elif d == 1:
            L = Lx * Ly

            def get_U(j, k, tri=0):
                if j < r:
                    u = bUs[0][j]
                elif j >= L - r:
                    u = bUs[1][-L + j]
                elif r <= j < L - r:
                    u = U
                else:
                    raise ValueError
                Nj = neighborhood(j, k, Ly, Lx, r, d, BC_type, tri=tri)
                return Nj, u

    elif BC_type == "0":
        u = Us

        def get_U(j, k, tri=0):
            Nj = neighborhood(j, k, Ly, Lx, r, d, BC_type, tri=tri)
            return Nj, u

    return get_U


def depolarize(state, Nj, E):
    """
    Depolarization noise of error rate E applied to state
    """
    if E == 0.0:
        return state
    np.random.seed(None)
    for trial in range(len(Nj)):
        rnd = np.random.rand()
        if rnd < E / len(Nj):  # E is single qubit error rate per neighborhood-sized gate
            # random site in neighborhood
            q = np.random.choice(Nj)
            # random Pauli op
            op = mx.ops[np.random.choice(["X", "Y", "Z"])]
            state = mx.op_on_state(op, [q], state)
    return state


def parity_evolve(
    L,
    Lx,
    T,
    dt,
    R,
    r,
    V,
    IC,
    BC,
    E=0,
    tris=[[0], [0]],
    rods=[[0], [0]],
    totalistic=False,
    hamiltonian=False,
    symmetric=False,
    initstate=None,
    **kwargs,
):
    """
    Generator of qca dynamics yields state at each time step
    """
    if Lx == 1:
        d = 1
    elif Lx > 1:
        assert Lx < L
        d = 2

    if tris != [[0], [0]]:
        use_triangles = True
        seq = tris
    else:
        use_triangles = False

    if rods != [[0], [0]]:
        use_rods = True
        seq = rods
    else:
        use_rods = False

    if use_rods and use_triangles:
        raise ValueError

    if not use_rods and not use_triangles:
        seq = [[0], [0]]
    Ly = int(L / Lx)

    Us = rule_unitaries(
        V,
        R,
        r,
        d,
        BC,
        dt,
        totalistic=totalistic,
        hamiltonian=hamiltonian,
        triangles=use_triangles,
        rods=use_rods,
    )
    ts = np.arange(dt, T + dt, dt)
    if initstate is None:
        initstate = make_state(L, IC)
    yield initstate
    state = initstate
    get_U = get_Ufunc(Us, Ly, Lx, r, d, BC, use_triangles, use_rods)
    for ti, t in enumerate(ts):
        #subseq = seq[ti % 2]
        for s in range(r + 1):
            subseq = seq[::(-1)**(ti % 2)][s]
            for i in range(s, L):
                j, k = np.unravel_index(i, (Ly, Lx))
                if (j + k) % (r + 1) == s:
                    for seqval in subseq:
                        Nj, u = get_U(j, k, seqval)
                        state = mx.op_on_state(u, Nj, state)
                        state = depolarize(state, Nj, E)
        yield state


def block_evolve(
    L,
    Lx,
    T,
    dt,
    R,
    r,
    V,
    IC,
    BC,
    E=0,
    blocks=[1, 2, 3, 4],
    totalistic=False,
    hamiltonian=False,
    symmetric=False,
    initstate=None,
    **kwargs,
):
    """
    Generator of qca dynamics yields state at each time step
    """
    tri_map = {(0, 0): [1, 3], (0, 1): [4, 2], (1, 0): [2, 4], (1, 1): [3, 1]}
    if Lx == 1:
        d = 1
    elif Lx > 1:
        assert Lx < L
        d = 2
    blocks = [b - 1 for b in blocks]

    triangles = True
    Ly = int(L / Lx)
    Us = rule_unitaries(
        V,
        R,
        r,
        d,
        BC,
        dt,
        totalistic=totalistic,
        hamiltonian=hamiltonian,
        triangles=triangles,
    )
    ts = np.arange(dt, T // 2 + dt, dt)
    if initstate is None:
        initstate = make_state(L, IC)
    yield initstate
    state = initstate
    get_U = get_Ufunc(Us, Ly, Lx, r, d, BC, triangles)
    bjs, bks = np.unravel_index(blocks, (2, 2))
    for ti, t in enumerate(ts):
        for Pi, P in enumerate([1, -1]):
            bjs = [(j + Pi) % 2 for j in bjs]
            bks = [(k + Pi) % 2 for k in bks]
            Ny = int(np.ceil((Ly + Pi) / 2))
            Nx = int(np.ceil((Lx + Pi) / 2))
            for J in range(Ny):
                for K in range(Nx):
                    for jj, kk in zip(bjs, bks):
                        j = 2 * J + P * jj
                        k = 2 * K + P * kk
                        if 0 <= j < Ly and 0 <= k < Lx:
                            Nj, u = get_U(j, k, tri_map[jj, kk][Pi])
                            state = mx.op_on_state(u, Nj, state)
                            state = depolarize(state, Nj, E)

            yield state


# A = np.array([[0,1],[1, 0]])
# plt.imshow(A)
# plt.show()
#
# js = [0,  0,  1,  1] + [0, 0, 1, 1] + [1, 0, 1, 0] + [1, 0, 1, 0]
# ks = [0,  1,  0,  1] + [1, 0, 1, 0] + [1, 1, 0, 0] + [0, 0, 1, 1]
#
#
# tri_map = {"00":[[0, 1],[1, 0]], "01":[[0,1], [0,1]], "10":[[1,0],[1,0]], "11":[[1,0],[0,1]]}
#
# for P in [1]:
#    for j, k in zip(js[::P], ks[::P]):
#        tri = tri_map[str(j)+str(k)]
#        if np.sum(A[tuple(tri)]) in [1, 2]:
#            A[j, k] = (A[j, k]+1)%2
#        plt.imshow(A, vmin=0, vmax=1)
#        plt.show()
#


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
    uid = sha1(dumps(name_dict, sort_keys=True).encode("utf-8")).hexdigest()
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
    rec = {task: ms.measures[task]["init"](
        params["L"], len(ts)) for task in tasks}
    rec.update({"ts": ts})
    # average of reduced density matricies
    if params["blocks"] != [0]:
        evolve = block_evolve
    else:
        evolve = parity_evolve
    for n in range(params["N"]):
        for ti, state in enumerate(evolve(**params)):
            for task in tasks:
                if task == "bipart":
                    bipart = ms.measures[task]["get"](state)
                    for l in range(params["L"] - 1):
                        rec[task][ti][l] += bipart[l] / params["N"]
                else:
                    rec[task][ti] += ms.measures[task]["get"](
                        state) / params["N"]
    return rec


def make_params_dict(params, L, Lx, T, dt, R, r, V, IC, BC, E, N):
    """ Explicit conversion of parameters to dictionary. Updates
        a base dictonary 'params' """
    p = copy(params)
    p.update(
        {
            "L": L,
            "Lx": Lx,
            "T": T,
            "dt": dt,
            "R": R,
            "r": r,
            "V": V,
            "IC": IC,
            "BC": BC,
            "E": E,
            "N": N,
        }
    )
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
    """ Repeat last element of shorter lists of params """
    lens = np.array([l for l in map(len, args)])
    ind = np.argmax(lens)
    longest = lens[ind]
    pads = longest - lens
    to_zip = [arg + [arg[-1]] * pad for arg, pad in zip(args, pads)]
    return [make_params_dict(params, *p) for p in zip_longest(*to_zip)]


# collect parameter list constructors into
# a dictonary of functions
params_list_map = {
    "product": product_params_list,
    "cycle": cycle_params_list,
    "repeat": repeat_params_list,
}

if __name__ == "__main__":
    L = 16
    Lx = 4
    Ly = int(L / Lx)
    T = 100
    dt = 1
    Rs = [1, 2, 4]
    r = 1

    fig, axs = plt.subplots(2, 2)
    Zgrids = np.zeros((len(Rs), T + 1, Ly, Lx))

    ims = []
    for ll, R in enumerate(Rs):
        lj, lk = np.unravel_index(ll, (2, 2))
        evo = parity_evolve(
            L,
            Lx,
            T,
            dt,
            R,
            r,
            "h",
            "f6-9",
            "1-00",
            #rods=[[1, 2, 3, 4], [1, 2, 3, 4]],
            tris=[[1, 3], [2, 4]],
            totalistic=True,
            hamiltonian=False,
        )
        for t, state in enumerate(evo):
            rhoj = ms.get_rhoj(state)
            Zgrids[ll, t, :, :] = ms.get_expectation(
                rhoj, mx.ops["Z"]).reshape(Ly, Lx)
        im = axs[lj, lk].imshow(Zgrids[ll, 0], vmin=-1, vmax=1, cmap="inferno")
        axs[lj, lk].axis("off")
        axs[lj, lk].set_title(f"R={R}")
        ims.append(im)
    axs[-1, -1].axis("off")

    def updatefig(i):
        fig.suptitle(f"t={i}")
        for ll, R in enumerate(Rs):
            z = Zgrids[ll, i]

            ims[ll].set_array(z)
        return ims

    ani = animation.FuncAnimation(
        fig, updatefig, frames=range(T + 1), interval=200, blit=False
    )
    plt.show()
    # ani.save('figures/animation/2d_triangles.mp4')

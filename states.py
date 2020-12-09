#!/usr/bin/python3
#
# =============================================================================
# This file enables the creation of quantum states. The make_state function
# takes a lattice size L and a state specification IC, which is either a string
# or a list of tuples. The List of tuples is for global superpositions: each
# tuple contains a coefficient and a state specification string.
#
# A state specification string (called spec string in the table below) starts
# with a single letter corresponding to a function in this file (it's a key in
# the dictionary called smap).  Lowercase keys are for separable states
# while capital keys are for entangled states. Everything after that first
# letter is a configuration. Underscores separate different config sections and
# dashes separate params within a config section:
#
#   function  | key |             config                | spec. string example
# ------------+-----+-----------------------------------+------------------------
#             |     |                                   | 'f0-3_t90-p0_t45-p180'
#     fock    |  f  |<i-j-k...>_t<th>-p<ph>_t<th>-p<ph> | 'f2_t90-p90'
#             |     |                                   | 'f0-2-4-6'
# ------------+-----+-----------------------------------+------------------------
#             |     |                                   | 'st90-P1'
#  spin_wave  |  s  |    T<n> OR t<th>-P<m> OR p<ph>    | 'sT2-p30'
#             |     |                                   | 'sT2-P1'
# ------------+-----+-----------------------------------+------------------------
#  rand_state |  r  | <p>-<s>_t<th>-p<ph>_t<th>-p<ph>   | 'r75_t45-p90'
#             |     |                                   | 'r5-234_t45-p90'
# ------------+-----+-----------------------------------+------------------------
#  rand_plus  |  p  |<i>_<p>-<s>_t<th>-p<ph>_t<th>-p<ph>| 'p10_75-123
#             |     |                                   | 'p9_5-123'
# ------------+-----+-----------------------------------+------------------------
#   doublon   |  d  |     t<th>-p<ph>_t<th>-p<ph>       | 'd'  = |1010...>
#             |     |                                   | 'dt0-p0_t180-p0'
# ------------+-----+-----------------------------------+------------------------
#  cluster    |  C  |          <m>-<n>_<th>             | 'C1-19_45'
# ------------+-----+-----------------------------------+------------------------
#  rand_throw |  R  |               <s>                 | 'R234'
# ------------+-----+-----------------------------------+------------------------
# porter_Thomas|  P  |               <s>                 | 'P345'
# ------------+-----+-----------------------------------+------------------------
#     Bell    |  B  |            <j-k>_<b>              | 'B0-1_3'
# ------------+-----+-----------------------------------+------------------------
#     GHZ     |  G  |                NA                 | 'G'
# ------------+-----+-----------------------------------+------------------------
#      W      |  W  |                NA                 | 'W'
# ------------+-----+-----------------------------------+------------------------
#             |     |                                   | 'c1_f0'
#   center    |  c  |       <Lc>_<spec string>          | 'c4_W'
#             |     |                                   | 'c2_r5'
#
# superposition:
# --------------
#
# Superpose the states of any number of spec strings using a list of tuples.
# Each tuple has two elements, the first is a coefficient and the second is the
# spec string, e.g.
#
#            [(coeff_0, spec string_0), ..., (coeff_N, spec string_N)].
#
# For example, the follwoing is equivalent to 'B0-1_2'
#
#                   [(1.0/sqrt(2), 'f0'), (1.0/sqrt(2), 'f1')]]
#
# Conventions: |0> = (1, 0) is at the top of the Bloch sphere. Is an 'excitation'
#              |1> = (0, 1) is at the bottom, and is the 'background'
#              th = theta is the polar angle in degrees winds from 0 to 90.
#              ph = phi is the azimuthal angle in degrees winds from 0 to 180.
#
# Description of config sections:
#   + fock: f<i-j-k...>_t<th>-p<ph>_t<th>-p<ph> a fock state of qubits
#       + section 1, <i-j-k...>: site indices of excitation
#       + section 2, t<th>-p<ph>: theta and phi in deg on Bloch sphere describing
#                                excitation qubits (default t180_p0 if not given)
#       + section 3, t<th>-p<ph>: theta and phi in deg on Bloch sphere describing
#                                background qubits (default t0_p0 if not given)
#
#   + spin_wave: st/T<th>-p/P<m> fock states with twists in theta and/or phi
#     across the lattice
#       + section 1, t<th> (p<ph>) holds theta (phi) constant at th (ph)
#                    T<n> (P<m>) winds theta (phi) n (m) times
#
#   + rand_state: r<p>-<s>_t<th>-p<ph>_t<th>-p<ph> a random fock state:
#       + section 1, <p>: probability of excitation at each site expressed as an
#                         int, e.g.  p=75 means prob of 3/4 for an excitation
#                    <s>: OPTIONAL - seed for random number generator
#       + sections 2 and 3, same as sections 2 and 3 in fock above
#
#   + rand_plus: p<i>_<p>-<s> a random fock state with site i set excited:
#       + section 1, <i>: site index of excitation
#       + section 2, <p>: probability of excitation at each site expressed as an
#                         int, e.g.  p=75 means prob of 3/4 for an excitation
#                    <s>: OPTIONAL - seed for random number generator
#       + sections 3 and 4, same as sections 2 and 3 in fock above
#
#   + cluster: C<m>-<n>_<th> cluster state based on grid graph:
#       + section 1, <m>: number of rows in a cluster state grid
#                    <n>: number of columns in a cluster state grid
#       + section 2  <th>: phase gate angle for application to edges grid graph
#
#   + rand_throw: R<s> flat distributions for random qubits:
#       + section 1, <s>: seed for random number generator
#
#   + porter_thomas: P<s> Porter-Thomas amplitudes and flat phase distributions
#                 of random qubits:
#       + section 1, <s>: seed for random number generator
#
#   + Bell: B<j-k>_<b> a member of the Bell basis embedded in the lattice
#       + section 1 <j-k>: two site indices to share the bell state
#       + section 2 <b>: specify which Bell state according to b=0, 1, 2, or 3.
#                       b=0 : 1/sqrt 2 (|00>+|11>)
#                       b=1 : 1/sqrt 2 (|00>-|11>)
#                       b=2 : 1/sqrt 2 (|01>+|10>)
#                       b=3 : 1/sqrt 2 (|01>-|10>)
#
#   + center: c<Lc>-<spec string> embed any IC into the center of the lattice
#       + section 1 <Lc>, the length of the central region. <IC> some IC spec
#       other than 'c'
#
#
# By Logan Hillberry
# =============================================================================


from cmath import sqrt, sin, cos, exp, pi
import numpy as np
from numpy.random import random
from matrix import listkron, dec_to_bin, op_on_state
from scipy.stats import rv_continuous

# basis vectors
# es = equal superposition of |0> and |1>
bvecs = {
    '0': np.array([1.0, 0.0], dtype=complex),
    '1': np.array([0.0, 1.0], dtype=complex),
    'es': np.array([1.0 / sqrt(2), 1.0 / sqrt(2)], dtype=complex)
    }


def edit_small_vals(mat, tol=1e-14, replacement=0.0):
    if type(mat) is not np.ndarray:
        mat = np.asarray(mat)
    mat[np.abs(mat) <= tol] = replacement
    return mat


def qubit(t, p):
    t = pi / 180.0 * t
    p = pi / 180.0 * p
    return cos(t / 2.0) * bvecs['0'] + exp(1j * p) * sin(t / 2.0) * bvecs['1']


def make_config_dict(config):
    config_list = config.split('_')
    n = len(config_list)
    ex_list = [int(ex) for ex in config_list[0].split('-') if ex != '']
    if n == 1:
        config_dict = {
            'ex_list': ex_list,
            'ex_config': {'t': 180, 'p': 0},
            'bg_config': {'t': 0, 'p': 0}
            }
    elif n == 2:
        ex_config = {ang[0]: eval(ang[1:]) for
                     ang in config_list[1].split('-')}
        config_dict = {'ex_list': ex_list,
                       'ex_config': ex_config,
                        'bg_config': {'t': 0, 'p': 0}}
    elif n == 3:
        ex_config = {ang[0]: eval(ang[1:]) for
                     ang in config_list[1].split('-')}
        bg_config = {ang[0]: eval(ang[1:]) for
                     ang in config_list[2].split('-')}
        config_dict = {'ex_list': ex_list,
                       'ex_config': ex_config,
                       'bg_config': bg_config}
    return config_dict

def gridedge(m, n):
    E = np.zeros((2 * n * m  - m - n, 2), dtype=np.int32)
    for i in range(n * m):
        if i % n != n - 1:
            E[i - i // n, 0], E[i - i // n, 1] = i, i + 1
        if i < (m - 1) * n:
            E[(n - 1) * m + i, 0], E[(n - 1) * m + i, 1] = i, i + n
    return E

def cluster(L, config):
    try:
        mn, ph = config.split("_")
        ph = float(ph) * np.pi / 180
        m, n = map(int, mn.split("-"))
    except:
        ph = 45.0 * np.pi / 180
        m, n = map(int, config.split("-"))
    assert L == m * n
    E = gridedge(m, n)
    Cphase = np.eye(4, dtype=np.complex128)
    Cphase[3, 3] = np.exp(1j * ph)
    # equal superposition for all qubits
    state = listkron([bvecs["es"] for _ in range(m * n)])
    # apply phase gate according to edges of graph
    for e in E:
        state = op_on_state(Cphase, e, state)
    return state


def cluster_all(L, config):
    state = np.zeros(2**L, dtype=np.complex128)
    for k in range(2**L):
        b = dec_to_bin(k, L)
        c = 1.0
        for j in range(L - 1):
            c *= (-1) ** (b[j] * b[j + 1])
        state += c * listkron([bvecs[str(bj)] for bj in b])
    return state / 2**(L / 2)


def fock(L, config):
    config_dict = make_config_dict(config)
    ex_list = np.array(config_dict['ex_list'])
    qubits = np.array([qubit(**config_dict['bg_config'])] * L)
    for ex in ex_list:
        qubits[ex, :] = qubit(**config_dict['ex_config'])

    state = listkron(qubits)
    return state


def doublon(L, config):
    try:
        D = config.split('_')[0]
        config = config[len(D):]
        D = int(D)
    except IndexError:
        D = 2
    if D == 'd':
        D = 2
    else:
        D = int(D)
    rel = L // 2
    ex_list= [rel + i for i in range(0, L, D) if (rel + i) < L]
    ex_list += [rel - i for i in range(0, L, D) if (rel - i) >= 0]
    ex_list = list(set(ex_list))
    print(ex_list)
    fock_config = '-'.join([str(i) for i in ex_list])
    fock_config = ''.join([fock_config, config])
    return fock(L, fock_config)


def rand_n(L, config):
    csplit = config.split('_')
    ns = csplit[0]
    nssplit = ns.split('-')
    n = int(nssplit[0])
    if len(nssplit) > 1:
        s = int(nssplit[1]) * L * n
    else:
        s = None
    random.seed(s)
    fock_config = '-'.join([str(i) for i in random.sample(range(L), n)])
    if len(csplit) > 1:
        config = csplit[1:]
        fock_config = ''.join([fock_config, config])
    return fock(L, fock_config)


def GHZ(L, config):
    s1 = [
     '1'] * L
    s2 = ['0'] * L
    state = 1.0 / sqrt(2.0) * (listkron([bvecs[key] for key in s1]) + listkron([bvecs[key] for key in s2]))
    return state


def W(L, config):
    return 1.0 / sqrt(L) * sum((fock(L, str(j)) for j in range(L)))


def Bell(L, config):
    jk, typ = config.split('_')
    j, k = jk.split('-')
    coeff = 1.0
    if typ in ('1', '3'):
        coeff = -1.0
    if typ in ('2', '3'):
        state = 1 / sqrt(2) * (fock(L, j) + coeff * fock(L, k))
    else:
        if typ in ('0', '1'):
            state = 1 / sqrt(2) * (listkron([qubit(0.0, 0.0)] * L) + coeff * fock(L, jk))
    return state


def Bell_array(L, config):
    try:
        bell_type = config[0]
    except:
        bell_type = '0'
    ic = 'B0-1_{}'.format(bell_type)
    singlet = make_state(2, ic)
    if L % 2 == 0:
        state = listkron([singlet] * int(L / 2))
    else:
        state = listkron([singlet] * int((L - 1) / 2) + [bvecs['0']])
    return state


def spin_wave(L, config):
    Tt, Pp = config.split('-')
    ang_dict = {
        'T': np.linspace(0.0, pi * float(Tt[1:]), L),
        't': [float(Tt[1:])] * L,
        'P': np.linspace(0.0, 2 * pi * float(Pp[1:]), L),
        'p': [float(Pp[1:])] * L
        }
    th_list = ang_dict[Tt[0]]
    ph_list = ang_dict[Pp[0]]
    qubit_list = [0.0] * L
    for j, (th, ph) in enumerate(zip(th_list, ph_list)):
        qubit_list[j] = qubit(th, ph)

    return listkron(qubit_list)


def rand_state(L, config):
    ps_qex_qbg_conf = config.split('_')
    ps = ps_qex_qbg_conf[0].split('-')
    p = float('.' + ps[0])
    s = None
    if len(ps) == 2:
        s = ps[1]
    if len(ps_qex_qbg_conf) == 1:
        state_dict = {'ex': bvecs['1'], 'bg': bvecs['0']}
    if len(ps_qex_qbg_conf) == 2:
        ex_th, ex_ph = ps_qex_qbg_conf[1].split('-')
        ex_th = float(ex_th[1:])
        ex_ph = float(ex_ph[1:])
        state_dict= {'ex': qubit(ex_th, ex_ph), 'bg': bvecs['0']}
    if len(ps_qex_qbg_conf) == 3:
        ex_th, ex_ph = ps_qex_qbg_conf[1].split('-')
        ex_th = float(ex_th[1:])
        ex_ph = float(ex_ph[1:])
        bg_th, bg_ph = ps_qex_qbg_conf[2].split('-')
        bg_th = float(bg_th[1:])
        bg_ph = float(bg_ph[1:])
        state_dict= {'ex': qubit(ex_th, ex_ph), 'bg': qubit(bg_th, bg_ph)}
    prob = [p, 1.0 - p]
    if s is not None:
        np.random.seed(int(s))
    distrib = np.random.choice(['ex', 'bg'], size=L, p=prob)
    return listkron([state_dict[i] for i in distrib])


def rand_plus(L, config):
    exs_ps_qex_qbg_conf = config.split('_')
    exs = exs_ps_qex_qbg_conf[0].split('-')
    exs = np.array([int(ex) for ex in exs])
    ps = exs_ps_qex_qbg_conf[1].split('-')
    p = float('.' + ps[0])
    prob = [p, 1.0 - p]
    s = None
    if len(ps) == 2:
        s = ps[1]
    if s is not None:
        np.random.seed(int(s))
    if len(exs_ps_qex_qbg_conf) == 2:
        state_dict = {'ex': bvecs['1'], 'bg': bvecs['0']}
    if len(exs_ps_qex_qbg_conf) == 3:
        ex_th, ex_ph = exs_ps_qex_qbg_conf[2].split('-')
        ex_th = float(ex_th[1:])
        ex_ph = float(ex_ph[1:])
        state_dict= {'ex': qubit(ex_th, ex_ph), 'bg': bvecs['0']}
    if len(exs_ps_qex_qbg_conf) == 4:
        ex_th, ex_ph = exs_ps_qex_qbg_conf[2].split('-')
        ex_th = float(ex_th[1:])
        ex_ph = float(ex_ph[1:])
        bg_th, bg_ph = exs_ps_qex_qbg_conf[3].split('-')
        bg_th = float(bg_th[1:])
        bg_ph = float(bg_ph[1:])
        state_dict= {'ex': qubit(ex_th, ex_ph), 'bg': qubit(bg_th, bg_ph)}
    distrib = np.random.choice(['ex', 'bg'], size=L, p=prob)
    distrib[exs] = 'ex'
    state = listkron([state_dict[q] for q in distrib])
    return state


def random_throw(L, config):
    np.random.seed(None)
    if len(config) > 0:
        np.random.seed(int(config))
    state = np.random.rand(2 ** L, 2) - 0.5
    state = (state.view(dtype=np.complex128))[(Ellipsis, 0)]
    state = state / np.sqrt(np.conj(state).dot(state)).real
    return state

class Porter_Thomas(rv_continuous):
    def _pdf(self, x, D):
        return 2 * x * D * np.exp(-D * x * x)


def porter_thomas1(L, config):
    np.random.seed(None)
    if len(config) > 0:
        np.random.seed(int(config))
    PT = Porter_Thomas(a=0, b=np.inf, name='Porter_Thomas')
    state = PT.rvs(size=2**L, D=2**L) * np.exp((np.random.random(2**L)) * 2 * np.pi * 1j)
    return state / np.sqrt(np.conj(state).dot(state)).real


def porter_thomas(L, config):
    np.random.seed(None)
    if len(config) > 0:
        np.random.seed(int(config))
    state = np.random.normal(0, 1, (2 ** L, 2))
    state = state[:, 0] + 1j * state[:, 1]
    state = state / np.sqrt(np.conj(state).dot(state)).real
    return state


def center(L, config):
    Lcent = config.split('_')[0]
    cent_IC = config.split('_')[1:]
    cent_IC = '_'.join(cent_IC)
    len_cent = int(Lcent)
    len_back = L - len_cent
    len_L = int(len_back / 2)
    len_R = len_back - len_L
    if cent_IC[0] == 'f':
        config_dict = make_config_dict(cent_IC[1:])
    else:
        config_dict = make_config_dict('0')
    bg_qubit = qubit(**config_dict['bg_config'])
    left = listkron([bg_qubit for _ in range(len_L)])
    cent = make_state(len_cent, cent_IC)
    right = listkron([bg_qubit for _ in range(len_R)])
    if len_back == 0:
        return cent
    elif len_back == 1:
        return listkron([cent, right])
    return listkron([left, cent, right])


smap = {
 'f': fock,
 'd': doublon,
 'n': rand_n,
 'c': center,
 's': spin_wave,
 'r': rand_state,
 'p': rand_plus,
 'R': random_throw,
 'P': porter_thomas,
 'C': cluster,
 'G': GHZ,
 'W': W,
 'B': Bell,
 'S': Bell_array,
 }


def make_state(L, IC):
    if type(IC) == str:
        name = IC[0]
        config = IC[1:]
        state = smap[name](L, config)
    else:
        if type(IC) == list:
            state = np.zeros(2 ** L, dtype=complex)
            for s in IC:
                coeff = s[0]
                name = s[1][0]
                config = s[1][1:]
                state += coeff * smap[name](L, config)

    state = edit_small_vals(state.real) + 1j * edit_small_vals(state.imag)
    state = state.astype(np.complex128)
    return state

if __name__ == "__main__":
    import measures as ms
    from matrix import ops
    import matplotlib.pyplot as plt
    L = 20
    ICs = [f"d{i}" for i in range(1, L // 2 + 4)]
    Zs = []
    for IC in ICs:
        state = make_state(L, IC)
        rhoj = ms.get_rhoj(state)
        Z = ms.get_expectation(rhoj, ops["Z"])
        Zs.append(Z)
    plt.imshow(Zs)
    plt.show()

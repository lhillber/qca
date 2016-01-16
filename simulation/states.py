#!/usr/bin/python3

# =============================================================================
# This file is used to store useful matricies as a global constans in a
# dictionary. It also enables the creation of quantum states. These are used as
# initial states in the qca project. the make_state function takes a littice
# size L and a state spec IC, which is either a string or a list of tuples. The
# List of tuples is for global superpositions: each tuple is a coefficient and a
# string
# Examples are provided below in the default behavior of this file.
# =============================================================================


from cmath import sqrt, sin, cos, exp, pi
import numpy as np
import simulation.matrix as mx

# Global constants
# ================
# dictionary of local operators and local basis (b for basis)
# -----------------------------------------------------------
pauli = {
        '0' : np.array( [[1.0,  0.0 ],[0.0,   1.0]], dtype=complex ),
        '1' : np.array( [[0.0,  1.0 ],[1.0,   0.0]], dtype=complex ),
        '2' : np.array( [[0.0, -1.0j],[1.0j,  0.0]], dtype=complex ),
        '3' : np.array( [[1.0,  0.0 ],[0.0 , -1.0]], dtype=complex )
        }

ops = {
        'H' : 1.0 / sqrt(2.0) * \
              np.array( [[1.0,  1.0 ],[1.0,  -1.0]], dtype=complex),

        'I' : np.array( [[1.0,  0.0 ],[0.0,   1.0]], dtype=complex ),
        'X' : np.array( [[0.0,  1.0 ],[1.0,   0.0]], dtype=complex ),
        'Y' : np.array( [[0.0, -1.0j],[1.0j,  0.0]], dtype=complex ),
        'Z' : np.array( [[1.0,  0.0 ],[0.0 , -1.0]], dtype=complex ),

        'S' : np.array( [[1.0,  0.0 ],[0.0 , 1.0j]], dtype=complex ),
        'T' : np.array( [[1.0,  0.0 ],[0.0 , exp(1.0j*pi/4.0)]], dtype=complex ),

        '0' : np.array( [[1.0,   0.0],[0.0,   0.0]], dtype=complex ),
        '1' : np.array( [[0.0,   0.0],[0.0,   1.0]], dtype=complex ),
      }

brhos = {
        '0' : np.array( [[1.0,   0.0],[0.0,   0.0]], dtype=complex ),
        '1' : np.array( [[0.0,   0.0],[0.0,   1.0]], dtype=complex ),
        }

bvecs = {
        '0'  : np.array( [1.0, 0.0], dtype=complex ),
        '1'  : np.array( [0.0, 1.0], dtype=complex ),

        'es' : np.array( [1./sqrt(2), 1./sqrt(2)], dtype=complex ),
        }

# Initial State Creation
# ======================

# qubit on the block sphere. th is from vertical and ph is from x around z.
# th and ph are expected in degrees
def qubit(t, p):
    t = pi/180.0 * t
    p = pi/180.0 * p
    return cos(t/2.0) * bvecs['0'] + exp(1j*p) * sin(t/2) * bvecs['1']

def make_config_dict(config):
    config_list = config.split('_')
    n = len(config_list)
    if n == 1:
        conf_dict = {'ex_list' : [int(ex) for ex in config.split('-')],
                'ex_config' : {'t':180, 'p':0},
                'bg_config' : {'t':0, 'p':0} }

    elif n == 2:
        ex_list = [int(ex) for ex in config_list[0].split('-')]
        ex_config = {ang[0] : eval(ang[1:]) for ang in config_list[1].split('-')}
        conf_dict = {'ex_list': ex_list,
                'ex_config' : ex_config,
                'bg_config' : {'t':0, 'p':0} }

    elif n == 3:
        ex_list = [int(ex) for ex in config_list[0].split('-')]
        ex_config = {ang[0] : eval(ang[1:]) for ang in config_list[1].split('-')}
        bg_config = {ang[0] : eval(ang[1:]) for ang in config_list[2].split('-')}
        conf_dict = {'ex_list': ex_list,
                'ex_config' : ex_config,
                'bg_config' : bg_config}
    return conf_dict


# Create Fock state
# -----------------
def fock(L, config):
    config_dict = make_config_dict(config)
    ex_list = np.array(config_dict['ex_list'])
    qubits = np.array([qubit(**config_dict['bg_config'])]*L)
    for ex in ex_list:
        qubits[ex, ::] = qubit(**config_dict['ex_config'])
    state = mx.listkron(qubits)
    return state


# Create GHZ state
# ----------------
def GHZ (L, congif):
    s1=['1']*(L)
    s2=['0']*(L)
    return (1.0/sqrt(2.0)) \
            * ((mx.listkron([bvecs[key] for key in s1]) \
                + mx.listkron([bvecs[key] for key in s2])))

# Create W state
# --------------
def W(L, config):
    return 1.0/sqrt(L) * sum(fock(L, str(j)) for j in range(L))

# Create a state with GHZ-type entanglement.
# Reduces to 1/sqrt(2) (|00> + |11>) in L = 2 limit
# ------------------------------------------------------
def entangled_list(L, config):
    js = map(int, config.split('_'))
    js = [L - j - 1 for j in js]
    dec = sum((2**x for x in js))
    return 1./sqrt(2) * (fock(L, 0) + fock(L, dec))

# Create a state with any of the obove states
# embeded in the center of the lattice
# -------------------------------------------
def center(L, config):
    len_cent = int(config[0])
    len_back = L - len_cent
    len_L = int(len_back/2)
    len_R = len_back - len_L
    cent_IC = [(config[1:], 1.0)]
    left = fock(len_L, 0)
    cent = make_state(len_cent, cent_IC)
    right = fock(len_R, 0)
    if len_back == 0:
        return cent
    elif len_back == 1:
        return mx.listkron([cent, right])
    else:
        return mx.listkron([left, cent, right])

def spin_wave(L, config):
    Tt, Pp = config.split('-')
    ang_dict = {'T' : np.linspace(0.0,  pi*float(Tt[1:]), L),
                't' : [float(Tt[1:])*pi/180.0]*L,
                'P' : np.linspace(0.0, 2*pi*float(Pp[1:]), L),
                'p' : [float(Pp[1:])*pi/180.0]*L,
                    }
    th_list = ang_dict[Tt[0]]
    ph_list = ang_dict[Pp[0]]
    qubit_list = [0.0]*L
    for j, (th, ph) in enumerate(zip(th_list, ph_list)):
        qubit_list[j] = qubit(th, ph)
    return mx.listkron(qubit_list)

# Create a state with random single-qubit states
# ----------------------------------------------
def rand_state(L, config):

    p_qex_qbg_conf = config.split('_')
    p = float('.'+p_qex_qbg_conf[0])

    if len(p_qex_qbg_conf)==1:
        state_dict = {'ex':bvecs['1'], 'bg':bvecs['0']}

    if len(p_qex_qbg_conf)==2:
        ex_th, ex_ph = p_qex_qbg_conf[1].split('-')
        ex_th = pi/180.0*float(ex_th[1:])
        ex_ph = pi/180.0*float(ex_ph[1:])

        state_dict = {'ex':qubit(ex_th, ex_ph), 'bg':bvecs['0']}

    if len(p_qex_qbg_conf)==3:
        ex_th, ex_ph = p_qex_qbg_conf[1].split('-')
        ex_th = pi/180.0*float(ex_th[1:])
        ex_ph = pi/180.0*float(ex_ph[1:])

        bg_th, bg_ph = p_qex_qbg_conf[2].split('-')
        bg_th = pi/180.0*float(bg_th[1:])
        bg_ph = pi/180.0*float(bg_ph[1:])

        state_dict = {'ex':qubit(ex_th, ex_ph), 'bg':qubit(bg_th, bg_ph)}

    prob = [p, 1.0 - p]

    distrib = np.random.choice(['ex','bg'], size=L, p=prob)
    return mx.listkron([state_dict[i] for i in distrib])


# kronecker the states in the distribution and return

# Make the specified state
# ------------------------
smap = { 'f' : fock,
         'c' : center,
         's' : spin_wave,
         'r' : rand_state,
         'G' : GHZ,
         'W' : W,
         'E' : entangled_list }

def make_state (L, IC):
    if type(IC) == str:
        name = IC[0]
        config = IC[1:]
        state = smap[name](L, config)

    elif type(IC) == list:
        state = np.zeros(2**L, dtype = complex)
        for s in IC:
                name = s[0][0]
                config = s[0][1:]
                coeff = s[1]
                state = state + coeff * smap[name](L, config)
    return state


if __name__ == '__main__':
    import simulation.measures as ms
    import simulation.states as ss


    L = 8
    T = 0
    IC = 'f1-4_t90-p0'

    state = make_state(L, IC)
    rj = np.asarray([mx.rdms(state, [j]) for j in range(L)])
    zj = np.asarray([np.trace(r.dot(ops['Z'])).real for r in rj])
    print(zj)

    '''
    r0k = np.asarray([mx.rdms(state, [0,k]) for k in range(1, L)])
    z0k = np.asarray([1.0]+[np.trace(r.dot(np.kron(ops['Z'], ops['Z']))) for r in r0k])

    g = z0k - zj[0]*zj
    print(zj)
    print(g)
    print()

    one_site = np.zeros((T+1, L, 2, 2), dtype = complex)
    two_site = np.zeros((T+1, L, L, 4, 4), dtype = complex)
    for t in range(T+1):
        for j in range(L):
            rtj = mx.rdms(state, [j])
            one_site[t, j] = rtj[::]
            for k in range(j+1, L):
                rtjk = mx.rdms(state, [j, k])
                two_site[t, j, k] = two_site[t, k, j] = rtjk[::]

    moment_dict = ms.moments_calc(one_site, two_site, L, T)
    g_dict = ms.g_calc(moment_dict, L, T)

    print('gzz')
    print(ms.get_row_vecs(g_dict['gzz'], j=0))
    print()
    print()
    L_list = [4]

    # spin down (or |1>) at sites 0 and 2 spin up (or |0>) at 1 and 3
    IC_list = ['l0_2']

    for L, IC in zip(L_list, IC_list):
        print ("L = ", str(L), " IC = ", str(IC) )
        print('Expect spin down (or |1>) at sites 0 and 2. Spin up (or |0>) at 1 and 3')

        print()

        print('state vector:')
        state = make_state(L, IC)
        print(state)

        print()

        # reduced density matrix for each site calculated from the state
        rho_list = [mx.rdms(state, [j]) for j in range(L)]

        # measure z projection at each site. Take real part because measurements
        # of Hermitian ops always give a real result
        meas_list = [np.trace(rho.dot(ss.ops['Z'])).real for rho in rho_list ]

        print('expectation value along Z axis:')
        print(meas_list)
    '''

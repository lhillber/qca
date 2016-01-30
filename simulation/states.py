#!/usr/bin/python3

# =============================================================================
# This file is used to store useful matricies as a global constans in a
# dictionary. It also enables the creation of quantum states. These are used as
# initial states in the qca project. the make_state function takes a littice
# size L and a state spec IC, which is either a string or a list of tuples. The
# List of tuples is for global superpositions: each tuple is a coefficient and a
# spec string.
#
# the spec string starts with a single letter corresponding to a function in
# this file (its a key for the dictionary smap below). Everything after that
# first letter is a configuration. Underscores separate different config
# sections and dashes separate data within a config section.
#
#   function  | key |             config               |       example
# ----------------------------------------------------------------------------
#             |     |                                  | 'f0-3_t90-p0_t45_p180'
#     fock    |  f  |<i-j-k...>_t<th>-p<ph>_t<th>-p<ph>| 'f2_t90-p90'
#             |     |                                  | 'f0-2-4-6'
# ----------------------------------------------------------------------------
#  rand_state |  r  |     <p>_t<th>-p<ph>_t<th>-p<ph>  | 'r75_t45_p90'
# ----------------------------------------------------------------------------
#             |     |                                  | 'st90-P1'
#  spin_wave  |  s  |    T<n> OR t<th>-P<m> OR p<ph>   | 'sT2-p30'
#             |     |                                  | 'sT2-P1'
# ----------------------------------------------------------------------------
#     Bell    |  B  |              <j-k>_<b>           | 'B0-1_3'
# ----------------------------------------------------------------------------
#     GHZ     |  G  |                NA                | 'G'
# ----------------------------------------------------------------------------
#      W      |  W  |                NA                | 'W'
# ----------------------------------------------------------------------------
#             |     |                                  | 'c2r'
#   center    |  c  |           <Lc><IC>               | 'c4W'
#             |     |                                  | 'c1f0'
# 
# Description of config sections:
#   + fock: a fock state of qubits
#       + section 1, <i-j-k...>: site indices of excitation
#       + section 2, t<th>-p<ph>: theta and phi in deg on Bloch sphere describing
#                                excitation qubits (default t180_p0 if not given)
#       + section 3, t<th>-p<ph>: theta and phi in deg on Bloch sphere describing
#                                bacground qubits (default t0_p0 if not given)

#   + rand_state: a random fock state:
#       + section 1, <p>: probability of excitation at each site expressed as an
#                         int. That is, p=75 means prop of 3/4 for an excitation
#       + sections 2 and 3, same ase sections 2 and 3 above
#
#   + spin_wave: fock states with twists in theata and/or phi across the lattice
#       + section 1, t<th> (p<ph>) holds theta (phi) constant at th (ph)
#                    T<n> (P<m>) twists theta (phi) n (m) times
#
#   + Bell: a member of the Bell basis embeded in the lattice
#       + section 1 <j-k>, two site indices to share the bell state
#       + section 2 <b>, specify which Bell state according to b. (b : state)
#                       0 : 1/|/2 (|00>+|11>)
#                       1 : 1/|/2 (|00>-|11>)
#                       2 : 1/|/2 (|01>+|10>)
#                       3 : 1/|/2 (|01>-|10>)
#
#   + center: embed any IC into the center of the lattice
#       + section 1 <LC>, the length of the central region. <IC> some other
#         IC spec
#
#
# By Logan Hillberry
# =============================================================================


from cmath import sqrt, sin, cos, exp, pi
import numpy as np
import simulation.matrix as mx

# Global constants
# ================
# dictionary of local operators and local basis (b for basis)
# -----------------------------------------------------------

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
    state = (1.0/sqrt(2.0)) \
            * (mx.listkron([bvecs[key] for key in s1]) \
                + mx.listkron([bvecs[key] for key in s2]))

    return state


# Create W state
# --------------
def W(L, config):
    return 1.0/sqrt(L) * sum(fock(L, str(j)) for j in range(L))

# Create a state with GHZ-type entanglement.
# Reduces to 1/sqrt(2) (|00> + |11>) in L = 2 limit
# ------------------------------------------------------
def Bell(L, config):
    jk, typ = config.split('_')
    j, k = jk.split('-')
    coeff = 1.0
    if typ in ('1', '3'):
        coeff = -1.0
    if typ in ('2', '3'):
        state = 1/sqrt(2)*(fock(L, j) + coeff*fock(L, k))

    elif typ in ('0', '1'):
        state = 1/sqrt(2)*(mx.listkron([qubit(0.0, 0.0)]*L) + coeff*fock(L, jk))
    return state

# Create a state with any of the obove states
# embeded in the center of the lattice
# -------------------------------------------
def center(L, config):
    Lcent = config.split('_')[0]
    cent_IC = config.split('_')[1::]
    cent_IC = '_'.join(cent_IC)
    print(cent_IC)
    len_cent = int(Lcent)
    len_back = L - len_cent
    len_L = int(len_back/2)
    len_R = len_back - len_L
    if cent_IC[0] == 'f':
        config_dict = make_config_dict(cent_IC[1::])
    else:
        config_dict = make_config_dict('0')
    bg_qubit = qubit(**config_dict['bg_config'])
    left = mx.listkron([bg_qubit for _ in range(len_L)])
    cent = make_state(len_cent, cent_IC)
    right = mx.listkron([bg_qubit for _ in range(len_R)])
    if len_back == 0:
        return cent
    elif len_back == 1:
        return mx.listkron([cent, right])
    else:
        return mx.listkron([left, cent, right])

def spin_wave(L, config):
    Tt, Pp = config.split('-')
    ang_dict = {'T' : np.linspace(0.0,  pi*float(Tt[1:]), L),
                't' : [float(Tt[1:])]*L,
                'P' : np.linspace(0.0, 2*pi*float(Pp[1:]), L),
                'p' : [float(Pp[1:])]*L,
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
        ex_th = float(ex_th[1:])
        ex_ph = float(ex_ph[1:])

        state_dict = {'ex':qubit(ex_th, ex_ph), 'bg':bvecs['0']}

    if len(p_qex_qbg_conf)==3:
        ex_th, ex_ph = p_qex_qbg_conf[1].split('-')
        ex_th = float(ex_th[1:])
        ex_ph = float(ex_ph[1:])

        bg_th, bg_ph = p_qex_qbg_conf[2].split('-')
        bg_th = float(bg_th[1:])
        bg_ph = float(bg_ph[1:])

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
         'B' : Bell }

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
    import simulation.time_evolve as te
    import matplotlib.pyplot as plt

    ND_list = []
    Y_list = []
    CC_list = []
    IPR_list = []
    L_list = range(2, 15)
    for L in L_list:
        T = 0
        IC = 'B0-'+str(L-1)+'_3'
        state = make_state(L, IC)
        one_site = np.zeros((T+1, L, 2, 2), dtype = complex)
        s = np.zeros((T+1, L), dtype = complex)
        two_site = np.zeros((T+1, L, L, 4, 4), dtype = complex)
        for t in range(T+1):
            for j in range(L):
                rtj = mx.rdms(state, [j])
                stj = ms.vn_entropy(rtj)
                one_site[t, j] = rtj[::]
                s[t, j] = stj
                for k in range(j+1, L):
                    rtjk = mx.rdms(state, [j, k])
                    two_site[t, j, k] = two_site[t, k, j] = rtjk[::]
        results = {'one_site': one_site, 'two_site' : two_site, 's':s}

        results['IPR'] = te.inv_participation_ratio(L, state)

        ms.moments_calc(results, L, T)
        ms.g_calc(results, L, T)
        ms.mtjk_calc(results, L, T)
        ms.nm_calc(results, L, T)

        ND_list = ND_list + list(results['ND'])
        CC_list = CC_list + list(results['CC'])
        Y_list  = Y_list  + list(results['Y'])
        IPR_list = IPR_list + list([results['IPR']])

    plt.plot(L_list, ND_list)
    plt.plot(L_list[1::], CC_list[1::])
    plt.plot(L_list, Y_list)
    #plt.plot(L_list, IPR_list)
    plt.show()

    '''
    L_list = [4]

    # spin down (or |1>) at sites 0 and 2 spin up (or |0>) at 1 and 3
    IC_list = ['f0-2']

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

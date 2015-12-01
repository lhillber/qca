#!/usr/bin/python3

from cmath import sqrt, sin, cos, exp, pi
import numpy as np
import matrix as mx

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
   
# Create Fock state
# -----------------
def fock (L, config, zero = '0', one = '1'):
    dec = int(config)
    state = [el.replace('0', zero).replace('1', one)
            for el in list('{0:0b}'.format(dec).rjust(L, '0'))]
    return mx.listkron([bvecs[key] for key in state])

# Create state with set of alive sites
# -------------------------------------
def alive_list(L, config):
    js = map(int, config.split('_'))
    js = [L - j - 1 for j in js]
    dec = sum((2**x for x in js))
    return fock(L, dec)

# Create state with one live sites
# --------------------------------
def one_alive (L, config):
    dec = 2**int(config)
    return fock(L, dec)

# Create states with equal superposition locally
# ----------------------------------------------
def es_list(L, config):
    js = map(int, config.split('_'))
    js = [L - j - 1 for j in js]
    dec = sum((2**x for x in js))
    return fock(L, dec, one='es')

# Create state with all sites living
# ----------------------------------
def all_alive (L, config):
    dec = sum ([2**n for n in range(0,L)])
    return fock(L, dec)

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
def W (L, config):
    return (1.0/sqrt(L)) \
            * sum ([one_alive(L, k) for k in range(L)])

# Create a state with GHZ-type entanglement.
# Reduces to 1/sqrt(2) (|00> + |11>) in L = 2 limit
# ------------------------------------------------------
def entangled_list (L, config):
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


def qubit(th, ph):
    return cos(th/2.0) * bvecs['0'] + exp(1j*ph) * sin(th/2) * bvecs['1']

def qubits(L, config):
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
smap = { 'd' : fock,
         's' : es_list,
         'l' : alive_list,
         'a' : all_alive,
         'c' : center,
         'q' : qubits,
         'r' : rand_state, 
         'G' : GHZ,
         'W' : W,
         'E' : entangled_list } 

def make_state (L, IC):
    state = np.array([0.0]*(2**L), dtype = complex)

    if type(IC) == str:
        name = IC[0]
        config = IC[1:]
        state = smap[name](L, config)

    if type(IC) == list:
        for s in IC: 
                name = s[0][0]
                config = s[0][1:]
                coeff = s[1]
                state = state + coeff * smap[name](L, config)
    return state


if __name__ == '__main__':
    import measures as ms 
    import states as ss
    L_list = [4]

    # spin down (or |1>) at sites 0 and 2 spin up (or |0>) at 1 and 3
    IC_list = ['l0_2']

    for L, IC in zip(L_list, IC_list):
        
        print()
        
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
       
        print('measurement results along Z axis:')
        print(meas_list)

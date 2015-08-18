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
    Tt, Pp = config.split('_')
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

# Make the specified state
# ------------------------
smap = { 'd' : fock,
         's' : es_list,
         'l' : alive_list,
         'a' : all_alive,
         'c' : center,
         'q' : qubits,
         'G' : GHZ,
         'W' : W,
         'E' : entangled_list } 

def make_state (L, IC):
    state = np.array([0.0]*(2**L), dtype = complex)
    for s in IC: 
            name = s[0][0]
            config = s[0][1:]
            coeff = s[1]
            state = state + coeff * smap[name](L, config)
    return state




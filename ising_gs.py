#!/usr/bin/python3

from cmath import sqrt, sin, cos, exp, pi
from math import log
import numpy as np
import matrix as mx
import states as ss


I = ss.pauli['0']
X = ss.pauli['1']
Z = ss.pauli['3']

def H_local(c):
    return -mx.listkron([Z,Z]) - c * mx.listkron([X, I])

def normalize(state):
    ip = (state.conj().dot(state))
    if ip != 0.0:
        state = 1.0/sqrt(ip) * state
    return state

def H_on_state(H_loc, state):
    L = int( log(len(state), 2) )
    for j in range(L): 
        js = [j, (j+1)%L]
        state = mx.op_on_state(H_loc, js, state)
    return state

def ising_gs(L, c, d = 1e5):
    H_loc = H_local(c) + d * np.eye(4) 
    
    state = np.array([1/sqrt(L)]*(2**L)) 
    for n in range(1000):
        state = H_on_state(H_loc, state)
        state = normalize(state)
    state[state < 1e-14] = 0.0 + 0.0j
    return state

print(ising_gs(2, 0.0))
print(ising_gs(2, 0.5))
print(ising_gs(2, 1.0))
print(ising_gs(2, 1000.0))
print()
print(ss.make_state(2, [('G',1.0)]))

#!/usr/bin/python

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time
from os import makedirs, environ
import scipy.sparse as sps
from scipy.linalg import kron,  expm
from math import pi, sqrt, log
import networkmeasures as nm
from itertools import product, cycle
import copy
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
import json
from collections import namedtuple, Iterable, OrderedDict

from numpy import sin, cos, log, pi
import matrix as mx
import states as ss
import sweep_eca as sweep

init_state = ss.make_state(3 ,[('E0_1', 1.0)])

r0 = mx.rdms(init_state, [0])
r1 = mx.rdms(init_state, [1])
r2 = mx.rdms(init_state, [2])

state_1 = mx.op_on_state(mx.listkron([ss.pauli['1']]*2), [0,2], init_state)
r0_1 = mx.rdms(state_1, [0])
r1_1 = mx.rdms(state_1, [1])
r2_1 = mx.rdms(state_1, [2])


rd = mx.rdms(state_1, [1,2])




# sx and sz entropies
# -------------------
def sz(th):
    p0 = 0.5 * (1.0 + cos(2.0*th))
    p1 = 0.5 * (1.0 - cos(2.0*th))
    return -p0*log(p0)/log(2.) - p1*log(p1)/log(2.)

def sx(th):
    pp = 0.5 * (1.0 + sin(2.0*th))
    pm = 0.5 * (1.0 - sin(2.0*th))
    return -pp*log(pp)/log(2.) - pm*log(pm)/log(2.)

e =0.001
th = np.linspace(0.0+e, 2*pi-e, 100)
const_th = np.array([pi/4.0]*100)
plt.plot(th, sz(th), label='sz')
plt.plot(th, sx(th), label='sx')
plt.plot(th, sz(const_th))
plt.legend()
#plt.show()




# update op
# ---------

G = np.array([[0,1,0,0,0,0,0,0],
              [0,0,1,0,0,0,0,0],
              [0,0,0,1,0,0,0,0],
              [0,0,0,0,1,0,0,0],
              [0,0,0,0,0,1,0,0],
              [0,0,0,0,0,0,1,0],
              [0,0,0,0,0,0,0,1],
              [1,0,0,0,0,0,0,0]])

for R in range(256):
    T = sweep.general_local_update_op(R, th=pi/2.0)
    I = np.eye(8)
    Tdag = mx.dagger(T)
    TdT = Tdag.dot(T)
    TTd = T.dot(Tdag)
    unitaryQ = np.array_equal(TdT, I) and np.array_equal(TTd, I) 
    if unitaryQ is True:
        TG = T.dot(G)
        GT = G.dot(T)
        #print(np.array_equal(TG,GT))
        print(R, mx.dec_to_bin(R^204,8))



for t in range(4):
    if t%2==0:
        for j in range(5):
            print(j)
    elif t%2==1:
        for j in range(3,0,-1):
            print(j)

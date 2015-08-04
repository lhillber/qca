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

import matrix as mx
import states as ss


init_state = ss.make_state(3 ,[('E0_1', 1.0)])

r0 = mx.rdms(init_state, [0])
r1 = mx.rdms(init_state, [1])
r2 = mx.rdms(init_state, [2])

state_1 = mx.op_on_state(mx.listkron([ss.pauli['1']]*2), [0,2], init_state)
r0_1 = mx.rdms(state_1, [0])
r1_1 = mx.rdms(state_1, [1])
r2_1 = mx.rdms(state_1, [2])

print('t = 0')
print(init_state)
print(r0)
print(r1)
print(r2)
print()
print('t = 1')
print(state_1)
print(r0_1)
print(r1_1)
print(r2_1)

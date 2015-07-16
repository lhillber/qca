#!/usr/bin/python

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time
from os import makedirs, environ
import scipy.sparse as sps
from scipy.linalg import kron,  expm
from math import pi, sqrt, log
import qca_util as util
import networkmeasures as nm
from itertools import product, cycle
import copy
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages


local_basis = { 'dead'  : np.array([[1.,0.]]).transpose(),
                'alive' : np.array([[0.,1.]]).transpose(),
                'es'    : np.array([[1./sqrt(2), 1./sqrt(2)]]).transpose() }

uabi = { '00' : (np.array([[1,0],[0,0]]), np.array([[0,1],[0,0]])), 
         '01' : (np.array([[0,1],[1,0]]), np.array([[0,0],[0,0]])), 
         '10' : (np.array([[1,0],[0,1]]), np.array([[0,0],[0,0]])),
         '11' : (np.array([[0,0],[0,1]]), np.array([[0,0],[1,0]]))}

local_rhos = { 0 : local_basis['dead'].dot(local_basis['dead'].transpose()), 
               1 : local_basis['alive'].dot(local_basis['alive'].transpose()) }


# string describing initial condition (IC)
# ----------------------------------------
def IC_name(IC):
    return '-'.join(['{:0.3f}{}'.format(val,name) \
                for (name, val) in IC])

def Rs_name(Rs):
    return '-'.join([str(R) for R in Rs])
    
 
# string describing simulation parameters
# ---------------------------------------
def sim_name(Rs, IC, L, tmax):
    return 'R{}_IC{}_L{}_tmax{}'.format( \
                Rs_name(Rs), IC_name(IC), L, tmax)

print(sim_name([200,220], [('c1d1',1.0,)], 5, 5))


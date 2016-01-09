#!/usr/bin/python3

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
import processing as pp
import fio as io
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

for R in range(256):
    T = sweep.local_update_op(R)
    I = np.eye(8)
    Tdag = mx.dagger(T)
    TdT = Tdag.dot(T)
    TTd = T.dot(Tdag)
    unitaryQ = np.array_equal(TdT, I) and np.array_equal(TTd, I) 
    if unitaryQ is True:
        print(R, mx.dec_to_bin(R^204,8))
        #print(T)





fixed_params_dict = { 'output_name' : ['sweep_block_4'],
                             'mode' : ['sweep', 'block'],
                                'R' : [150, 102],
                               'IC' : ['s18'],
                                'L' : [19],
                             'tmax' : [1000] }

var_params_dict = {'center_op' : [['H'], ['H','T'], ['H','X','T']]}

def concat_dicts(d1, d2):
    d = d1.copy()
    d.update(d2)
    return d

def dict_product(dicts):
        return (dict(zip(dicts, x)) for x in product(*dicts.values()))

def import_for_comp(fixed_params_dict, var_params_list):
    
    unordered_params_dict_list = [0]*len(list(dict_product(fixed_params_dict)))
    for index, fixed_dict in enumerate(dict_product(fixed_params_dict)):
        unordered_params_dict_sublist = []
        for var_dict in dict_product(var_params_dict):
            unordered_params_dict_sublist = unordered_params_dict_sublist + \
                            [concat_dicts(fixed_dict, var_dict)]
        unordered_params_dict_list[index] = unordered_params_dict_sublist

    params_list_list = [[pp.make_params( unordered_params_dict['output_name'],
                                unordered_params_dict['mode'],
                                unordered_params_dict['center_op'],
                                unordered_params_dict['R'],
                                unordered_params_dict['IC'],
                                unordered_params_dict['L'],
                                unordered_params_dict['tmax'])


                    for unordered_params_dict in
                    unordered_params_dict_sublist ]
                    
                    for unordered_params_dict_sublist in
                    unordered_params_dict_list ] 
                    

    name_list = [[io.sim_name(params) 
        for params in params_list]
        for params_list in params_list_list]

    # res_list = [io.read_results(params) for params in params_list]
    #print(name_list) 

#import_for_comp(fixed_params_dict, var_params_dict)

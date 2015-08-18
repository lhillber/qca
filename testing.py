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


rd = mx.rdms(state_1, [1,2])
basestring = (str, bytes)

def isnamedtuple(obj):
    return isinstance(obj, tuple) \
    and hasattr(obj, "_fields") \
    and hasattr(obj, "_asdict") \
    and callable(obj._asdict)

def serialize(data):
    if data is None or isinstance(data, (bool, int, float, basestring)):
        return data
    
    if isinstance(data, list):
        return [serialize(val) for val in data]
    
    if isinstance(data, OrderedDict):
        return {"py/collections.OrderedDict": 
                [[serialize(k), serialize(v)] for k, v in data.items()]}
    
    if isnamedtuple(data):
        return {
                "py/collections.namedtuple" :
                { 
                  "type"   : type(data).__name__, 
                  "fields" : list(data._fields), 
                  "values" : [serialize(getattr(data,f)) for f in data._fields]
                }
               }

    if isinstance(data, dict):
        if all(isinstance(k, basestring) for k in data):
            return {k : serialize(v) for k, v in data.items()}
        return {"py/dict" : [[serialize(k), serialize(v)] for k, v in data.items()]}

    if isinstance(data, tuple): 
        return {"py/tuple" : [serialize(val) for val in data]}

    if isinstance(data, set): 
        return {"py/set" : [serialize(val) for val in data]}

    if isinstance(data, np.ndarray):
        return {"py/numpy.ndarray":
                { "values" : data.tolist(),
                  "dtype"  : str(data.dtype)
                }
               }
    
    raise TypeError("Type %s not data-serializable" % type(data))


def restore(dct):
    if "py/dict" in dct:
        return dict(dct["py/dict"])
    
    if "py/tuple" in dct:
        return tuple(dct["py/tuple"])
    
    if "py/set" in dct:
        return set(dct["py/set"])

    if "py/collections.namedtuple" in dct: 
        data = dct["py/collections.namedtuple"]
        return namedtuple(data["type"], data["fields"])(*data["values"])
    
    if "py/numpy.ndarray" in dct:
        data = dct["py/numpy.ndarray"]
        return np.array(data["values"], dtype=data["dtype"])

    if "py/collections.OrderedDict" in dct:
        return OrderedDict(dct["py/collections.OrderedDict"])
    
    return dct



def data_to_json(data):
    return json.dumps(serialize(data))

def json_to_data(s):
    return json.loads(s, object_hook=restore)





data = {'np'  : np.array( [[1.0, 0.0],[-1.0, 0.0]]), 
        'l'   : [1,2],
        'lnp' : [np.array([1.0,0]), np.eye(3)],
        'dnp' : {0 : np.array([0.0,0]), 1 : np.eye(3)}}

jdata = data_to_json(data)

with open('../testing/test_data.res', 'w') as outfile:
    outfile.write(jdata)


with open('../testing/test_data.res', 'r') as infile:
    jdata_in = infile.read()
    data_in = json_to_data(jdata_in)
    print(data_in['dnp'][0])



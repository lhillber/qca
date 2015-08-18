#!/usr/bin/python

from os import environ, makedirs
from collections import namedtuple, Iterable, OrderedDict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import json


# File I/O functions
# ==================

# make data serializable
# from http://robotfantastic.org/
# serializing-python-data-to-json-some-edge-cases.html
# ----------------------------------------------------

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

# string describing initial condition (IC)
# ----------------------------------------
def IC_name(IC):
    return '-'.join(['{:0.3f}{}'.format(val.real, name) \
                for (name, val) in IC])

# string describing simulation parameters
# ---------------------------------------
def sim_name(R, IC, L, tmax):
    return 'R{}_IC{}_L{}_tmax{}'.format( \
                R, IC_name(IC), L, tmax)

# make an output directory
# ------------------------
def base_name(output_name, output_type):
    bn = environ['HOME']+'/Documents/qca/output/' + output_name + '/' + output_type 
    makedirs(bn, exist_ok=True)
    return bn

# full path to a file to be opened
# --------------------------------
def file_name(output_name, output_type, name, ext):
    return base_name(output_name, output_type) + '/' + name + ext

# save simulation results
# -----------------------
def write_results(results, params):
    output_name, R, IC, L, tmax = params 
    with open(file_name(output_name,'data', sim_name(R, IC, L, tmax), '.res'), 'w') as outfile:
        outfile.write(data_to_json(results))
    return

# load simulation results
# -----------------------
def read_results(params):
    input_name, R, IC, L, tmax = params 
    with open(file_name(input_name, 'data', sim_name(R, IC, L, tmax), '.res'), 'r') as infile:
       results = infile.read()
    return json_to_data(results)

# save multi page pdfs of plots
# -----------------------------
def multipage(fname, figs=None, clf=True, dpi=300): 
    pp = PdfPages(fname) 
    if figs is None:
        figs = [plt.figure(fignum) for fignum in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
        if clf==True:
            fig.clf()
    pp.close()
    return


#!/usr/bin/python3

from os import environ, makedirs
from collections import namedtuple, Iterable, OrderedDict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import json
import h5py


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
        if data.dtype == complex:
            return {"py/numpy.ndarray_complex":
                    { "real_vals" : data.real.tolist(),
                      "imag_vals" : data.imag.tolist(),
                      "dtype"  : str(data.dtype) 
                    }
                   }
    
        else:
            return {"py/numpy.ndarray":
                    { "vals" : data.tolist(),
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
    
    if "py/numpy.ndarray_complex" in dct:
        print('ENTER')
        data = dct["py/numpy.ndarray_complex"]
        print(data["real_vals"])
        return np.array(data["real_vals"], dtype=complex) +\
            1j*np.array(data["imag_vals"], dtype=complex)

    if "py/numpy.ndarray" in dct:
        data = dct["py/numpy.ndarray"]
        return np.array(data["vals"], dtype=data["dtype"])

    if "py/collections.OrderedDict" in dct:
        return OrderedDict(dct["py/collections.OrderedDict"])
    
    return dct

def data_to_json(data):
    return json.dumps(serialize(data))

def json_to_data(s):
    return json.loads(s, object_hook=restore)

# string describing initial condition (IC)
# ----------------------------------------
def make_IC_name(IC):
    if type(IC) is str:
        return IC
    else:
        return '-'.join(['{:0.3f}{}'.format(val.real, name) \
                    for (name, val) in IC])

# string describing center site update op (V):
# --------------------------------------------
def make_V_name(V):
    if type(V) is str:
        return V
    else:
        return ''.join(V)

# string describing simulation parameters
# ---------------------------------------
def sim_name(params, IC_name=None, V_name=None):
    sname = "{}_L{}_T{}_R{}".format(
            params['mode'], params['L'], params['T'], params['R'])

    if V_name is None:
        sname = sname + '_V' + make_V_name(params['V'])
    else:
        sname = sname + '_V' + V_name

    if IC_name is None:
        sname = sname + '_IC' + make_IC_name(params['IC'])
    else:
        sname = sname + '_IC' + IC_name

    return sname

# make an output directory
# ------------------------
# NOTE: documents folder is assumed to be a lowercase d (because it should be...)
# and qca dir is assumed to be in documents. If this isn't true, this function
# will make all non_existing direcories, so be carful!
def base_name(output_dir, sub_dir):
    bn = environ['HOME'] + '/documents/qca/output/' + \
         output_dir + '/' + sub_dir + '/'
    makedirs(bn, exist_ok=True)
    return bn

# default path to a file to be opened
# -----------------------------------
def default_file_name(output_dir, sub_dir, name, ext, v=0):
    return base_name(output_dir, sub_dir) + name +'_v'+str(v) + ext

# make default or use suppled file name
# -------------------------------------
def file_name(params, sub_dir='data', ext='.hdf5'):
    # check for a full path to save results to
    if 'fname' in params:
        fname = params['fname']

    # otherwise create default location and file name 
    else:
        output_dir = params['output_dir']
        ic_name = make_IC_name(params['IC'])
        fname = default_file_name(output_dir, sub_dir, sim_name(params), ext)

        # make a unique name for IC's made with a random throw
        # NOTE: the name will be unique, but not properly numbered, FIX THIS
        if ic_name[0] == 'r' and isfile(fname):
            fname_list = list(fname) 
            fname_list[-5] = str(eval(fname[-5])+1)
            fname = ''.join(fname_list)

    return fname


# save simulation results
# ----------------------
def write_states(fname, state_gen):
        f = h5py.File(fname, 'w')
        for t, state in enumerate(state_gen):
            f.create_dataset(str(t), data=state, dtype=complex)
        f.close()

# load simulation results
# ----------------------
def read_states(fname):
    f = h5py.File(fname, 'r')
    for t in f:
        state = f[t]
        yield state
    f.close()

# load simulation result at a specific time step
# ----------------------------------------------
def read_state(fname, t):
    f = h5py.File(fname, 'r')
    state = f[t]
    f.close()
    return state

def write_hdf5(fname, data_dict, force_rewrite=False):
    f = h5py.File(fname, 'a')
    for key, dat in data_dict.items():
        dt = dat.dtype
        if key in f:
            if force_rewrite:
                del f[key]
            if not force_rewrite:
                f[key][::] = dat
        if key not in f:
            f.create_dataset(key, data=dat, dtype=dt)
    f.close()

def read_hdf5(fname, keys):
    f = h5py.File(fname, 'r')
    if type(keys) == str:
        if isinstance(f[keys], h5py.Group):
            gkeys = f[keys].keys()
            gdat = [0]*len(gkeys) 
            for gkey in gkeys:
                gdat[j] = f[keys][gkey][::]
            dat = gdat
        elif isinstance(f[keys], h5py.Dataset):
            dat = f[keys][::]

    elif type(keys) == list:
        dat = [0]*len(keys)
        for i, key in enumerate(keys):
            if isinstance(f[key], h5py.Group):
                gkeys = f[key].keys()
                gdat = [0]*len(gkeys) 
                for j, gkey in enumerate(gkeys):
                    gdat[j] = f[key][gkey][::]
                dat[i] = gdat
            elif isinstance(f[key], h5py.Dataset):
                dat[i] = f[key][::]
    f.close()
    return dat

# save simulateion results
# ------------------------
def write_results(results, fname):
    with open(fname, 'wb') as outfile:
        outfile.write(bytes(data_to_json(results), 'UTF-8'))
    return

# load simulation results
# -----------------------
def read_results(params=None, fname=None):
    if fname is None:
        input_dir = params['output_dir']
        fname = file_name(input_name, 'data', sim_name(params), '.res')
    with open(fname, 'rb') as infile:
       results = infile.read().decode('UTF-8')
    return json_to_data(results)

# save multi page pdfs of plots
# -----------------------------
def multipage(fname, figs=None, clf=True, dpi=300): 
    pp = PdfPages(fname) 
    if figs is None:
        figs = [plt.figure(fignum) for fignum in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf', bbox_inches='tight')
        if clf==True:
            fig.clf()
    pp.close()
    return

if __name__ == '__main__':
    from cmath import sqrt
    eq = 1.0/sqrt(2)

    params =  {
                'output_dir' : 'testing/state_saving',

                'L'    : 15,
                'T' : 1,
                'mode' : 'sweep',
                'R'    : 102,
                'V'    : ['H','X'],
                'IC'   : 'c1s0'
                                   }

    print(sim_name(params))

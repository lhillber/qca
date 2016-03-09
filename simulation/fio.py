#!/usr/bin/python3

from os import environ, makedirs
from os.path import isfile
from collections import namedtuple, Iterable, OrderedDict
from itertools import product

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import json
import h5py
from simulation.outputdir import base_dir


# parameter list creation
# =======================

# combine two dicts (MAKES A COPY)
# --------------------------------
def concat_dicts(d1, d2):
    d = d1.copy()
    d.update(d2)
    return d


# generate combinations of fixed params to be filled by var params
# ----------------------------------------------------------------
def dict_product(dicts):
        return (dict(zip(dicts, x)) for x in product(*dicts.values()))


# make nested list of params dicts
# --------------------------------
def make_params_list_list(fixed_params_dict, var_params_dict):
    params_list_list = [0]*len(list(dict_product(fixed_params_dict)))
    for index, fixed_dict in enumerate(dict_product(fixed_params_dict)):
        params_dict_sublist = []
        for var_dict in dict_product(var_params_dict):
            params_dict_sublist = params_dict_sublist + \
                            [concat_dicts(fixed_dict, var_dict)]
        params_list_list[index] = params_dict_sublist

    return np.asarray(params_list_list)


# prepare nested dicts for hdf5 dump
# ----------------------------------
def flatten_dict(d):
    def items():
        for key, value in list(d.items()):
            if isinstance(value, dict):
                for subkey, subvalue in flatten_dict(value).items():
                    yield key + "/" + subkey, subvalue
            else:
                yield key, value
            del d[key]
    return dict(items())


# File I/O functions
# ==================

# string describing initial condition (IC)
# ----------------------------------------
def make_IC_name(IC):
    if type(IC) is str:
        return IC

    elif type(IC) is list:
        return '-'.join(['{:0.3f}{}'.format(val.real, name) \
                    for (name, val) in IC])

    elif type(IC) is np.ndarray():
        return 'custom'


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
    sname = "{}_L{}_T{}".format(
            params['mode'], params['L'], params['T'])

    if 'S' in params:
        sname = sname + '_S' + str(params['S'])

    elif 'R' in params:
        sname = sname + '_R' + str(params['R'])

    if V_name is None:
        sname = sname + '_V' + make_V_name(params['V'])
    else:
        sname = sname + '_V' + V_name

    if IC_name is None:
        sname = sname + '_IC' + make_IC_name(params['IC'])
    else:
        sname = sname + '_IC' + IC_name

    if 'BC' in params:
        sname = sname + '_BC' + params['BC']

    return sname


# make an output directory
# ------------------------
# NOTE: documents folder is assumed to be a lowercase d (because it should be...)
# and qca dir is assumed to be in documents. If this isn't true, this function
# will make all non_existing direcories, so be carful!
def base_name(output_dir, sub_dir):
    bn = environ['HOME'] + base_dir + \
         output_dir + '/' + sub_dir + '/'
    makedirs(bn, exist_ok=True)
    return bn


# default path to a file to be opened
# -----------------------------------
def default_file_name(params, sub_dir, ext, v=0, IC_name=None, V_name=None):
    output_dir = params['output_dir']
    sname = sim_name(params, IC_name=IC_name, V_name=V_name)
    return base_name(output_dir, sub_dir) + sname +'_v'+str(v) + ext


# make default or use suppled file name
# -------------------------------------
def make_file_name(params, sub_dir='data', ext='.hdf5', iterate=False):
    # check for a full path to save results to
    if 'fname' in params:
        fname = params['fname']

    # otherwise create default location and file name
    else:
        if 'IC_name' in params:
            IC_name = params['IC_name']
        else:
            IC_name = make_IC_name(params['IC'])
        if 'V_name' in params:
            V_name = params['V_name']
        else:
            V_name = make_V_name(params['V'])
        fname = default_file_name(params, sub_dir, ext,
                v=0, IC_name=IC_name, V_name=V_name)

        # make a unique name for IC's made with a random throw
        # iterate MIGHT cause a race condition when running sims in parallel
        if iterate and isfile(fname):
            v = eval(fname.split('/')[-1].split('v')[1].split('.')[0])
            while isfile(fname):
                fname = default_file_name(params, sub_dir, ext,
                        v=v, IC_name=IC_name, V_name=V_name)
                v = v+1
    h5py.File(fname, 'a').close()
    return fname


def write_hdf5(fname, data_dict, force_rewrite=False):
    f = h5py.File(fname, 'a')
    flat_data_dict = flatten_dict(data_dict)
    res_size = sum(val.nbytes for val in flat_data_dict.values())
    for key, dat in flat_data_dict.items():
        dt = dat.dtype
        if key in f:
            if force_rewrite or dat.shape != f[key].shape:
                del f[key]
            if not force_rewrite:
                f[key][::] = dat
        if key not in f:
            f.create_dataset(key, data=dat, dtype=dt)
    f.close()
    return res_size

def read_hdf5(fname, keys):
    f = h5py.File(fname, 'r')
    if type(keys) == str:
        if isinstance(f[keys], h5py.Group):
            gkeys = f[keys].keys()
            gdat = {}
            for gkey in gkeys:
                gdat[gkey] = f[keys][gkey][::]
            dat[key] = gdat
        elif isinstance(f[keys], h5py.Dataset):
            dat = f[keys][::]

    elif type(keys) == list:
        dat = {}
        for i, key in enumerate(keys):
            if isinstance(f[key], h5py.Group):
                gkeys = f[key].keys()
                gdat = {} 
                for j, gkey in enumerate(gkeys):
                    gdat[gkey] = f[key][gkey][::]
                dat[key] = gdat

            elif isinstance(f[key], h5py.Dataset):
                dat[key] = f[key][::]
    f.close()
    return dat


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
                'T'    : 1,
                'mode' : 'sweep',
                'S'    : 6,
                'V'    : 'HX',
                'IC'   : 'c1s0'
                                   }
    print(make_file_name(params, iterate=False))

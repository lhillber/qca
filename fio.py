#!/usr/bin/python

import json
from os import environ, makedirs

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages

# File I/O functions
# ==================

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
    results = np.asarray(results).tolist()
    with open(file_name(output_name,'data', sim_name(R, IC, L, tmax), '.res'), 'w') as outfile:
        json.dump(results, outfile)
    return

# load simulation results
# -----------------------
def read_results(params):
    input_name, R, IC, L, tmax = params 
    with open(file_name(input_name, 'data', sim_name(R, IC, L, tmax), '.res'), 'r') as infile:
       results =  json.load(infile)
    return results

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


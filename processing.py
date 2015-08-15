#!/usr/bin/python

from cmath  import sqrt
import numpy as np
import information as im
import fio as io
import networkmeasures as nm
import measures as ms
import run 
import matplotlib        as mpl
import matplotlib.pyplot as plt
import plotting as pt

# default plot font to bold and size 16
# -------------------------------------
font = {'weight':'bold', 'size':16}
mpl.rc('font',**font)

eq = 1.0/sqrt(2.0)

def get_diag_vecs(mats):
    mats = np.asarray(mats)
    return np.array([mat.diagonal() for mat in mats])

def get_offdiag_mats(mats):
    L = len(mats[0])
    mats = np.asarray(mats)
    for t, mat in enumerate(mats): 
        mat[np.arange(L), np.arange(L)] = 0.0
        mats[t] = mat
    return mats

def spatialnetworksQ(results, typ):
    n_mats = results[typ]
    corr, loc = get_offdiag_mats(n_mats), get_diag_vecs(n_mats)
    return im.spatialnetworksQ(corr, loc)

def measure_networks(nets, tasks):
    measures = {} 
    for task in tasks:
        measures[task] = [ms.NMcalc(net, tasks=tasks)[task] for net in nets]
    return measures

def make_net_measures_dict(results, tasks):
    nz_network = spatialnetworksQ(results, 'nz')
    nx_network = spatialnetworksQ(results, 'nx')
    mi_network = results['mi']
    
    nz_net_measures = measure_networks(nz_network, tasks)
    nx_net_measures = measure_networks(nx_network, tasks)
    mi_net_measures = measure_networks(mi_network, tasks) 
    return { 'nz': nz_net_measures,
             'nx': nx_net_measures,
             'mi': mi_net_measures }

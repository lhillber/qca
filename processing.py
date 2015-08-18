#!/usr/bin/python

from cmath  import sqrt
import copy
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
    mats_out = copy.deepcopy(mats)
    for t, mat in enumerate(mats_out): 
        mats_out[np.arange(L), np.arange(L)] = 0.0
        mats_out[t] = mat
    return mats_out

def spatialnetworksQ(results, typ):
    n_mats = results[typ]
    corr, loc = get_offdiag_mats(n_mats), get_diag_vecs(n_mats)
    return im.spatialnetworksQ(corr, loc)

def measure_networks(nets, tasks=['Y','CC'], typ='avg'):
    measures = {} 
    for task in tasks:
        measures[task] = np.asarray([ms.NMcalc(net, typ=typ,
                                    tasks=tasks)[task] for net in nets])
    return measures

def make_net_dict(results, net_types=['nz', 'nx', 'mi']):
    net_dict = {}
    for net_typ in net_types:
        if net_typ == 'mi':
            network = results['mi']
        else: 
            network = spatialnetworksQ(results, net_typ)
        net_dict[net_typ] = network
    return net_dict

def comp_import(R, CIC, QIC, L, tmax):
    Cname = io.sim_name(R, CIC, L, tmax)
    Cfname = io.file_name(output_name, 'plots', 'C'+Cname, '.pdf')

    Qname = io.sim_name(R, QIC, L, tmax)
    Qfname = io.file_name(output_name, 'plots', 'Q'+fname, '.pdf')

    











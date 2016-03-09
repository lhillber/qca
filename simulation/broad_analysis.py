#!/usr/bin/python3

from os import environ
import simulation.fio as io
import simulation.plotting as pt
import simulation.measures as ms
import simulation.matrix as mx
import matplotlib.pyplot as plt

import simulation.fitting as ft
import simulation.fit_grids as fit_grids

import numpy as np
import h5py
import matplotlib          as mpl
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import UnivariateSpline
from sklearn.mixture import GMM


from itertools import product
from math import pi, sin, cos

# default plot font
# -----------------
font = {'size':12, 'weight' : 'normal'}
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rc('font',**font)


def color_map(val, val_max=90, val_min=0):
    return val/(val_max - val_min)

def mode_marker_map(mode):
    mm_map = {'sweep':'^', 'block':'s', 'alt':'o'}
    return mm_map[mode] 

def obs_label(obs_key):
    if obs_key == 's':
        obs_label = r'$s_{\mathrm{vn}}$'
    elif obs_key in ['ND', 'CC', 'Y']:
        obs_label = r'$'+obs_key+'$'
    return obs_label

def set_color(color_by, params):
    # set color
    if color_by == 'th':
        V = params['V']
        val = pt.get_th(V)
        val_min = min(degs)
        val_max = max(degs)
        cbar_label = r'$\theta$'
    elif color_by == 'S':
        S = params['S']
        val = S
        val_min = min(Ss)
        val_max = max(Ss)
        cbar_label = r'$S$'
    elif color_by == 'IC':
        IC = params['IC']
        val = ICs.index(IC)
        val_min = 0 
        val_max = len(ICs)
        cbar_label = 'IC index'

    elif color_by == 'L':
        L = params['L']
        val = L
        val_min = min(Ls)
        val_max = max(Ls)
        cbar_label = 'L'
    return val, val_min, val_max, cbar_label


def plot_scatter(fig, ax, m, obs_means, colors, val_min, val_max, cbar_label,
        obs_list, nx_ticks=6, ny_ticks=6):
        marker = 'o'
        x = obs_means[m,0]
        y = obs_means[m,1]
        c = colors[m,1]
        xmin = min(x)
        xmax = max(x)
        dx = (xmax - xmin)/nx_ticks
        ymin = min(y)
        ymax = max(y)
        dy = (ymax-ymin)/ny_ticks

        sc = ax.scatter(x, y, c=c, marker = marker, vmin=val_min,
                vmax=val_max, cmap=cm.rainbow)

        dval = max(1, int((val_max-val_min)/6))
        plt.colorbar(sc, ticks=[range(int(val_min), int(val_max+1), dval)], label=cbar_label)

        ax.set_xlabel(obs_label(obs_list[0]))
        ax.set_ylabel(obs_label(obs_list[1]))
        #ax.set_xlim(xmin, xmax)
        #ax.set_ylim(ymin, ymax)
        ax.xaxis.major.locator.set_params(nbins=6)
        ax.yaxis.major.locator.set_params(nbins=6)
        ax.margins(0.02)
        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)
        #plt.show()

if __name__ == '__main__':
    Ls = [11, 12, 13]
    degs = range(0, 105, 15)
    Ss = range(1,16)
    ICs = ['G', 'W', 'c2_B0-1_0', 'c2_B0-1_1', 'c2_B0-1_2', 'c2_B0-1_3' ]
    #ICs = ['f0', 'c3_f1', 'c3_f0-1', 'c3_f0-2', 'c3_f0-1-2']
    color_by_list = ['th', 'S', 'IC', 'L']
    obs_list_list = [['ND', 'Y'], ['CC', 'Y'], ['ND', 'CC'], ['ND', 's'], 
            ['CC','s'], ['Y', 's']]

    # outer params
    fixed_params_dict = {
                'output_dir' : ['entangled_IC'],
                'T'   : [1000],
                'BC'  : ['1_00'],
                'mode': ['alt','sweep','block'],
                 }

    #inner params
    var_params_dict = {

                'L'   : Ls,
                'V' : ['HP_'+str(deg) for deg in degs],
                'IC'  : ICs,
                'S'   : Ss
                 }

params_list_list = io.make_params_list_list(fixed_params_dict, var_params_dict)
fignum=0
for color_by in color_by_list:
    for obs_list in obs_list_list:
        M, N = params_list_list.shape
        J = len(obs_list)
        obs_means = np.zeros((M, J, N))
        colors = np.zeros((M, J, N))
        fig = plt.figure(fignum, figsize=(3, 2.83))
        ax = fig.add_subplot(111)
        fignum += 1
        for m, params_list in enumerate(params_list_list):
            for j, obs_key in enumerate(obs_list):
                for n, params in enumerate(params_list):
                    val, val_min, val_max, cbar_label = set_color(color_by, params)
                    res_path = io.default_file_name(params, 'data', '.hdf5')
                    res = h5py.File(res_path)
                    obs = res[obs_key][500:]
                    obs_mean = np.nanmean(obs)
                    obs_means[m,j,n] = obs_mean
                    colors[m,j,n] = val
            plot_scatter(fig, ax, m, obs_means, colors, val_min, val_max,
                    cbar_label, obs_list)
bn = io.base_name(fixed_params_dict['output_dir'][0], 'plots/scatters') 
fname = params['mode'] + '_L11-12-13' + '_scatter.pdf'

io.multipage(bn+fname)

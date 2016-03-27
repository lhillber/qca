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
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
from matplotlib import collections  as mc
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
        obs_label = r'$s^{\mathrm{vN}}$'
    elif obs_key in ['ND', 'CC', 'Y']:
        obs_label = r'$'+obs_key+'$'
    return obs_label

def set_color(color_by, params, mode_index={'sweep':0,'block':1,'alt':2}):
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
        val_max = max(Ss)+1
        cbar_label = r'$S$'
    elif color_by == 'IC':
        IC = params['IC']
        val = ICs.index(IC)
        val_min = 0 
        val_max = len(ICs)-1
        cbar_label = 'IC index'

    elif color_by == 'L':
        L = params['L']
        val = L
        val_min = min(Ls)
        val_max = max(Ls)
        cbar_label = 'L'

    elif color_by == 'mode':
        mode = params['mode']
        val = mode_index[mode]
        val_min = 0
        val_max = 3
        cbar_label = 'MODE'
    return val, val_min, val_max, cbar_label


def split_by_slope(x, y, ax, sm=3, ex=0.025):
        s0 = []
        s1 = []
        xs = np.linspace(min(x), max(x), 100)
        for j in range(n):
            xj, yj = x[j], y[j]
            if max(yj, xj) > ex :
                s = yj/xj
            else:
                s = np.nan
            if s > sm:
                s1.append(s)
            else:
                s0.append(s)

        slope0 = np.nanmean(s0)
        slope1 = np.nanmean(s1)

        ax.plot(xs, slope0*xs)
        ax.plot(xs, slope1*xs)
        ax.plot(xs, sm*xs)
        lc = mc.LineCollection([[(0,ex),(ex,ex)],[(ex,0),(ex,ex)]],
                color=['black', 'black'])
        ax.add_collection(lc)
        ax.plot(xs, sm*xs)
        return slope0, sm, slope1

import matplotlib.colors as mcolors

def colorbar_index(colorbar_label, ncolors, cmap):
    cmap = cmap_discretize(cmap, ncolors)
    mappable = cm.ScalarMappable(cmap=cmap)
    mappable.set_array([])
    mappable.set_clim(-0.5, ncolors+0.5)
    colorbar = plt.colorbar(mappable)
    colorbar.set_ticks(np.linspace(0, ncolors, ncolors))
    colorbar.set_ticklabels(list(map(int, np.linspace(int(val_min),
        int(val_max), ncolors))))
    colorbar.set_label(cbar_label)

def cmap_discretize(cmap, N):
    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)
    colors_i = np.concatenate((np.linspace(0, 1., N), (0.,0.,0.,0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N+1)
    cdict = {}
    for ki,key in enumerate(('red','green','blue')):
        cdict[key] = [ (indices[i], colors_rgba[i-1,ki], colors_rgba[i,ki])
                       for i in range(N+1) ]
    # Return colormap object.
    return mcolors.LinearSegmentedColormap(cmap.name + "_%d"%N, cdict, 1024)

def plot_scatter(fig, ax, m, obs_means, colors, val_min, val_max, cbar_label,
        obs_list, cmap, nx_ticks=6, ny_ticks=6, marker = 'o'):
        c = colors[m,1]
        nc = len(set(c))
        x = obs_means[m,0]
        y = obs_means[m,1]

        xmin = min(x)
        xmax = max(x)
        dx = (xmax - xmin)/nx_ticks

        ymin = min(y)
        ymax = max(y)
        dy = (ymax-ymin)/ny_ticks

        dval = max(1, int((val_max-val_min)/6))


        # define the bins and normalize
        bounds = [range(int(val_min), int(val_max), dval)]
        bounds = np.linspace(int(val_min), int(val_max), nc)
        print(bounds)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        sc = ax.scatter(x, y, c=c, marker = marker, vmin=val_min,
                vmax=val_max, cmap=cmap)

        xs = np.linspace(0, max(x), 100)
        ys = np.linspace(0, max(y), 100)


        #slop0, sm, slope1 = split_by_slope(x, y, ax)


        #cax = fig.add_axes([0.95, 0.1, 0.03, 0.8])
        #cb = mpl.colorbar.ColorbarBase(cax, label=cbar_label, cmap=cmap, norm=norm,
        #        spacing='proportional', ticks=bounds, boundaries=bounds, format='%1i')
        colorbar_index(cbar_label, ncolors=nc, cmap=cmap)  
        ax.set_xlabel(obs_label(obs_list[0]))
        ax.set_ylabel(obs_label(obs_list[1]))
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, ymax)
        ax.xaxis.major.locator.set_params(nbins=6)
        ax.yaxis.major.locator.set_params(nbins=6)
        ax.margins(0.02)
        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)
        #plt.show()


def plot_2D_hist(fig, ax, m, obs_means,  cbar_label,
        obs_list, nx_ticks=6, ny_ticks=6):
        x = obs_means[m,0]
        x[np.isnan(x)] = 0
        y = obs_means[m,1]
        y[np.isnan(y)] = 0
        xmin = min(x)
        xmax = max(x)
        dx = (xmax - xmin)/nx_ticks

        ymin = min(y)
        ymax = max(y)
        dy = (ymax-ymin)/ny_ticks

        plt.hist2d(x, y, bins=60, norm=LogNorm())
        plt.colorbar(sc, ticks=[range(int(val_min), int(val_max+1), dval)], label=cbar_label)


        ax.set_xlabel(obs_label(obs_list[0]))
        ax.set_ylabel(obs_label(obs_list[1]))
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.xaxis.major.locator.set_params(nbins=6)
        ax.yaxis.major.locator.set_params(nbins=6)
        ax.margins(0.02)
        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)
        #plt.show()

def plot_3D_hist(fig, m, colors, val_min, val_max, obs_means,  cbar_label,
    obs_list, nx_ticks=6, ny_ticks=6, bins=20, cmap=cm.rainbow):

    fig = plt.figure(fignum, figsize=(4,4))
    ax = fig.add_subplot(111, projection='3d',azim=10, elev=50)

    x = obs_means[m,0]
    x[np.isnan(x)] = 0
    y = obs_means[m,1]
    y[np.isnan(y)] = 0

    hist, xedges, yedges = np.histogram2d(x, y, bins=bins)

    elements = (len(xedges) - 1) * (len(yedges) - 1)
    xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1])

    xmin = min(x)
    xmax = max(x)
    dx = (xmax - xmin)/nx_ticks
    bdx = (xmax - xmin)/bins

    ymin = min(y)
    ymax = max(y)
    dy = (ymax-ymin)/ny_ticks
    bdy = (ymax-ymin)/bins

    xpos = xpos.T.flatten()
    ypos = ypos.T.flatten()
    zpos = np.zeros(elements)
    bdx =  1 * bdx * np.ones_like(zpos)
    bdy =  1 * bdy * np.ones_like(zpos)
    dz = np.log10(hist.flatten())

    nrm=mpl.colors.Normalize(0,max(dz))
    colors=cmap(nrm(dz))

    ax.bar3d(xpos, ypos, zpos, bdx, bdy, dz, color=colors)

    cax, kw = mpl.colorbar.make_axes(ax,shrink=.75,pad=.02)
    cb1 = mpl.colorbar.ColorbarBase(cax, cmap=cm.rainbow, norm=nrm)

    ax.set_xlabel(obs_label(obs_list[0]))
    ax.set_ylabel(obs_label(obs_list[1]))
    ax.set_zlabel(r'$\log_{10}(\mathrm{counts})$')

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.xaxis.major.locator.set_params(nbins=6)
    ax.yaxis.major.locator.set_params(nbins=6)
    ax.margins(0.02)

    #plt.show()

if __name__ == '__main__':
    output_dir = 'fock_IC'
    data_repo = '/mnt/ext0/qca_output/'+output_dir+'/data/'
    #data_repo = None

    typ = 'hist3D'
    modes = ['alt','sweep','block']
    #uID = '_somethin_'
    uID = ''
    IC_label = 'fock'
    Ls = [11,12,13,14]
    degs = range(0, 105, 15)
    Ss = range(1,15)

    #ICs = ['G', 'W', 'c2_B0-1_0', 'c2_B0-1_1', 'c2_B0-1_2', 'c2_B0-1_3' ]
    ICs = ['f0', 'c3_f1', 'c3_f0-1', 'c3_f0-2', 'c3_f0-1-2']
    #ICs = ['r5-10', 'r5-20', 'r5-30']

    #color_by_list = ['mode', 'th', 'S', 'IC', 'L']
    color_by_list = ['th']

    #obs_list_list = [['ND', 'Y']]
    obs_list_list = [['ND', 'Y'], ['CC', 'Y'], ['ND', 'CC'], ['ND', 's'], 
                     ['CC','s'], ['Y', 's']]


    # outer params
    fixed_params_dict = {
                'output_dir' : [output_dir],
                'T'   : [1000],
                'BC'  : ['1_00'],
                 }

    #inner params
    var_params_dict = {
                'L'   : Ls,
                'V' : ['HP_'+str(deg) for deg in degs],
                'IC'  : ICs,
                'S'   : Ss,
                'mode': modes
                 }

    # define the colormap
    cmap = cm.rainbow
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

    params_list_list = io.make_params_list_list(fixed_params_dict, var_params_dict)
    fignum=0
    for color_by in color_by_list:
        for obs_list in obs_list_list:
            M, N = params_list_list.shape
            J = len(obs_list)
            obs_means = np.zeros((M, J, N))
            colors = np.zeros((M, J, N))
            if typ=='scatter':
                fig = plt.figure(fignum, figsize=(3, 2.83))
                ax = fig.add_subplot(111)
            fignum += 1
            for m, params_list in enumerate(params_list_list):
                for j, obs_key in enumerate(obs_list):
                    for n, params in enumerate(params_list):
                        val, val_min, val_max, cbar_label = set_color(color_by, params)

                        if data_repo is not None:
                            sname = io.sim_name(params)
                            res_path = data_repo + sname + '_v0.hdf5'
                        else:
                            res_path = io.default_file_name(params, 'data', '.hdf5')
                        res = h5py.File(res_path)
                        obs = res[obs_key][500:]
                        obs_mean = np.nanmean(obs)
                        obs_means[m,j,n] = obs_mean
                        colors[m,j,n] = val

                if typ == 'scatter':
                    plot_scatter(fig, ax, m, obs_means, colors, val_min, val_max,
                            cbar_label, obs_list, cmap)
                    clip = True
                if typ == 'hist3D':
                    plot_3D_hist(fignum, m, colors, val_min, val_max, obs_means, cbar_label, obs_list)
                    clip = False

    bn = io.base_name(fixed_params_dict['output_dir'][0], 'plots/scatters') 

    fname = IC_label + '_'+'-'.join(modes) + '_L'+'-'.join(map(str,Ls)) + uID +typ+'.pdf'

    print(fname)
    io.multipage(bn+fname, clip=clip)



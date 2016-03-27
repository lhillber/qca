#!/usr/bin/python3

# =============================================================================
# This script computes quantities of interest for various known quantum states
#
# By Logan Hillberry
# =============================================================================

from math import log, pi

import h5py
import numpy as np
import scipy as sp

import simulation.states          as ss
import simulation.measures        as ms
import simulation.matrix          as mx
import simulation.fio             as io
import simulation.plotting             as pt
import simulation.networkmeasures as nm

import matplotlib        as mpl
import matplotlib.pyplot as plt
import matplotlib.cm     as cm
import matplotlib.colors as mcolors


# default plot font
# -----------------
font = {'size':12, 'weight' : 'normal'}
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rc('font',**font)


def mutual_info_calc(state):
    L = int(log(len(state),2))
    sj = np.zeros(L)
    for j in range(L):
        rj =mx.rdms(state, [j])
        sj[j] = ms.vn_entropy(rj)
    sjk = np.zeros((L,L))
    mjk = np.zeros((L,L))
    for j in range(L):
        for k in range(j, L):
            if j == k:
                sjk[j,j] = sj[j]
                mjk[j,j] = 0 
            else:
                rjk =mx.rdms(state, [j, k])
                sjk[j,k] = ms.vn_entropy(rjk)
                mjk[j,k] = 0.5 * (sj[j] + sj[k] - sjk[j,k])
                mjk[k,j] = 0.5 * (sj[j] + sj[k] - sjk[j,k])
    return mjk

def get_state(L, IC):
    if IC == 'sarray':
        if L%2 == 0:
            l=2
            ic = 'B0-1_0'
            singlet = ss.make_state(l, ic)
            state = mx.listkron([singlet]*int(L/2))
            cont = False
        else:
            print('singlet array requires even L: L='+str(L))
            cont = True
            state = 0
    else:
        state = ss.make_state(L, IC)
        cont = False
    return state, cont

def make_typical_nm(L_list, IC_list, importQ=False):
    bnd = io.base_name('typical_nm', 'data')
    fnamed = 'typical_net_meas.hdf5'
    if importQ:
        typical_nm = h5py.File(bnd+fnamed)
        return typical_nm
    else:
        L_list_cp = L_list.copy()
        typical_nm = dict()
        for IC in IC_list:
            typical_nm[IC] = dict()
            ND_list, CC_list, Y_list = [], [], []
            for i, L in enumerate(L_list):
                state, cont = get_state(L, IC)
                if cont:
                    L_list_cp.remove(L)
                    continue
                mjk = mutual_info_calc(state)
                ND = nm.density(mjk)
                ND_list.append(ND)
                CC = nm.clustering(mjk)
                CC_list.append(CC)
                Y = nm.disparity(mjk)
                Y_list.append(Y)
                print(IC, L, ND)
            typical_nm[IC]['ND'] = np.array(ND_list)
            typical_nm[IC]['CC'] = np.array(CC_list)
            typical_nm[IC]['Y']  = np.array(Y_list)
            typical_nm[IC]['L']  = np.array(L_list_cp)
            io.write_hdf5(bnd + fnamed, typical_nm, force_rewrite=True)
        return typical_nm


def colorbar_index(colorbar_label, ncolors, cmap, val_min, val_max):
    cmap = cmap_discretize(cmap, ncolors)
    mappable = cm.ScalarMappable(cmap=cmap)
    mappable.set_array([])
    mappable.set_clim(-0.5, ncolors+0.5)
    colorbar = plt.colorbar(mappable)
    colorbar.set_ticks(np.linspace(0, ncolors, ncolors))
    colorbar.set_ticklabels(list(map(int, np.linspace(int(val_min),
        int(val_max), ncolors))))
    colorbar.set_labe(colorbar_label)


def cmap_discretize(cmap, N):
    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)
    colors_i = np.concatenate((np.linspace(0, 1., N), (0.,0.,0.,0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N+1)
    cdict = {}
    for ki, key in enumerate(('red','green','blue')):
        cdict[key] = [ (indices[i], colors_rgba[i-1,ki], colors_rgba[i,ki])
                       for i in range(N+1) ]
    # Return colormap object.
    return mcolors.LinearSegmentedColormap(cmap.name + "_%d"%N, cdict, 1024)


def make_c_list(IC, L_list, typical_nm):
    c_full = np.array(L_list)/max(L_list)
    c = typical_nm[IC]['L']/max(L_list)
    return c, c_full


def scatter_plot(fig, typical_nm, obs_list, IC, c, c_full, i, cmap=cm.jet):
    ax = fig.add_subplot(1,3,i+1)
    x_key = obs_list[0]
    y_key = obs_list[1]
    x = typical_nm[IC][x_key]
    y = typical_nm[IC][y_key]

    ax.scatter(x, y, 
            marker=IC_marker_dict[IC],
            label=pretty_names_dict[IC],
            s=70, alpha=1, c=c, cmap=cmap, linewidth='0.00000001',
            vmin=min(c_full), vmax=max(c_full))

    ax.set_xlabel(pretty_names_dict[x_key])
    ax.set_ylabel(pretty_names_dict[y_key])
    ax.xaxis.major.locator.set_params(nbins=6)
    ax.yaxis.major.locator.set_params(nbins=6)
    ax.margins(0.1)
    return ax


def plot_config(fig, ax, L_list, IC_list):
    legend = ax.legend(frameon=False, scatterpoints=1,
            loc='lower right', fontsize=11, bbox_to_anchor=(2.6, -0.08),
            markerscale=0.8, handletextpad=0.1)
    for i in range(len(IC_list)):
        legend.legendHandles[i].set_color('k')
    colorbar_index(r'$L$', ncolors=len(L_list), cmap=cmap,
            val_min=min(L_list), val_max=max(L_list))
    fig.subplots_adjust(wspace=0.5, right=0.8)

def plot_scatters(L_list, IC_list, typical_nm, fignum=1, saveQ=True):
    fig = plt.figure(fignum, figsize=(9,3))
    for IC in IC_list:
        c, c_full = make_c_list(IC, L_list, typical_nm)
        for i, obs_list in enumerate(obs_list_list):
            ax = scatter_plot(fig, typical_nm, obs_list, IC, c, c_full, i, cmap=cmap)
    plot_config(fig, ax, L_list, IC_list)

    if saveQ:
        bnp = io.base_name('typical_nm', 'plots')
        fnamep = 'typical_net_meas.pdf'
        io.multipage(bnp + fnamep, clip=True)
    else:
        plt.show()

if __name__ == '__main__':

    L_list=[4, 6, 8, 10,12,14,16, 18, 20, 22]
    IC_list=['G', 'W', 'c2_B0-1_3', 'sarray']
    obs_list_list = [['ND', 'CC'], ['ND', 'Y'], ['CC', 'Y']]


    cmap = cm.gist_rainbow_r
    marker_list = ['s','o','^','v']
    IC_marker_dict = dict(zip(IC_list, marker_list))
    pretty_names_dict = {'ND' : r'$ND$',
                         'CC' : r'$CC$',
                          'Y' : r'$Y$',
                          'G' : r'$|GHZ\rangle$',
                          'W' : r'$|W\rangle$',
                 'c2_B0-1_3'  : 'localized\n  singlet',
                     'sarray' : 'singlet\n array'
                        }


    typical_nm = make_typical_nm(L_list, IC_list, importQ=True)
    #plot_scatters(L_list, IC_list, typical_nm)




    def plot_loglog_fit(x, y, ax):
        m, b = np.polyfit(np.log10(x), np.log10(y), 1)
        def lin_fit(x):
            x = np.array(x)
            return m*x + b

        # chi squared of fit
        chi2 = np.sum((10**lin_fit(np.log10(x)) - y) ** 2)

        # plot
        ax.plot(x, 10**lin_fit(np.log10(x)),'-r', label='fit')
        ax.loglog(x, y,'xk',markersize=4, linewidth='0.01', label='data')

        ax.set_title(' $m= {:.3f}$ $b= {:.3f}$ \n$\chi^2 =\
                {:s}$'.format(m, b, pt.as_sci(chi2, 3)),fontsize=10)
        ax.set_xlim([3, max(x)+4])

    def plot_nmVsL(L_list, IC_list, typical_nm, fignum=2, saveQ=False):
        fig = plt.figure(fignum, figsize=(7,8))
        obs_list_flat = list(set([obs for obs_list in obs_list_list 
                                      for obs in obs_list]))
        M = len(IC_list)
        N = len(obs_list_flat)
        for m, IC in enumerate(IC_list):
            for n, obs in enumerate(obs_list_flat):
                i = 1+m*(M-1) + n
                ax = fig.add_subplot(M,N,i)
                x = typical_nm[IC]['L']
                y = typical_nm[IC][obs]
                if m != M-1:
                    pass
                    #plt.setp([ax.get_xticklabels()], visible=False)
                if np.std(y)<1e-10:
                    if np.mean(y) < 1e-14:
                        ax.plot(x,[0]*len(x))
                        ax.set_title(pretty_names_dict[obs]+r'$ = 0$',
                                fontsize=10)
                        plt.setp([ax.get_xticklabels()], visible=True)
                        ax.xaxis.major.locator.set_params(nbins=6)
                        ax.yaxis.major.locator.set_params(nbins=5)
                    else:
                        ax.plot(x,[np.mean(y)]*len(x))
                        ax.set_title(pretty_names_dict[obs]+r'$ = '+
                                str(np.mean(y))+r'$', fontsize=10)
                        plt.setp([ax.get_xticklabels()], visible=True)
                        ax.xaxis.major.locator.set_params(nbins=6)
                        ax.yaxis.major.locator.set_params(nbins=5)
                else:
                    plot_loglog_fit(x, y, ax)
                    #ax.semilogx(x,y)
                if n == 0:

                    ax.text(0-0.5, 1.2, pretty_names_dict[IC],
	                transform=ax.transAxes, fontsize=12)
                    pass
                    #ax.set_title(pretty_names_dict[IC]+':', x=-0.05, y=1.17)
                ax.set_ylabel(pretty_names_dict[obs])
                if m == M-1:
                    ax.set_xlabel(r'$L$')
        fig.subplots_adjust(hspace=1, wspace=0.8)
        if saveQ:
            bnp = io.base_name('typical_nm', 'plots')
            fnamep = 'nmVsL_loglog.pdf'
            io.multipage(bnp + fnamep, clip=False)
        else:
            plt.show()


    plot_nmVsL(L_list, IC_list, typical_nm, fignum=2, saveQ=True)

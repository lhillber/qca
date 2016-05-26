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

from itertools import cycle
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

def pretty_names_dict(key):
    pn = {           'ND' : r'$\mathcal{D}$',
                     'CC' : r'$\mathcal{C}$',
                      'Y' : r'$\mathcal{Y}$',
                      'G' : r'$|GHZ\rangle$',
                      'W' : r'$|W\rangle$',
             'c2_B0-1_3'  : r'$|LS\rangle$',
                 'sarray' : r'$|SA\rangle$',
                      'R' : 'RS',
                      'N' : 'RN'
                    }
    if key[0] == 'R':
        key = 'R'
    return pn[key]

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
        else:
            print('singlet array requires even L: L='+str(L))
            l=2
            ic = 'B0-1_0'
            singlet = ss.make_state(l, ic)
            state = mx.listkron([singlet]*int((L-1)/2) + [ss.bvecs['0']])
            state = None
    else:
        state = ss.make_state(L, IC)
    return state

def random_network(L):
    network = np.random.rand(L,L)
    for i in range(L):
        for j in range(i,L):
            network[i,j] = network[j,i]
    np.fill_diagonal(network, 0.0)
    return network

def make_typical_nm(L_list, IC_list, importQ=False, nsamp=300):
    bnd = io.base_name('typical_nm', 'data')
    fnamed = 'typical_net_meas_rand.hdf5'
    if importQ:
        typical_nm = h5py.File(bnd+fnamed)
        return typical_nm
    else:
        L_list_cp = L_list.copy()
        typical_nm = dict()
        print('IC', 'L', 'ND', 'CC', 'Y')
        for IC in IC_list:
            typical_nm[IC] = dict()
            ND_list, CC_list, Y_list = [], [], []
            for i, L in enumerate(L_list):
                if IC == 'R':
                    ND, CC, Y = 0, 0 ,0
                    mjk = np.zeros((L,L))
                    for s in range(nsamp):
                        IC_tmp = IC +str(s)
                        state = get_state(L, IC_tmp)
                        mjk = mutual_info_calc(state)
                        ND  += nm.density(mjk)/nsamp
                        CC  += nm.clustering(mjk)/nsamp
                        Y   += nm.disparity(mjk)/nsamp

                elif IC == 'N':
                    ND, CC, Y = 0, 0 ,0
                    for s in range(nsamp):
                        mjk = random_network(L)
                        ND  += nm.density(mjk)/nsamp
                        CC  += nm.clustering(mjk)/nsamp
                        Y   += nm.disparity(mjk)/nsamp
                else:
                    state = get_state(L, IC)
                    if state is not None:
                        mjk = mutual_info_calc(state)
                        ND  = nm.density(mjk)
                        CC  = nm.clustering(mjk)
                        Y   = nm.disparity(mjk)

                if state is not None:
                    Y_list.append(Y)
                    CC_list.append(CC)
                    ND_list.append(ND)
                    print(IC, L, ND, CC, Y)
            typical_nm[IC]['ND'] = np.array(ND_list)
            typical_nm[IC]['CC'] = np.array(CC_list)
            typical_nm[IC]['Y']  = np.array(Y_list)
            typical_nm[IC]['L']  = np.array(L_list_cp)

        typ_copy = typical_nm.copy()
        io.write_hdf5(bnd + fnamed, typical_nm, force_rewrite=True)
        return typ_copy


def colorbar_index(colorbar_label, ncolors, cmap, val_min, val_max):
    cmap = cmap_discretize(cmap, ncolors)
    mappable = cm.ScalarMappable(cmap=cmap)
    mappable.set_array([])
    mappable.set_clim(-0.5, ncolors+0.5)
    cax = plt.gcf().add_axes([0.935, 0.25, 0.013, 0.68])
    colorbar = plt.colorbar(mappable, cax=cax)
    colorbar.set_ticks(np.linspace(0, ncolors, ncolors))
    colorbar.set_ticklabels(list(map(int, np.linspace(int(val_min),
        int(val_max), ncolors))))
    colorbar.set_label(colorbar_label, rotation=0, y=0.55, labelpad=1.8)


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
    c = typical_nm[IC]['L'][::]/max(L_list)
    return c, c_full


def scatter_plot(fig, typical_nm, obs_list, IC, c, c_full, i, cmap=cm.jet,
        letter='(a)'):
    ax = fig.add_subplot(1,3,i+1)
    x_key = obs_list[0]
    y_key = obs_list[1]
    x = typical_nm[IC][x_key]
    y = typical_nm[IC][y_key]
    vmax = max(c_full)
    vmin = min(c_full)
    if len(x) == 2:
        c = c[1:4:2]
    ax.scatter(x, y, 
            marker=IC_marker_dict[IC],
            label=pretty_names_dict(IC),
            s=70, alpha=1, c=c, cmap=cmap, linewidth='0.00000001',
            vmin=vmin, vmax=vmax)

    ax.set_xlabel(pretty_names_dict(x_key), labelpad=1)
    ax.set_ylabel(pretty_names_dict(y_key), labelpad=1.5)
    ax.set_title(letter)
    ax.set_xlim(left=-0.05, right=0.6)
    ax.xaxis.major.locator.set_params(nbins=4)
    ax.yaxis.major.locator.set_params(nbins=4)
    ax.margins(0.1)
    return ax


def plot_config(fig, ax, L_list, IC_list):
    legend = ax.legend(frameon=False, scatterpoints=1,
            loc='lower right', fontsize=12, bbox_to_anchor=(0.25, 0.10, 0.1, 1),
            markerscale=0.7, handletextpad=-0.3, mode='expand',
            labelspacing=0.085)
    for i in range(len(IC_list)):
        legend.legendHandles[i].set_color('k')
    colorbar_index(r'$L$', ncolors=len(L_list), cmap=cmap,
            val_min=min(L_list), val_max=max(L_list))
    fig.subplots_adjust(wspace=0.45, left=0.08, bottom=0.2, top=0.895, right=0.92)

def plot_scatters(L_list, IC_list, typical_nm, fignum=1, saveQ=True):
    fig = plt.figure(fignum, figsize=(6.5, 2.1))
    letters = ['(a)', '(b)', '(c)']
    for IC in IC_list:
        c, c_full = make_c_list(IC, L_list, typical_nm)
        for i, obs_list in enumerate(obs_list_list):
            ax = scatter_plot(fig, typical_nm, obs_list, IC, c, c_full, i, 
                    cmap=cmap, letter=letters[i])
    plot_config(fig, ax, L_list, IC_list)

    if saveQ:
        bnp = io.base_name('typical_nm', 'plots')
        fnamep = 'typical_net_meas.pdf'
        io.multipage(bnp + fnamep, clip=False)
    else:
        plt.show()


def MI_W(L):
   return  1/2*(2*(-(1-1/L)*np.log2(1-1/L) - 1/L*np.log2(1/L))\
    +2/L*np.log2(2/L) + (1-2/L)*np.log2(1-2/L))

def MI_GHZ(L):
   return -(1-1/L)*np.log2(1-1/L) - 1/L*np.log2(1/L)

def nm_exact(IC, obs, L):
    f_map = {'G':
                 {'ND' : 0.5 +0*L,
                  'CC' : 0.5 +0*L,
                   'Y' : 1/(L-1)},
             'W':
                 {'ND' : MI_W(L),
                  'CC' : MI_W(L),
                   'Y' : 1/(L-1)},
     'c2_B0-1_3':
                 {'ND' : 2/(L*(L-1)),
                  'CC' : 0+0*L,
                   'Y' : 2/L},
        'sarray':
                 {'ND' : 1/(L-1),
                  'CC' : 0+0*L,
                   'Y' : 1.0 +0*L},

            'R': {'ND' : 0,
                  'CC' : 0,
                   'Y' : 0}
            }
    return f_map[IC][obs]


def plot_semilogy(x, y, ax):
    ax.semilogy(x,y)
    plt.setp([ax.get_xticklabels()], visible=True)
    ax.xaxis.major.locator.set_params(nbins=6)


def plot_loglog_fit(x, y, ax):
    m, b = np.polyfit(np.log10(x), np.log10(y), 1)
    def lin_fit(x):
        x = np.array(x)
        return m*x + b

        # chi squared of fit
        chi2 = np.sum((10**lin_fit(np.log10(x)) - y) ** 2)

        # plot
        ax.plot(x, 10**lin_fit(np.log10(x)),'-r', label='fit')
        ax.loglog(x, y,'sk', alpha=0.6, markersize=4, linewidth='0.01', label='data')

        ax.set_title(' $m= {:.3f}$,  $b= {:.3f}$ \n$\chi^2 =\
                {:s}$'.format(m, b, pt.as_sci(chi2, 3)),fontsize=10)
        ax.set_xlim([3, max(x)+4])


def plot_nmVsL(L_list, IC_list, typical_nm, fignum=2, saveQ=False):
    fig = plt.figure(fignum, figsize=(7,8.5))
    #obs_list_flat = list(set([obs for obs_list in obs_list_list 
    #                              for obs in obs_list]))
    obs_list_flat = ['ND', 'CC', 'Y']
    letters = {0:'a', 1:'b', 2:'c', 3:'d'}
    M = len(IC_list)
    N = len(obs_list_flat)
    for m, IC in enumerate(IC_list):
        if IC[0] == 'R':
            IC = 'R'
        for n, obs in enumerate(obs_list_flat):
            i = 1+m*(M-1) + n
            ax = fig.add_subplot(M,N,i)
            x = typical_nm[IC]['L']
            xs = np.linspace(min(x), max(x), 100)
            y = typical_nm[IC][obs]
            ys = nm_exact(IC, obs, xs)
            #error = max(np.abs(y - nm_exact(IC, obs, x[:])))
            if m != M-1:
                pass
                #plt.setp([ax.get_xticklabels()], visible=False)
            ax.set_ylim(0, max(y)*5/4)
            if np.std(y)<1e-10:
                ax.set_ylim(np.mean(y)-0.1, np.mean(y)+0.1)
            if IC == 'W':
                ax.set_ylim(top=0.4)
            if IC == 'c2_B0-1_3':
                if obs == 'Y':
                    ax.set_ylim(top=0.6)
            #plot_loglog_fit(x, y, ax)
            #plot_semilogy(x, y, ax)
            ax.plot(x,y, 'ok', alpha=0.6, markersize=4)
            plt.setp([ax.get_xticklabels()], visible=True)
            ax.xaxis.major.locator.set_params(nbins=6)
            ax.yaxis.major.locator.set_params(nbins=3)
            ax.plot(xs, ys, '-k', alpha=0.7)

            #ax.set_title('$\mathrm{error}_\mathrm{max} = 0$' , fontsize=12)
            #if error != 0:
            #    ax.set_title('$\mathrm{error}_\mathrm{max} = ' +
            #            pt.as_sci(error,1)+'$', fontsize=12)
            if n == 0:
                ax.text(-0.5, 1.1, '('+letters[m]+')  '+pretty_names_dict(IC),
                    transform=ax.transAxes, fontsize=13)

            ax.set_ylabel(pretty_names_dict(obs))
            if m == M-1:
                ax.set_xlabel(r'$L$')

            ax.margins(0.1)
    fig.subplots_adjust(hspace=0.4, wspace=0.7, left=0.13, right=.98,
            top=.94, bottom=.05)
    if saveQ:
        bnp = io.base_name('typical_nm', 'plots')
        fnamep = 'nmVsL.pdf'
        io.multipage(bnp + fnamep, clip=False)
    else:
        plt.show()

if __name__ == '__main__':
    importQ=True
    L_list=[11,14,17,20]
    IC_list=['G', 'W', 'c2_B0-1_3', 'sarray', 'R','N']
    obs_list_list = [['ND', 'CC'], ['ND', 'Y'], ['CC', 'Y']]
    cmap = cm.gist_rainbow_r
    marker_list = ['s','o','^','v','*','d']
    IC_marker_dict = dict(zip(IC_list, cycle(marker_list)))

    typical_nm = make_typical_nm(L_list, IC_list, importQ=importQ)
    plot_scatters(L_list, IC_list, typical_nm, fignum=1, saveQ=True)

    #plot_nmVsL(L_list, IC_list, typical_nm, fignum=2, saveQ=True)

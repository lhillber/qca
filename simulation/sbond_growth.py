#!/usr/bin/python3
# =============================================================================
# Growth of bond entropy in Goldilocks rules 1, 6, 9, and 14.
# By Logan Hillberry
# =============================================================================


import h5py
import numpy as np
from itertools import product, cycle
import matplotlib.pyplot as plt
import matplotlib as mpl
import simulation.fio as io
import simulation.plotting as pt
import matplotlib.ticker as ticker

font = {'size':12, 'weight' : 'normal'}
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rc('font',**font)



# locate the data you're interested in
output_dir = 'fock_IC'
mount_point = '/mnt/ext0/'
data_repo = mount_point + 'qca_output/' + output_dir + '/data/'
#data_repo = None


# describe the simulations you'd like to load
Ls = [11, 13, 15, 17, 19, 21]
Ss = [1,6,9,14]
# params looped through at outer layer
fixed_params_dict = {
            'output_dir' : [output_dir],
            'T'          : [1000],
            'BC'         : ['1_00'],
            'mode'       : ['alt'],
            'S'          : Ss
             }

# params looped through at inner layer
var_params_dict = {
            'L'   : Ls,
            'V'   : ['HP_'+str(deg) for deg in [0]],
            'IC'  : ['c3_f1'],
             }


def rule_plot(params_list_list, show_S=[6], show_L=[15]):
    fignum = 0
    c_list = ['c', 'orange', 'crimson', 'limegreen']
    c_list = cycle(c_list)
    m_list = ['s','o','^','d']
    m_list = cycle(m_list)
    fig = plt.figure(fignum,figsize=(3,3))
    ax = fig.add_subplot(1,1,1)
    for m, params_list in enumerate(params_list_list):
        avg_list = []
        for n, params in enumerate(params_list):
            if data_repo is not None:
                sname = io.sim_name(params)
                data_path = data_repo + sname + '_v0.hdf5'
            else:
                sname = io.sim_name(params)
                data_path = io.default_file_name(params, 'data', '.hdf5')
            data_set = h5py.File(data_path)
            L = params['L']
            S = params['S']
            T = params['T']
            sb = data_set['sbond'][::] 

            if L == 21:
                center_sb = data_set['scenter']
                print(center_sb[0])
            else:
                sb[::, int(L/2)] = np.zeros(params['T']+1)
                sb /= [min(c+1, L-c-1) for c in range(L-1)]

                center_sb = sb[::, int(L/2)]

                avg_sb = np.mean(sb, axis=1)
                avg_avg_sb = np.mean(avg_sb[500::])

            avg_center_sb = np.mean(center_sb[500::])
            avg_list.append([L, S, avg_avg_sb])

            #S_plot(ax, c_list, center_sb, r'$L={}$'.format(L), r'$S={}$'.format(S))

        L_avg_plot(ax, c_list, m_list, avg_list, r'$S={}$'.format(S))

    bn = io.base_name(output_dir, 'plots/sbond/')
    #fn = 'S1_sbond_center_growth'+'.pdf'
    fn = 'S6_sbond_center_growth'+'.pdf'
    io.multipage(bn+fn)
    #plt.show()

def S_plot(ax, c_list, sb, label, title):
    c = next(c_list)
    ax.plot(range(len(sb)), sb, label=label, c=c)
    ax.set_xlabel('Time step')
    ax.set_ylabel(r'$s^{\mathrm{bond}}_{\mathrm{center}}$')
    legend = ax.legend(ncol=1, borderpad=0.2,
            loc='lower right', fontsize=11,
            handlelength=1, handletextpad=-0.05,
            labelspacing=0.0, columnspacing=0.5)
    ax.set_title(title)

def L_avg_plot(ax, c_list, m_list, avg_list, label):
    sc = next(c_list)
    m = next(m_list)
    avg_list = np.asarray(avg_list)
    ax.plot(avg_list[::, 0], avg_list[::, 2], c=sc, markeredgecolor=sc,
            label=label, lw=1.3, markeredgewidth=2,
    marker=m, markerfacecolor='None')

    ax.set_ylabel(r'$s^{\mathrm{bond}}_{\mathrm{center}}$')
    ax.set_xlabel(r"L")
    ax.legend(loc='upper left', fontsize=12,  numpoints=1, handlelength=1,
            ncol=2, handletextpad=0.35, labelspacing=0.0, columnspacing=0.7)
    ax.set_ylim([0.2, 1.07])

    ax.set_xticks([11,13,15,17,19, 21])
    #ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    #ax.xaxis.major.locator.set_params(nbins=9)
    ax.yaxis.major.locator.set_params(nbins=6)
    ax.margins(0.07)


if __name__ == '__main__':
    params_list_list = io.make_params_list_list(fixed_params_dict, var_params_dict)
    rule_plot(params_list_list)

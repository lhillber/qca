#!/usr/bin/python3
import matplotlib.pyplot as plt
import matplotlib as mpl
import simulation.plotting as pt
import matplotlib.gridspec as gridspec
import simulation.fio as io
import simulation.measures as ms
import numpy as np
import h5py

font = {'size':10, 'weight' : 'normal'}
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rc('font',**font)


def make_V_name(V):
    if V in ['X', 'Y', 'Z']:
        name = '\sigma^{}'.format(V.lower())
    else:
        name = V
    if len(name.split('_')) == 2:
        name = '('.join(name.split('_'))+'^\circ)'
    else:
        name = '('.join(name.split('_'))
    return name

def make_mode_name(mode):
    if mode is 'sweep':
        name = '\mathrm{SWP}'
    elif mode is 'alt':
        name = '\mathrm{ALT}'
    elif mode is 'block':
        name = '\mathrm{BLK}'
    else:
        name =''
    return name

def make_U_name(mode, S, V):
    S = str(S)
    name = r'$U^{%s}_{%s}(%s)$' % (make_mode_name(mode), S, make_V_name(V))

    return name

if __name__ == '__main__':
    for modee in ['alt']:
        fixed_params_dict = {
                    'output_dir' : ['fock_IC'],
                    'L' : [17],
                    'T' : [1000],
                    'IC': ['f0'],
                    'BC': ['1_00'],
                    'mode': [modee],
                    'S' : [12]
                     }

        var_params_dict = {
                    'V' : ['HP_0']
                     }


        params_list_list = io.make_params_list_list(fixed_params_dict, var_params_dict)

        span = [0, 1000]
        nr = len(params_list_list)
        gs = gridspec.GridSpec(nr,3)
        fig = plt.figure(figsize=(4.0,3.7))
        for r, params_list in enumerate(params_list_list):
            for c, params in enumerate(params_list):
                mode = params['mode']
                S = params['S']
                V = params['V']
                T = params['T']
                L = params['L']
                output_dir = params['output_dir']

                ax = fig.add_subplot(gs[r,c])

                print(io.default_file_name(params, 'data', '.hdf5'))
                res = h5py.File(io.default_file_name(params, 'data', '.hdf5'))
                exp = ms.get_diag_vecs(res['zz'][::])

                Ptj = exp
                Ptj = res['s'][::]

                Ptj = ((1.0 - exp)/2.0)[span[0]:span[1], 0:L]
                xlabel, ylabel = '', ''
                xtick_labels = False

                title = make_U_name(mode, S, V)
                y_tick_labels = ['']*(span[1] - span[0])
                x_tick_labels = ['']*L
                xlabel = 'Site'
                if r == 0:
                    xlabel =''
                if r == nr - 1:
                    title = ''
                    x_tick_labels = range(L)
                if c == 0:
                    ylabel = 'Iteration'
                    y_tick_labels = range(*span)

                im = ax.imshow( Ptj,
                            origin = 'lower',
                            vmax = 1,
                            interpolation = 'none',
                            aspect = 'auto',
                            rasterized = True)

                ax.set_title(title, fontsize=9)
                ax.set_ylabel(ylabel)
                ax.set_xlabel(xlabel)

                n_xticks = 3
                delta = max(1, int(len(x_tick_labels)/n_xticks))
                ax.set_xticks(range(0, len(x_tick_labels), delta ))
                ax.set_xticklabels(x_tick_labels[::delta])

                n_yticks = 6
                delta = max(1, int(len(y_tick_labels)/n_yticks))
                ax.set_yticks(range(0, len(y_tick_labels), delta ))
                ax.set_yticklabels(y_tick_labels[::delta])

            im_ext = im.get_extent()
            box = ax.get_position()
           # cax = plt.axes([box.x1+0.01, box.y0, 0.02, box.height - 0.03])
           # cb = plt.colorbar(im, cax = cax, ticks = [0.0,0.5, 1.0])
           # cb.ax.tick_params(labelsize=9)
           # cb.set_label(r'$s_j$', rotation=0, labelpad = -24, y=1.12)
           # cb.set_label(r'$P_1(j,t)$', rotation=0, labelpad = -22, y=1.10)
        #cax = plt.axes([box.x1-0.05, box.y0+0.2, 0.02, box.height + 0.02])
        #cb = plt.colorbar(im, cax = cax)
        #cb.set_label(r'$P_1(j,t)$', rotation=0, labelpad = -18, y=1.13)
        fig.subplots_adjust(wspace=-0.4, hspace=0.1)

        bn = io.base_name(output_dir,'plots')
        #plt.savefig(bn+params['mode']+'_traj_comp.pdf', bbox_inches='tight', dpi=300)
        plt.show()

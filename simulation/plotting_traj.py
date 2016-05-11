#!/usr/bin/python3
import matplotlib.pyplot as plt
import matplotlib as mpl
import simulation.plotting as pt
import matplotlib.gridspec as gridspec
import simulation.fio as io
import simulation.measures as ms
import numpy as np
import h5py

font = {'size':12, 'weight' : 'normal'}
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
    output_dir = 'fock_IC'
    data_repo = '/mnt/ext0/qca_output/'+output_dir+'/data/'
    #data_repo=None
    for modee in ['alt']:
        fixed_params_dict = {
                    'output_dir' : [output_dir],
                    'L' : [19],
                    'T' : [1000],
                    'IC': ['c3_f1'],
                    'BC': ['1_00'],
                    'mode': [modee],
                    'V' : ['HP_0']
                     }

        var_params_dict = {
                    'S' : range(1,16)
                     }


        params_list_list = io.make_params_list_list(fixed_params_dict, var_params_dict)

        span = [0, 61]
        #nr = len(params_list_list)
        nr = 3
        nc = 5
        fig = plt.figure(figsize=(6.5,8.5))
        i = 0
        for r, params_list in enumerate(params_list_list):
            for c, params in enumerate(params_list):
                mode = params['mode']
                S = params['S']
                V = params['V']
                T = params['T']
                L = params['L']
                output_dir = params['output_dir']
                ax = fig.add_subplot(nr, nc, i+1)
                i+=1


                if data_repo is not None:
                    sname = io.sim_name(params)
                    res_path = data_repo + sname + '_v0.hdf5'
                else:
                    res_path = io.default_file_name(params, 'data', '.hdf5')

                res = h5py.File(res_path)
                exp = ms.get_diag_vecs(res['zz'][::])

                Ptj = exp
                #Ptj = ((1.0 - exp)/2.0)[span[0]:span[1], 0:L]

                Ptj = res['sbond'][span[0]:span[1]-1,::]
                L = len(Ptj[1])
                Ptj /= [min(c+1, L-c) for c in range(L)]

                xlabel, ylabel = '', ''
                xtick_labels = False

                #title = make_U_name(mode, S, V)
                title = r'$S=$'+str(S)
                y_tick_labels = ['']*(span[1] - span[0])
                x_tick_labels = ['']*L
                xlabel = ''
                if r == 0:
                    xlabel =''
                if i in (11, 12, 13, 14, 15):
                    x_tick_labels = range(L)
                    xlabel='Cut'
                if i in (1, 6, 11):
                    ylabel = 'Iteration'
                    y_tick_labels = range(span[0], span[1])

                im = ax.imshow(Ptj,
                            origin = 'lower',
                            vmax = 1,
                            interpolation = 'none',
                            aspect = '1',
                            rasterized = True,
                            extent=[0,L+1,0,60]
                            )

                ax.set_title(title, fontsize=10)
                ax.set_ylabel(ylabel)
                ax.set_xlabel(xlabel)

                n_xticks = 3
                delta = max(1, int(len(x_tick_labels)/n_xticks))
                ax.set_xticks(range(0, len(x_tick_labels), delta ))
                ax.set_xticklabels(x_tick_labels[::delta])

                n_yticks = 3
                delta = max(1, int(len(y_tick_labels)/n_yticks))
                ax.set_yticks(range(0, len(y_tick_labels), delta ))
                ax.set_yticklabels(y_tick_labels[::delta])
                ax.spines['bottom'].set_position(('data',0.1))
                ax.spines['left'].set_bounds(0.1, 60)
                ax.spines['right'].set_bounds(0.1, 60)
            im_ext = im.get_extent()
            box = ax.get_position()
           # cax = plt.axes([box.x1+0.01, box.y0, 0.02, box.height - 0.03])
            cax = plt.axes([box.x1-0.03, box.y0, 0.02, box.height - 0.01])
            cb = plt.colorbar(im, cax = cax, ticks = [0.0,0.5, 1.0])
           # cb.ax.tick_params(labelsize=9)
           # cb.set_label(r'$s_j$', rotation=0, labelpad = -24, y=1.12)
            cb.set_label(r'$s^{\mathrm{bond}}$', rotation=0, labelpad = -24,
                    y=1.1, fontsize=14)
           # cb.set_label(r'$P_1(j,t)$', rotation=0, labelpad = -22, y=1.10)
        #cax = plt.axes([box.x1-0.05, box.y0+0.2, 0.02, box.height + 0.02])
        #cb = plt.colorbar(im, cax = cax)
        fig.subplots_adjust(wspace=-0.5, hspace=0.2)

        bn = io.base_name(output_dir,'plots')
        #plt.savefig(bn+params['mode']+'_sbond_comp.pdf',
        #        dpi=300, clip=True)
        io.multipage(bn+params['mode']+'_sbond_comp.pdf')
        #plt.show()

#!/usr/bin/python3

import matplotlib.pyplot as plt
import matplotlib as mpl
import simulation.plotting as pt
import matplotlib.gridspec as gridspec
import simulation.fio as io
import simulation.measures as ms
import numpy as np
import h5py

from matplotlib import animation

font = {'size':10, 'weight' : 'normal'}
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rc('font',**font)



# First set up the figure, the axis, and the plot element we want to animate

# initialization function: plot the background of each frame
def init():
        grd.set_data([[],[]])
        return grd,

# animation function.  This is called sequentially
def animate(i):
        page = grid_list[i]
        grd.set_data(page)
        return grd,


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




output_dir = 'Hphase'
#data_repo = '/mnt/ext0/qca_output/'+output_dir+'/data/'
data_repo = None

deg_list = range(0, 185, 5)
fixed_params_dict = {
            'output_dir' : [output_dir],
            'L' : [21],
            'T' : [60],
            'IC': ['f0'],
            'BC': ['1_00'],
            'mode': ['alt'],
            'S' : [6]
             }

var_params_dict = {
            'V' : ['HP_' + str(deg) for deg in deg_list]
             }

params_list_list = io.make_params_list_list(fixed_params_dict, var_params_dict)


def plot_grid(grid, ax,  span=[0,60], n_xticks = 4, n_yticks = 6):
    im = ax.imshow( grid,
                    origin = 'lower',
                    cmap=plt.cm.jet,
                    vmin = np.mean(grid) - 1/2*np.std(grid),
                    vmax = np.mean(grid) + np.std(grid),
                    interpolation = 'none',
                    aspect = '1',
                    rasterized = True)

    x_tick_labels = range(len(grid[0]))
    y_tick_labels = range(*span)

    ylabel = 'Iteration'
    xlabel = 'Site'
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    delta = max(1, int(len(x_tick_labels)/n_xticks))
    ax.set_xticks(range(0, len(x_tick_labels), delta ))
    ax.set_xticklabels(x_tick_labels[::delta])

    delta = max(1, int(len(y_tick_labels)/n_yticks))
    ax.set_yticks(range(0, len(y_tick_labels), delta ))
    ax.set_yticklabels(y_tick_labels[::delta])


    #box = ax.get_position()
    #cax = plt.axes([box.x1-0.1, box.y0+0.1, 0.04, box.height - 0.19])
    #cb = plt.colorbar(im, cax = cax, ticks = [-1.0, -0.5, 0.0, 0.5, 1.0])
    #cb.ax.tick_params(labelsize=12)
    #cb.set_label(r'$\langle \sigma_j^z \rangle$', rotation=0, labelpad = -22,
    #        y=1.12)

    return im


grid_list = []
title_list = []

n_xticks = 4
n_yticks = 5
t_span=[0,1000]
slider = 100
dt=1
span = [0, dt]
for params_list in params_list_list:
    for params in params_list:
        output_dir = params['output_dir']
        mode = params['mode']
        S = params['S']
        V = params['V']
        T = params['T']
        L = params['L']
        IC = params['IC']
        title = make_U_name(mode, S, V)
        if data_repo is not None:
            sname = io.sim_name(params)
            res_path = data_repo + sname + '_v0.hdf5'
        else:
            res_path = io.default_file_name(params, 'data', '.hdf5')
        res = h5py.File(res_path)
        exp = ms.get_diag_vecs(res['zz'][::])
        exp = 0.5*(1-exp)
        s = res['s'][::]
        M = res['m']
        g2 = res['gxx'][::]
        grid = exp
        grid_list.append(grid)
        title_list.append(title)

fig = plt.figure(figsize=(2,3))
ax = fig.add_subplot(111)
im=plot_grid(grid_list[0], ax, span=span)
title = ax.set_title(title_list[0], fontsize=10)
fig.subplots_adjust(bottom=0.2)

def init():
    im.set_data(grid_list[0])
    title.set_text(title_list[0])
    return (im, title)

# animation function.  This is called sequentially
def animate(i):
    a=im.get_array()
    a=grid_list[i]
    im.set_array(a)
    im.set_clim(vmin=np.min(a), vmax=np.max(a))
    title.set_text(title_list[i])
    return (im, title)


anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=len(grid_list), interval=150)

bn = io.base_name(output_dir, 'plots')
anim.save(bn + 'L21_S6_alt_phase.avi', fps=8,
        dpi=500)

#plt.show()

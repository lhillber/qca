#!/usr/bin/python

import numpy             as np
import states            as ss
import processing        as pp
import matplotlib        as mpl
import scipy.stats       as sts
import scipy.fftpack     as spf
import matplotlib.pyplot as plt
import fio               as io

from scipy.ndimage import zoom
from mpl_toolkits.mplot3d import Axes3D

# default plot font to bold and size 16
# -------------------------------------
font = {'family':'normal', 'weight':'bold', 'size':14}
mpl.rc('font',**font)


# General plotting utilities
# ==========================

# plot time series
# ---------------0
def plot_time_series(time_series, title, label='', loc='lower right', fignum=1,
        ax=111, color='B', cut_first=0, linewidth=0.2, linestyle='-',
        marker='', markersize=1.5, markeredgecolor='B'):
    fig = plt.figure(fignum)
    fig.add_subplot(ax)
    plt.plot(range(cut_first, len(time_series)+cut_first), time_series,
            label=label, color=color, linestyle=linestyle, linewidth=linewidth, 
            marker=marker, markersize=markersize, markeredgecolor=color)
    plt.title(title)
    plt.legend(loc=loc)
    plt.tight_layout() 

def plot_average_time_series(time_series, title, label='', loc='lower right', 
        fignum=1, ax=111, tol=.1, color='B'):

    r_avg, n_equib, val_equib, dval_equib = \
            pp.running_average(time_series, tol=tol)
    
    ts = [n_equib, len(time_series)]
    ys = [val_equib]*2
    yerr=[dval_equib, 0]
    fig = plt.figure(fignum)
    fig.add_subplot(ax)
    plt.errorbar(ts, ys, yerr=yerr, marker='o', fmt='-', ms=5, color=color)
    plt.title(title)
    plt.legend(loc=loc)
    plt.tight_layout() 

# make Fourier transform of time series data
# ------------------------------------------
def make_ft(time_series, dt):
    Nsteps = len(time_series)
    times = [n*dt for n in range(Nsteps)]
    
    if Nsteps%2 == 1:
        time_sereis = np.delete(time_series,-1)
        Nsteps = Nsteps - 1
    
    time_series = time_series - np.mean(time_series)
    amps =  (2.0/Nsteps)*np.abs(spf.fft(time_series)[0:Nsteps/2])
    freqs = np.linspace(0.0,1.0/(2.0*dt),Nsteps/2)
    return freqs, amps

# plot Fourier transform
# ----------------------
def plot_ft(freqs, amps, dt, title, fignum=1, ax=111, color='B'):

    #Nyquist criterion
    high_freq = 1.0/(2.0*dt)
    low_freq = 1.0/(dt*len(amps))
    
    amp_ave = np.mean(amps)
    fig = plt.figure(fignum)
    fig.add_subplot(ax)
    
    if amp_ave>1e-14:
        plt.semilogy(freqs, amps, color=color)
    else:
        plt.plot(freqs, amps, color=color)
    
    plt.title(title)
    
    plt.xlabel('Frequency')
    plt.xlim([low_freq, high_freq])
   
    plt.ylabel('Intensity')
    plt.ylim(amp_ave/100., 10.*amps.max())
    
    #plt.fill_between(freqs, 0, amps)
    plt.tight_layout()


# plot space time grids
# ---------------------
def plot_spacetime_grid(grid_data, title, cmap=plt.cm.jet, norm=None, fignum=1,
        ax=111, nx_ticks=4, ny_ticks=15):

    vmin, vmax = 0.0, 1.0
    if np.max(grid_data) > 1.0:
        vmax = np.max(grid_data)
    if np.min(grid_data) < 0.0:
        vmin = np.min(grid_data)
 
    x_min = 0
    x_max = len(grid_data[0])
    y_min = 0
    y_max = len(grid_data)

    xtick_lbls = range(x_min, x_max)[::int(x_max/nx_ticks)]
    xtick_locs = xtick_lbls 

    ytick_lbls = range(y_min, y_max)[::int(y_max/ny_ticks)]
    ytick_locs = ytick_lbls
    fig = plt.figure(fignum)
    fig.add_subplot(ax)
    plt.imshow(grid_data,
                    vmin = vmin,
                    vmax = vmax,
                    cmap = cmap,
                    norm = norm,
                    interpolation = 'none',
                    aspect = 'auto',
                    rasterized = True)
   
    plt.xticks(xtick_locs, xtick_lbls)
    plt.yticks(ytick_locs, ytick_lbls)

    plt.title(title)
    plt.colorbar()
    plt.tight_layout() 

def make_edge_strength_hist(mat, bins=30):
    edges = [m for mj in mat for m in mj]
    hist = sts.binned_statistic(edges, edges, statistic='count', bins=bins)[0]
    return hist

def plot_histogram_surface(mi_nets, title, cmap=plt.cm.jet, fignum=1, ax=111,
        bins=30, smoothing=2):

    tmax = len(mi_nets) 
    Z = np.asarray([make_edge_strength_hist(mi_nets[t], bins=bins) for t in range(tmax)])
    fig = plt.figure(fignum)
    axe=fig.add_subplot(ax, projection='3d')

    X = np.linspace(0, 1, bins)
    Y = np.arange(0, tmax, 1)

    X, Y = np.meshgrid(X, Y)

    Z = zoom(Z, smoothing)
    X = zoom(X, smoothing)
    Y = zoom(Y, smoothing)

    axe.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.jet, 
                     linewidth=0, antialiased=False)

    zmax = 80
    axe.set_zlim3d(0,zmax)
    plt.xlabel('Mutual information')
    plt.ylabel('time')
    #plt.zlabel('occurrences')



# Specific plotting utilities
# ===========================

# plot probability, discretized probability, and single-site entropy
# ------------------------------------------------------------------
def board_plots(rjt, sjt, fignum=1, axs=(131, 132, 133)):
    n_res      = pp.local_exp_vals(rjt, ss.ops['1'])
    sj_res     = sjt 
    disc_n_res = [ [0 if p < 0.5 else 1 for p in p_vec] for p_vec in n_res]
    
    disc_cmap = mpl.colors.ListedColormap([plt.cm.jet(0.), plt.cm.jet(1.)])
    bounds = [0,0.5,1]
    disc_norm = mpl.colors.BoundaryNorm(bounds, disc_cmap.N)
    
    titles = [ r'$ \langle n_j \rangle $', \
               r'$ \lfloor \langle n_j \rangle + 0.5 \rfloor $', \
               r'$ s_j $' ]
    
    results = [n_res, disc_n_res, sj_res]
    mycmaps = [plt.cm.jet, disc_cmap, plt.cm.jet]
    norms = [None, disc_norm, None]
    fig = plt.figure(fignum)
    for title, res, cmap, norm, ax in zip(titles, results, mycmaps, norms, axs):
        kwargs = {'cmap': cmap, 'norm': norm, 'fignum' : fignum, 'ax' : ax} 
        fig.add_subplot(ax)
        plot_spacetime_grid(res, title, **kwargs)
        plt.title(title)
        plt.xlabel('Site number')
     #   plt.xticks(xtick_locs, xtick_lbls)
        plt.ylabel('Time')
    plt.tight_layout() 


# plot entropy of all bi-partite cuts
# -----------------------------------
def cut_entropy_plots(results, L, suptitle, fignum=1, axs=(121, 122)):

#    xtick_locs = range(0,L-1,2)
#    xtick_lbls = xtick_locs
     
    EC_res         = results['ec']
    center_cut_res = (np.array(EC_res).transpose()[int(L/2)]).transpose()

    fig =  plt.figure(fignum)

    fig.add_subplot(axs[0]) 
    plot_spacetime_grid(EC_res, 'Entropy of bi-partite cuts', 
                         cmap=plt.cm.jet, norm=None, fignum=fignum, ax=axs[0])
    plt.xlabel('Cut number')
#    plt.xticks(xtick_locs, xtick_lbls)
    plt.ylabel('Time')
    
    fig.add_subplot(axs[1])
    plot_time_series(center_cut_res, 'Entropy of cut'+str(int(L/2)), 
                     fignum=fignum, ax = axs[1])
    plt.xlabel('time')
    plt.ylabel('Entropy')
    
    plt.suptitle(suptitle) 
    plt.tight_layout() 

# plot network measures
# ---------------------
def nm_spacetime_plots(st_data, tasks = ['EV', 'CC', 'Y'], fignum=1):
    ax = 131 
    for i, task in enumerate(tasks):
        dat = st_data[task]
        title = task
        plot_spacetime_grid(dat, title, fignum=fignum, ax=ax)
        plt.ylabel('Time')
        plt.xlabel('Site number')
        ax += 1

def nm_time_series_plots(avg_data, title, tasks = ['ND', 'CC','Y'], fignum=1, tol=0.1):
    color_list=['B','G','R']
    for i, task in enumerate(tasks):
        dat = avg_data[task]
        label = task
        plot_average_time_series(dat, title, label=label, fignum=fignum,
                color=color_list[i], tol=tol)
        plot_time_series(dat, title, label=label, fignum=fignum, 
                loc='upper right', color=color_list[i])

def nm_ft_plots(avg_data, dt, tasks = ['ND', 'CC','Y'], fignum=1):
    ax = 311 
    color_list=['B','G','R']
    for i, task in enumerate(tasks):
        title = task
        dat = avg_data[task]
        freqs_dat, amps_dat = make_ft(dat, dt)
        plot_ft(freqs_dat, amps_dat, 1, title, fignum=fignum, ax=ax, color=color_list[i])
        ax += 1

# call for time series plots
# --------------------------
def plot_main(params, 
        st_tasks =['EV', 'CC', 'Y' ], avg_tasks=['ND', 'CC', 'Y'],
        net_types=['mi'], name=None):

    R    = params[ 'R'   ]
    L    = params[ 'L'   ]
    tmax = params[ 'tmax']
    output_name = params['output_name']

    if name is None:
        name = io.sim_name(params)
    else:
        name = name

    print('Importing results...')
    results = io.read_results(params)
    
    mi_nets = results['mi']
    ipr = results['ipr'] 
    sjt = pp.make_local_vn_entropy(results)
    
    rjt_mat = pp.make_rjt_mat(results) 
    x_grid  = pp.local_exp_vals(rjt_mat, ss.ops['X'])
    y_grid  = pp.local_exp_vals(rjt_mat, ss.ops['Y'])
    z_grid  = pp.local_exp_vals(rjt_mat, ss.ops['Z'])

    nm_spacetime_grids = pp.measure_networks(mi_nets, typ='st')
    nm_time_series = pp.measure_networks(mi_nets, typ='avg')

    plot_spacetime_grid(x_grid, 'X projection', fignum=1, ax=131)
    plot_spacetime_grid(y_grid, 'Y projection', fignum=1, ax=132)
    plot_spacetime_grid(z_grid, 'Z projection', fignum=1, ax=133)

    nm_time_series_plots(nm_time_series, 'Mutual information network measures', 
                        tasks=avg_tasks, fignum=2, tol=0.001)

    nm_ft_plots(nm_time_series, 1, fignum=3)

    nm_spacetime_plots(nm_spacetime_grids,   tasks=st_tasks , fignum=4)

    cut_entropy_plots(results, L, 'R '+str(R), fignum=5) 

    board_plots(rjt_mat, sjt, fignum=6)

    plot_time_series(ipr, 'Inverse participation ratio', fignum=7, ax=211)
    
    freqs_ipr, amps_ipr = make_ft(ipr, 1)
    plot_ft(freqs_ipr, amps_ipr, 1, '', fignum=7, ax=212)

    plot_histogram_surface(mi_nets, 'Mutual information distribution', fignum=8,
            smoothing=2, bins=20)    

    io.multipage(io.file_name(output_name, 'plots', name, '.pdf'))    



if __name__ == '__main__':
    from math import sin, pi
    T = 0.03
    dt = 0.001
    ts = [n*dt for n in range(100)]
    ys = [sin(2*pi*t/T) for t in ts]

    freqs, amps = make_ft(ys, dt) 
    
    max_index = np.argmax(amps)
    max_amp   = amps[max_index]
    max_freq  = freqs[max_index]

    print(1.0/max_freq, T)

    plot_time_series(ys, '', fignum=1, ax=211)


    plot_ft(freqs, amps, dt, '', fignum = 1, ax=212)

    plt.show()

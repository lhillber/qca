#!/usr/bin/python

import numpy             as np
import simulation.states            as ss
import matplotlib        as mpl
import scipy.stats       as sts
import scipy.fftpack     as spf
import matplotlib.pyplot as plt
import matplotlib.transforms as trans
import simulation.fio               as io
import simulation.measures

from math import pi
from collections import OrderedDict
import matplotlib.gridspec as gridspec


# default plot font
# -----------------
font = {'family':'serif','size':10}
mpl.rc('font',**font)


# plot spacetime grid on an axis
# ------------------------------
def plot_grid(data, ax, nc=1,
        title='', ylabel='Iteration', xlabel='Site',
        xtick_labels=True, ytick_labels=True,
        nx_ticks=4, ny_ticks=10, wspace=-0.25,
        cbar=True, cmap=plt.cm.jet, plot_kwargs={}, span=None):

    if span is None:
        span = [0, len(data)]

    L = len(data[0])
    T = len(data)
    fig = plt.gcf()
    im = ax.imshow( data[span[0]:span[1],::],
                origin = 'lower',
                cmap = cmap,
                interpolation = 'none',
                aspect = '1',
                rasterized = True,
                extent=[0, L-1, span[0], span[1]],
                **plot_kwargs)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_xlim(0, L-1)
    ax.set_ylim(span[0], span[1])
    ax.xaxis.set_ticks([0, int(L/4), int(2*L/4), int(3*L/4)])
    ax.yaxis.set_ticks(range(span[0], span[1]))
    ax.locator_params(axis='x', nbins=nx_ticks)
    ax.locator_params(axis='y', nbins=ny_ticks)
    ax.grid(True)



    fig.subplots_adjust(top=0.9, wspace=wspace)
    if cbar is True:
        im_ext = im.get_extent()
        box = ax.get_position()
        cax = plt.axes([box.x1+wspace/nc, box.y0, 0.01, box.height])
        cb = plt.colorbar(im, cax = cax)
        cb.ax.tick_params(labelsize=9)
    if not ytick_labels:
        plt.setp([ax.get_yticklabels()], visible=False)
    if not xtick_labels:
        plt.setp([ax.get_xticklabels()], visible=False)

# plot space or time averages of grids
# -----------------------------------
def plot_grid_avgs(data, ax, avg='space',
        title='', ylabel=None, xlabel=None,
        xtick_labels=True, ytick_labels=True,
        nx_ticks=None, ny_ticks=None,
        plot_kwargs={}, span=None, rotate=False):

    if avg is 'space':
        if ylabel is None:
            ylabel = 'Average'
        else: ylabel = ylabel
        if xlabel is None:
            xlabel = 'Iteration'
        else: xlabel
        axis = 1

    if avg is 'time':
        if ylabel is None:
            ylabel = 'Average'
        else: ylabel = ylabel
        if xlabel is None:
            xlabel = 'Site'
        else: xlabel
        axis = 0

    data = np.mean(data, axis=axis)

    plot_time_series(data, ax,
        title=title, ylabel=ylabel, xlabel=xlabel,
        xtick_labels=xtick_labels, ytick_labels=ytick_labels,
        nx_ticks=nx_ticks, ny_ticks=ny_ticks,
        plot_kwargs=plot_kwargs, span=span, rotate=rotate)

def plot_grid_with_avgs(data, fignum):
    fig = plt.figure(fignum)
    L = len(data[0])
    T = len(data)
    gs = gridspec.GridSpec(100, 100, bottom=0.15, left=0.15, right=0.95)

    dx=1
    dy=2
    w1=12
    w2=16
    h=16
    axC = fig.add_subplot(gs[h+dy:, w1+dx:w1+dx+w2])
    axT = fig.add_subplot(gs[0   :h,w1+dx:w1+dx+w2], sharex=axC)
    axR = fig.add_subplot(gs[h+dy:, 0:w1], sharey=axC)

    axC.spines['right'].set_position(('data',L-1))
    axC.spines['left'].set_position(('data',0))
    axC.spines['top'].set_position(('data', T))
    axC.spines['bottom'].set_position(('data',0))
    axT.spines['right'].set_position(('data',L-1))
    axT.spines['left'].set_position(('data',0))

    axC.spines['right'].set_smart_bounds(True)
    axC.spines['left'].set_smart_bounds(True)
    axC.spines['top'].set_smart_bounds(True)
    axC.spines['bottom'].set_smart_bounds(True)
    axT.spines['right'].set_smart_bounds(True)
    axT.spines['left'].set_smart_bounds(True)
    axT.spines['top'].set_smart_bounds(True)
    axT.spines['bottom'].set_smart_bounds(True)

    extent = axC.get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted())
    data = (1.0-data)/2

    plot_grid(data, axC, cbar=False,
            ytick_labels=False, xlabel='Site', ylabel='')

    plot_grid_avgs(data, axR, avg='space', rotate=True,
    ylabel='Iteration', xlabel='', title = 'spatial      ', nx_ticks=4)

    plot_grid_avgs(data, axT, avg='time', xtick_labels=False,
            title='temporal', xlabel='', ylabel='', ny_ticks=4)


# plot multiple spacetime grids as subplots
# -----------------------------------------
def plot_grids(grid_data, fignum=1, span=[0, 60], wspace=-0.25,
        titles=None, xlabels=None, ylabel='Iteration', suptitle=''):
    nc = len(grid_data)
    if titles is None:
        titles = ['']*nc
    if xlabels is None:
        xlabels = ['']*nc
    gs = gridspec.GridSpec(1, nc)
    fig = plt.figure(fignum)
    for c, (grid, title, xlabel) in \
            enumerate(zip(grid_data, titles, xlabels)):
        ax = fig.add_subplot(gs[c])
        ytick_labels=False
        ylabel = ''
        if c is 0:
            ytick_labels=True
            ylabel = ylabel
        plot_grid(grid, ax,
                nc=nc,
                ylabel=ylabel,
                xlabel=xlabel,
                title=title,
                ytick_labels=ytick_labels,
                wspace=wspace,
                span=span)
    fig.suptitle(suptitle)


# plot time series on an axis
# ---------------------------
def plot_time_series(time_series, ax,
        title='', ylabel='Measure', xlabel='Iteration',
        xtick_labels=True, ytick_labels=True, nx_ticks=None, ny_ticks=None,
        loc=None, plot_kwargs=None, span=None, rotate=False):

    if span is None:
        span = [0, len(time_series)]
    if plot_kwargs is None:
        plot_kwargs = {'label':'', 'color':'B',
                'linewidth':1, 'linestyle':'-',
                'marker':'s', 'markersize':1.5, 'markeredgecolor':'B'}

    indep_var = range(span[0], span[1])
    dep_var = time_series[span[0]:span[1]]

    if rotate:
        ax.plot(dep_var, indep_var,  **plot_kwargs)

    else:
        ax.plot(indep_var, dep_var,  **plot_kwargs)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    if nx_ticks is not None:
        ax.locator_params(axis='x', nbins=nx_ticks)
    if ny_ticks is not None:
        ax.locator_params(axis='y', nbins=ny_ticks)

    if loc is not None:
        ax.legend(loc=loc)
    if not ytick_labels:
        plt.setp([ax.get_yticklabels()], visible=False)
    if not xtick_labels:
        plt.setp([ax.get_xticklabels()], visible=False)

# make Fourier transform of time series data
# ------------------------------------------
def make_ft(time_series, dt=1):
    time_series = np.nan_to_num(time_series)
    Nsteps = len(time_series)
    times = [n*dt for n in range(Nsteps)]

    if Nsteps%2 == 1:
        time_sereis = np.delete(time_series,-1)
        Nsteps = Nsteps - 1

    # dt = 2*pi*dt
    time_series = time_series - np.mean(time_series)
    amps =  (2.0/Nsteps)*np.abs(spf.fft(time_series)[0:Nsteps/2])
    freqs = np.linspace(0.0,1.0/(2.0*dt), Nsteps/2)
    return freqs, amps

# plot Fourier transform on an axis
# ---------------------------------
def plot_ft(freqs, amps, ax, dt=1,
        title='', ylabel='Intensity', xlabel='Frequency',
        xtick_labels=True, ytick_labels=True, loc=None, plot_kwargs=None):

    if plot_kwargs is None:
        plot_kwargs = {'label':'', 'color':'B',
                'linewidth':1, 'linestyle':'-',
                'marker':'', 'markersize':1, 'markeredgecolor':'B'}

    #Nyquist criterion
    #dt = 2*pi*dt
    high_freq = 1.0/(2.0*dt)
    low_freq = 1.0/(dt*len(amps))

    amp_ave = np.mean(amps)
    if amp_ave>1e-14:
        ax.semilogy(freqs, amps, )
    else:
        ax.plot(freqs, amps, **plot_kwargs)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_xlim([low_freq, high_freq])
    ax.set_ylim(amp_ave/100., 10.*amps.max())

    if loc is not None:
        ax.legend(loc=loc)
    if not ytick_labels:
        plt.setp([ax.get_yticklabels()], visible=False)
    if not xtick_labels:
        plt.setp([ax.get_xticklabels()], visible=False)


# plot time series of many measures
# ---------------------------------
def plot_measures(meas_dict, fignum=1):
    nr = len(list(meas_dict.keys()))
    gs = gridspec.GridSpec(nr, 1)
    fig = plt.figure(fignum)

    for r, (name, vals) in enumerate(meas_dict.items()):
        ax = fig.add_subplot(gs[r])
        xtick_labels=False
        xlabel = ''

        if r is nr-1:
            xtick_labels=True
            xlabel = 'Iteration'

        plot_time_series(vals, ax,
                xlabel=xlabel,
                ylabel=name,
                xtick_labels=xtick_labels)

        ax.grid( 'on' )


# plot Fourier transform of many measures
# ---------------------------------------
def plot_measure_FTs(meas_dict, fignum=1):
    nr = len(list(meas_dict.keys()))
    gs = gridspec.GridSpec(nr, 1)
    fig = plt.figure(fignum)

    for r, (name, vals) in enumerate(meas_dict.items()):
        ax = fig.add_subplot(gs[r])
        xtick_labels=False
        xlabel = ''

        if r is nr-1:
            xtick_labels=True
            xlabel = 'Frequency'

        freqs, amps = make_ft(vals)
        plot_ft(freqs, amps, ax,
                xlabel=xlabel,
                ylabel=r'$\mathcal{F}$('+name+')',
                xtick_labels=xtick_labels)

        ax.grid( 'on' )
    plt.tight_layout()


# make histogram of mutual information adjacency matrix
# -----------------------------------------------------
def make_edge_strength_hist(mjk, bins=30, rng=(0,1)):
    edges = [m for mj in mjk for m in mj]
    hist, binedges, binnumbers = sts.binned_statistic(edges, edges,
            statistic='count', bins=bins, range=rng)
    return hist

# plot histogram of mutual information through time as a contour plot
# -------------------------------------------------------------------
def plot_edge_strength_contour(mtjk, bins=30, rng=(0,1), emax=40,
        fignum=1, cmap=plt.cm.jet,
        title='Mutual information edge strength'):

    fig = plt.figure(fignum)
    ax = fig.add_subplot(111)

    T = len(mtjk)
    X = np.linspace(rng[0], rng[1], bins)
    Y = range(0, T, 1)
    Z = np.asarray([make_edge_strength_hist(mtjk[t], bins=bins, rng=rng) for t in range(T)])
    X, Y = np.meshgrid(X, Y)

    levels = np.linspace(0, emax, 20)
    cs = ax.contourf(X, Y, Z,
            levels=levels,
            origin='lower',
            cmap=cmap,
            rasterize=True)

    fig.colorbar(cs)
    ax.set_xlabel('Edge strength')
    ax.set_ylabel('Iteration')
    ax.set_title('Mutual information edge strength')

# call plotting sequence
# ----------------------
def plot(params, fname, force_rewrite=False, j=0):
    print('Plotting measures...')

    # get correlators at constant row j
    x_g2grid, y_g2grid, z_g2grid =\
            map(lambda mats: measures.get_row_vecs(mats, j=j), io.read_hdf5(fname, ['gxx', 'gyy', 'gzz']))

    # get spin projections along x, y, and z
    x_grid, y_grid, z_grid =\
            map(measures.get_diag_vecs, io.read_hdf5(fname, ['xx', 'yy', 'zz']))

    # get mi measure results and place in ordered dict for plotting
    nm_keys = ['ND', 'CC', 'Y']
    meas_list = io.read_hdf5(fname, nm_keys)
    meas_dict = OrderedDict( (key,data)
            for key, data in zip(nm_keys, meas_list))

    # get local and bond entropies
    stj = io.read_hdf5(fname, 's')

    stc = io.read_hdf5(fname, 'sc')

    # get mutual information adjacency matrices
    mtjk = io.read_hdf5(fname, 'm')

    # plot spin projections
    plot_grids([x_grid, y_grid, z_grid],
            titles=['$X$', '$Y$', '$Z$'],
            suptitle='Spin projections',
            wspace=-0.20,
            fignum=0)

    # plot two-point correlator w.r.t site j
    plot_grids([x_g2grid, y_g2grid, z_g2grid],
            titles=['$X$', '$Y$', '$Z$'],
            suptitle=r'$g_2(j=$'+str(j)+r'$,k;t)$',
            wspace=-0.20,
            fignum=3)

    # plot local and bond entropies
    plot_grids([stj, stc],
            titles=[r'$S(i,t)$', r'$S_c(i,t)$'],
            xlabels=['site', 'cut'],
            suptitle='von Neumann entropies',
            wspace=-0.29,
            fignum=4)

    # plot probabilities of spin down and space/time averages
    plot_grid_with_avgs(z_grid, 5)

    # plot mi measures and their FT's
    plot_measures(meas_dict, fignum=6)
    plot_measure_FTs(meas_dict, fignum=7)

    # plot distribution of mutual information over time
    plot_edge_strength_contour(mtjk,
            bins=60, rng=(0,.1), emax=30, fignum=8)

    # iterate version numbers for random throw IC's
    iterate = False
    if params['IC'][0] == 'r':
        iterate = True

    # create the full path to where data will be stored
    out_fname = io.make_file_name(params, sub_dir='plots', ext='.pdf', iterate = iterate)

    io.base_name(params['output_dir'], 'plots')
    path_list = fname.split('/')
    sub_dir_ind = path_list.index('data')
    path_list[sub_dir_ind] = 'plots'
    path_ext_list = '/'.join(path_list).split('.')
    path_ext_list[1] = '.pdf'
    out_fname = ''.join(path_ext_list)

    # save all figures to one pdf
    io.multipage(out_fname)
    print('plots saved to: ', out_fname)

def fft_check():
    from math import sin
    T = 0.03
    f = 300
    dt = 0.001
    ts = np.array([n*dt for n in range(1000)])
    ys = np.array([sin(2*pi*t/T) + 2*sin(2*pi*f*t) for t in ts])

    freqs, amps = make_ft(ys, dt=dt)

    max_index = np.argmax(amps)
    max_amp   = amps[max_index]
    max_freq  = freqs[max_index]

    fig = plt.figure(1)
    gs = gridspec.GridSpec(2,1)

    plot_time_series(ys, plt.subplot(gs[0]))
    plot_ft(freqs, amps, plt.subplot(gs[1]), dt=dt)
    plt.subplot(gs[1]).scatter(max_freq, max_amp)
    print('freq found: ',max_freq,' given freq: ', f)
    plt.show()

if __name__ == '__main__':
    import time_evolve
    params =  {
                    'output_dir' : 'testing/state_saving',

                    'L'    : 12,
                    'T'    : 100,
                    'mode' : 'block',
                    'R'    : 150,
                    'V'    : ['H','T'],
                    'IC'   : 'l0'
                                    }

    import numpy
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable


    #fname = time_evolve.run_sim(params, force_rewrite=False)
    #measures.measure(params, fname, force_rewrite=False)
    #plot(params, fname)

    #fft_check()



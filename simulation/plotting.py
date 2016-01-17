#!/usr/bin/python3

import numpy               as np
import simulation.states   as ss
import matplotlib          as mpl
import scipy.stats         as sts
import scipy.fftpack       as spf
import simulation.fio      as io
import simulation.measures as measures

from math import pi
from collections import OrderedDict

import matplotlib.gridspec as gridspec
import h5py
import matplotlib.transforms as trans

# default plot font
# -----------------
font = {'family':'serif','size':10}
mpl.rc('font',**font)
import matplotlib.pyplot as plt


# Labeling functions
# ------------------
def inner_title(ax, title, l=0.1, u=0.95): return ax.text(l, u, str(title),
        horizontalalignment='left', transform=ax.transAxes, fontsize=12)


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
    #ax.grid(True)


    ##wspace kwg
    fig.subplots_adjust(top=0.9, wspace=0.5)
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

def plot_grid_with_avgs(data, fignum=1, span=None, suptitle=''):
    fig = plt.figure(fignum)
    L = len(data[0])
    T = len(data)
    gs = gridspec.GridSpec(100, 100, bottom=0.15, left=0.15, right=0.95)
    if span is None:
        span = [0, min([60, T])]

    dx=3
    dy=-9
    w1=16
    w2=12
    h=16 

    axC = fig.add_subplot(gs[h+dy:, 0:w1])
    axT = fig.add_subplot(gs[0:h, 0:w1], sharex=axC)
    axR = fig.add_subplot(gs[h+dy:, w1+dx:w1+dx+w2], sharey=axC)

    axC.spines['top'].set_position(('data', span[1]))
    axC.spines['bottom'].set_position('zero')
    axC.spines['left'].set_smart_bounds(True)
    axC.spines['right'].set_smart_bounds(True)


    axR.spines['top'].set_position(('data', span[1]))
    axR.spines['bottom'].set_position('zero')
    axR.spines['left'].set_smart_bounds(True)
    axR.spines['right'].set_smart_bounds(True)


    data = (1.0-data)/2

    plot_grid(data, axC, cbar=False,
            nx_ticks=5, xlabel='Site', ylabel='Iteration', span=span)

    plot_grid_avgs(data, axR, avg='space', rotate=True,
        ylabel='', xlabel='P', ytick_labels=False,
        nx_ticks=3, span=span)
    inner_title(axR, 'Spatial', u=.87)

    plot_grid_avgs(data, axT, avg='time', xtick_labels=False,
            title='temporal', xlabel='', ylabel='P', ny_ticks=5)
    inner_title(axC, 'Average Probability of measuring 1', l=-.85, u=1.16)

# plot multiple spacetime grids as subplots
# -----------------------------------------
def plot_grids(grid_data, fignum=1, span=None, wspace=-0.25,
        titles=None, xlabels=None, ylabels=None, suptitle=''):
    nc = len(grid_data)
    if titles is None:
        titles = ['']*nc
    if xlabels is None:
        xlabels = ['']*nc
    if ylabels is None:
        ylabels = ['Iteration'] + ['']*(nc - 1)
    if span is None:
        span = [0, min([60, len(grid_data[0])])]
    gs = gridspec.GridSpec(1, nc)
    fig = plt.figure(fignum)
    for c, (grid, title, xlabel, ylabel) in \
            enumerate(zip(grid_data, titles, xlabels, ylabels)):
        ax = fig.add_subplot(gs[c])
        ytick_labels=False
        if c == 0:
            ytick_labels=True
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


# plot Fourier transform on an axis
# ---------------------------------
def plot_ft(freqs, amps, ax, dt=1,
        title='', ylabel='Intensity', xlabel='Frequency',
        xtick_labels=True, ytick_labels=True, loc=None, plot_kwargs=None,
        nx_ticks=None, ny_ticks=None):

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
        ax.semilogy(freqs, amps)
    else:
        ax.plot(freqs, amps, **plot_kwargs)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_xlim([low_freq, high_freq])
    ax.set_ylim(amp_ave/5, 1.5*amps.max())

    if loc is not None:
        ax.legend(loc=loc)
    if not ytick_labels:
        plt.setp([ax.get_yticklabels()], visible=False)
    if not xtick_labels:
        plt.setp([ax.get_xticklabels()], visible=False)

    if nx_ticks is not None:
        ax.locator_params(axis='x', nbins=nx_ticks)
    if ny_ticks is not None:
        ax.locator_params(axis='y', nbins=ny_ticks)

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
def plot_measure_fts(Fmeas_dict, freqs, fignum=1):
    nr = len(list(Fmeas_dict.keys()))
    gs = gridspec.GridSpec(nr, 1)
    fig = plt.figure(fignum)

    for r, (name, vals) in enumerate(Fmeas_dict.items()):
        ax = fig.add_subplot(gs[r])
        xtick_labels=False
        xlabel = ''

        if r is nr-1:
            xtick_labels=True
            xlabel = 'Frequency'

        amps = Fmeas_dict[name]
        plot_ft(freqs, amps, ax,
                xlabel=xlabel,
                ylabel=r'$|\mathcal{F}$('+name+r'$)|^2$',
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



def plot_grid_time_avg_stats(grids_stats_dict, fignum=1, 
                titles=['x', 'y', 'z'], coords = ['xx', 'yy', 'zz']):
    typ = 'time'
    suptitle = 'Temporal averages'
    fig = plt.figure(fignum)
    fig.subplots_adjust(top=0.91, wspace=0.5, hspace=0.5)
    fignum = fignum+1
    for i, (coord, title) in enumerate(zip(coords, titles)):
        ylabels =['']*3
        if i == 0:
            ylabels = ['avg', 'std',
                    '$|\mathcal{F}(\mathrm{avg})|^2$']

        for j, (stat, ylabel) in enumerate(zip(
            ['avg', 'std','amps'], ylabels)):
            ax = fig.add_subplot(3,3,i+1+(j*3))

            if j == 1:
                title = ''
            if j == 2:
                plot_ft(grids_stats_dict[coord][typ]['freqs'][::],
                        grids_stats_dict[coord][typ]['amps'][::],
                        ax, ylabel = ylabel, nx_ticks=4)
            elif j != 2:
                xlabel = 'site'
                plot_time_series(grids_stats_dict[coord][typ][stat][::], ax,
                        xlabel=xlabel, ylabel=ylabel, ny_ticks=4,
                        title=title)
            ax.grid('on')
    plt.suptitle('Temporal averages')


def plot_grid_space_avg_stats(grids_stats_dict, fignum=1,
                titles=['x', 'y', 'z'], coords = ['xx', 'yy', 'zz']):
    typ = 'space'
    for i, (coord, title) in enumerate(zip(coords, titles)):
        xlabel = 'Iteration'
        ylabels =['']*3
        ylabels = ['avg', 'std',
                '$|\mathcal{F}(\mathrm{avg})|^2$']


        for j, (stat, ylabel) in enumerate(zip(['avg','std','amps'], ylabels)):
            fig = plt.figure(fignum)

            fig = plt.figure(fignum)
            ax = fig.add_subplot(3,1,j+1)
            fig.subplots_adjust(top=0.91, wspace=0.5, hspace=0.5)

            if j == 1:
                title = ''
            if j == 2:
                plot_ft(grids_stats_dict[coord][typ]['freqs'][::],
                        grids_stats_dict[coord][typ]['amps'][::],
                        ax, ylabel = ylabel, nx_ticks=6)
            elif j != 2:
                plot_time_series(grids_stats_dict[coord][typ][stat][::], ax,
                        xlabel=xlabel, ylabel=ylabel, ny_ticks=4,
                        title=title)
            ax.grid('on')
        fignum = fignum+1
        plt.suptitle('Spatial averages')
        #plt.tight_layout()

# call plotting sequence
# ----------------------
def plot(params, corrj=None):
    print('Plotting results...')
    results = h5py.File(params['fname'], 'r')
    # get spin projections along x, y, and z
    x_grid, y_grid, z_grid = [ measures.get_diag_vecs(results[ab][::])
                for ab in ('xx', 'yy', 'zz') ]

    proj_grids_stats = results['stats']

    # get g2 correlators at constant row j
    if corrj is None:
        corrj = results['gstats']['corrj'][0]
    x_g2grid, y_g2grid, z_g2grid = [measures.get_row_vecs(
        results[ab][::], j=corrj) for ab in ['gxx', 'gyy', 'gzz']]

    g2grids_stats = results['gstats']

    # get mi measure results and place in ordered dict for plotting
    meas_keys = ['ND', 'CC', 'Y', 'IPR']
    meas_list = [results[meas_key][::] for meas_key in meas_keys]
    freqs = results['freqs'][::]
    Fmeas_list = [results['F'+meas_key][::] for meas_key in meas_keys]
    meas_dict = OrderedDict( (key, data)
            for key, data in zip(meas_keys, meas_list))
    Fmeas_dict = OrderedDict( (key, Fdata)
            for key, Fdata in zip(meas_keys, Fmeas_list))

    # get local and entropies
    stj = results['s'][::]

    # get mutual information adjacency matrices
    mtjk = results['m'][::]

    # plot spin projections
    proj_titles = [r'$\langle \sigma^x_j \rangle$', 
                   r'$\langle \sigma^y_j \rangle$',
                   r'$\langle \sigma^z_j \rangle$']
    plot_grids([x_grid, y_grid, z_grid], fignum=0,
            titles=proj_titles,
            suptitle='Spin Projections',
            xlabels=['site', 'site', 'site'],
            wspace=.05)


    plot_grid_time_avg_stats(proj_grids_stats, fignum=1,
            titles=proj_titles)

    # this makes three figures
    plot_grid_space_avg_stats(proj_grids_stats, fignum=2, titles=proj_titles)

    # plot two-point correlator w.r.t site corrj
    g2_titles = ['$g_2(\sigma^x_{%i},\sigma^x_k;t)$' % corrj,
                 '$g_2(\sigma^y_{%i},\sigma^y_k;t)$' % corrj,
                 '$g_2(\sigma^z_{%i},\sigma^z_k;t)$' % corrj]

    plot_grids([x_g2grid, y_g2grid, z_g2grid], fignum=5,
            titles=g2_titles,
            suptitle='Two Point Correlator',
            xlabels=['site', 'site', 'site'],
            wspace=0.05)

    plot_grid_time_avg_stats(g2grids_stats, fignum=6, titles=g2_titles)

    plot_grid_space_avg_stats(g2grids_stats, fignum=7, titles=g2_titles)

    # plot local and bond entropies
    entropies = [stj]
    if 'sc' in results:
        stc = results['sc'][::]
        entropies.append(stc)

    if len(entropies) == 2:
        wspace=0.088
    elif len(entropies) == 1:
        wspace=-0.23

    plot_grids(entropies,
            titles=[r'$S(j,t)$', r'$S_c(j,t)$'],
            xlabels=['site', 'cut'],
            suptitle='von Neumann entropies',
            wspace=wspace,
            fignum=10)

    # plot probabilities of spin down and space/time averages
    plot_grid_with_avgs(z_grid, fignum=11, suptitle='average probability of measuring 1')

    # plot mi measures and their FT's
    plot_measures(meas_dict, fignum=12)

    # plot measure Fourier transforms
    plot_measure_fts(Fmeas_dict, freqs, fignum=13)

    # plot distribution of mutual information over time
    plot_edge_strength_contour(mtjk,
            bins=60, rng=(0,.1), emax=150, fignum=14)



    # create the full path to where plots will be saved
    fname = params['fname']
    io.base_name(params['output_dir'], 'plots')
    path_list = fname.split('/')
    sub_dir_ind = path_list.index('data')
    path_list[sub_dir_ind] = 'plots'
    path_ext_list = '/'.join(path_list).split('.')
    path_ext_list[-1] = '.pdf'
    out_fname = ''.join(path_ext_list)

    # save all figures to one pdf
    io.multipage(out_fname)
    results.close()
    plt.close('all')
    return out_fname

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

    #fft_check()



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
import matplotlib.gridspec as gridspec
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


# fit a gaussian mixture of n compnents to a grid row (a list of data)
# row is 'ydata' 'js' is x data
# useful: http://www.astroml.org/book_figures/chapter4/fig_GMM_1D.htmlI
def gmix(row, js, n=2):
    distrib = []
    mx = max(row)
    mn = min(row)

    # build a distribution for the fit
    for j, val in enumerate(row):
        distrib += [j]*int(1000*val)

    distrib = np.array(distrib)
    model = GMM(n).fit(distrib[:, np.newaxis])
    logprob, responsibilities = model.score_samples(js[:,np.newaxis])
    pdf = np.exp(logprob)
    pdf_individual = responsibilities * pdf[:, np.newaxis]
    means = model.means_
    sds = model.covars_

    # reshaping and duplicating data for n=1 component
    if n == 1:
        means = np.array([means[0]]*2)
        sds = np.array([sds[0]]*2)
        pdf_individual = np.asarray([pdf_individual,
            pdf_individual]).reshape(len(pdf_individual),2)

    means = [means[0][0], means[1][0]]
    sds = [sds[0][0], sds[1][0]]
    return pdf, pdf_individual, means, sds, distrib


# two component fit means and sds and previous time step means used to determine
# a localized trajectory.
def sort_pdf_comps(pdf_individual, means, sds, prev_means, t, sds_frac=0.6):
    means = [means[0], means[1]]
    sds = [sds[0]**(0.5), sds[1]**(0.5)]
    pdf0 = pdf_individual[:,0]
    pdf1 = pdf_individual[:,1]
    pdfs = [pdf0, pdf1]

    # two components are interacting if they are sufficently close
    interaction = False
    if abs(means[0] - means[1]) < sds_frac*(sds[0] + sds[1]):
        means = [0.5*(means[0]+means[1])]*2
        sds = [0.5*(sds[0]+sds[1])]*2
        pdfs = [0.5*(np.array(pdf0)+np.array(pdf1))]*2
        pdf0, pdf1 = pdfs
        interaction = True

    if prev_means == [None, None]:
        prev_means = [0.0, 0.0]
        ind_0 = np.argmax([max(pdf0), max(pdf1)])
        ind_1 = np.argmin([max(pdf0), max(pdf1)])
        means = [means[ind_0], means[ind_1]]

    d00 = means[0] - prev_means[0]
    d01 = means[0] - prev_means[1]
    ind_0 = np.argmin([abs(d00), abs(d01)])
    ind_1 = not ind_0

    pdf0 = pdfs[ind_0]
    pdf1 = pdfs[ind_1]
    pdfs = [pdf0, pdf1]
    means = [means[ind_0], means[ind_1]]
    sds = [sds[ind_0], sds[ind_1]]

    ind_0 = np.argmax([max(pdf0), max(pdf1)])
    ind_1 = np.argmin([max(pdf0), max(pdf1)])
    pdf0 = pdfs[ind_0]
    pdf1 = pdfs[ind_1]
    pdfs = [pdf0, pdf1]
    means = [means[ind_0], means[ind_1]]
    sds = [sds[ind_0], sds[ind_1]]

    return pdfs, means, sds, interaction


# Find the valley between two component fit
def pdf_intersection(pdf0, pdf1, tol=1e-15):
    sgn = np.sign(pdf0-pdf1)
    sgn[sgn<tol] = -1     # replace zeros with -1
    try:
        mnj = np.where(np.diff(sgn))[0][0]

    # if only one components, set to the valley to the mean
    except:
        mnj = np.argmax(pdf0)
    return mnj


# plot the gaussian fits on each row
def plot_g_fits(row, js, pdf, pdf0, pdf1, jmx_0, jmx_1, mnj):
    plt.scatter(js[[jmx_0, jmx_1]], pdf[[jmx_0, jmx_1]], c='g', marker='s')
    plt.errorbar(js[mnj], pdf[mnj], yerr=0.05, c='k', marker='v')
    plt.plot(row, alpha=0.8)
    plt.plot(js, pdf0, '--k')
    plt.plot(js, pdf1, '-k')
    plt.plot(js, pdf)	


# collect all fit params through time
def fit_gmix(grid, n=2, sds_frac=0.6):
    M1_0, M2_0, M1_1, M2_1, intersection = [], [], [], [], []
    for t, row in enumerate(grid):
        L = len(row)
        row = row / sum(row)
        js = np.linspace(0, L-1, 100)

        pdf, pdf_individual, means, sds, distrib = gmix(row, js, n=n)

        if t<2:
            prev_means = [None, None]

        pdfs, means, sds, interaction = sort_pdf_comps(pdf_individual, means, 
                                                       sds, prev_means, t,
                                                       sds_frac=sds_frac)
        prev_means = means
        pdf0, pdf1 = pdfs
        mean_ind_0, mean_ind_1 = map(int, means)
        jmx_0, jmx_1 = map(np.argmax, pdfs)

        mnj = 0
        if t>3:
           mnj = pdf_intersection(pdf0, pdf1)

        mnind = int(js[mnj])
        M1_0.append(means[0])
        M1_1.append(means[1])
        M2_0.append(sds[0])
        M2_1.append(sds[1])
        intersection.append(int(js[mnj]))

        #slice_fit__plot = plot_g_fits(row, js, pdf, pdf0, pdf1, jmx_0, jmx_1, mnj)
        #plt.show()
    return map(np.array, (M1_0, M1_1, M2_0, M2_1, intersection))


# collect time series of fit params and fit to their cange across a sweep
# we're collecting the transport properties in a dict
def transport_calc(params, tmax_1, span=[0, 61], n=2, tmin=2, sds_frac=0.6,
    speed_f = ft.flin, speed_Bs_est = [1.0, 0.0],
    diff_f = ft.flin, diff_Bs_est = [1.0, 0.0]):

    transport = {}
    res = h5py.File(io.default_file_name(params, 'data', '.hdf5'))
    exp = ms.get_diag_vecs(res['zz'][::])
    Ptj = ((1.0 - exp)/2.0)[span[0]:span[1]]
    reflect_t = np.argmax(Ptj[:,0])
    transport['grid'] = Ptj

    M1_0, M1_1, M2_0, M2_1, intersection = fit_gmix(Ptj, n=n, sds_frac=sds_frac)

    keys = ['Bs', 'chi2', 'dBs']
    for moment in [1, 2]:
        for peak in [0, 1]:
            moment_name = 'M'+str(moment)+'_'+str(peak)
            mom = eval(moment_name)
            tmax = reflect_t

            if moment == 2:
                tmin = 1
                tmax = np.argmax(M2_0[0:params['L']-2])

            if moment == 1 and peak == 1:
                if tmax_1 is None:
                    tmax_1 = tmax
                transport['tmax_1'] = tmax_1
                tmax = tmax_1

            if moment == 1 and peak == 0:
                transport['intersection'] = intersection[tmin:tmax]

            times = np.array(range(tmin, tmax))

            Bs_chi2_dBs = \
                list(ft.f_fits(speed_f, speed_Bs_est, times, mom[tmin:tmax]))
            transport[moment_name] = eval(moment_name)
            transport['times_'+moment_name] = times
            for j, key in enumerate(keys):
                transport_key = '_'.join([moment_name, key])
                transport[transport_key] = Bs_chi2_dBs[j]
    return transport


# make plots of the fits
def plot_speed_diff(params, transport, fignum=1, speed_f=ft.flin, diff_f=ft.flin):
    grid_fig = plt.figure(fignum, figsize=(6, 4))
    grid_ax = grid_fig.add_subplot(1,2,1)

    # GRID PLOT
    im = grid_ax.imshow(transport['grid'], origin='lower', aspect=1, interpolation='none')
    plt.colorbar(im)


    # MAIN PEAK FIT PLOTS
    times = transport['times_M1_0']
    #grid_ax.errorbar(transport['M1_0'][times[0]:times[-1]+1], times, 
    #        xerr=transport['M2_0'][times[0]:times[-1]+1], c='k')


    pt.plot_time_series(speed_f(transport['M1_0_Bs'], times), grid_ax, 
            times=times,
            rotate=True,
            ny_ticks=10,
            title=r'$P_1(j, t)$',
            ylabel='Iteration', 
            xlabel='Site',
            plot_kwargs={'color':'k', 'linewidth':1}
            )


    # INTERSECTION LINE
    grid_ax.errorbar(transport['intersection'], times, c='w', linewidth=1)


    # SECONDARY PEAK FIT PLOTS

    '''
    times = transport['times_M1_1']
    grid_ax.errorbar(transport['M1_1'][times[0]:times[-1]+1], times, 
            xerr=transport['M2_1'][times[0]:times[-1]+1], c='r')
    '''

    pt.plot_time_series(speed_f(transport['M1_1_Bs'], times), grid_ax, 
            times=times,
            rotate=True,
            ny_ticks=10,
            title=r'$P_1(j, t)$',
            ylabel='Iteration', 
            xlabel='Site',
            plot_kwargs={'color':'r', 'linewidth':1}
            )

    grid_ax.set_xlim( [-0.5, params['L'] - 1 + 0.5])
    grid_ax.set_ylim( [ - 0.5, 60 + 0.5])


    # DIFFUSION PLOTS
    diff_ax = grid_fig.add_subplot(1,2,2)

    times = transport['times_M2_0']
    pt.plot_time_series(transport['M2_0'], diff_ax,
            rotate=True,
            ny_ticks=10,
            #title=make_U_name(mode,  S, V), 
            )

    pt.plot_time_series(diff_f(transport['M2_0_Bs'], times), diff_ax,
            times=times,
            rotate=True,
            title=r'$\sigma_P$', 
            xlabel='width $\sigma_p$ [sites]', 
            ylabel='',
            ytick_labels=False,
            plot_kwargs={'color':'G', 'linewidth':2}
            )

    plt.suptitle(pt.make_U_name(params['mode'],  params['S'], params['V']))
    plt.subplots_adjust(top=.87)


def add_speed(params, transport, fignum, speeds_0, dspeeds_0, diffs_0, ddiffs_0, 
              speeds_1, dspeeds_1, diffs_1, ddiffs_1, ths, **kwargs):

    V = params['V']
    th = pt.get_th(V)
    ths.append(th)
    if th<=18:
        kwargs.update(dict(sds_frac=0.99))
    transport = transport_calc(params, n=2, tmax_1 = transport['tmax_1'], **kwargs)

    speeds_0.append(abs(transport['M1_0_Bs'][0]))
    dspeeds_0.append(abs(transport['M1_0_dBs'][0]))
    diffs_0.append(abs(transport['M2_0_Bs'][0]))
    ddiffs_0.append(abs(transport['M2_0_dBs'][0]))

    speeds_1.append(abs(transport['M1_1_Bs'][0]))
    dspeeds_1.append(abs(transport['M1_1_dBs'][0]))
    diffs_1.append(abs(transport['M2_1_Bs'][0]))
    ddiffs_1.append(abs(transport['M2_1_dBs'][0]))

    if fignum%1 == 0:
        plot_speed_diff(params, transport, fignum=fignum,
                speed_f=kwargs['speed_f'], diff_f=kwargs['diff_f'])

    fignum += 1
    return transport, fignum


def plot_speed_diffs(params, transport, fignum, speeds_0, dspeeds_0, diffs_0, ddiffs_0, 
                speeds_1, dspeeds_1, diffs_1, ddiffs_1, ths, c):

    lsize = 8
    figsize = (2, 2)
    ms = 3.5

    speed_ax = plt.figure(0, figsize=figsize).add_subplot(111)
    diff_ax = plt.figure(1, figsize=figsize).add_subplot(111)
    speed_diff_ax = plt.figure(2, figsize=figsize).add_subplot(111)
    mode_name = '$'+pt.make_mode_name(params['mode'])+'$'

    speed_ax.errorbar(ths, speeds_0, yerr=dspeeds_0, color=c,
            label=mode_name, fmt='o', ms=ms)
    #speed_ax.set_ylim(bottom=0)
    #speed_ax.set_xlim([-10,99])
    speed_ax.legend(loc='upper right', fontsize=lsize)
    speed_ax.set_xlabel('Phase Gate Angle [deg.]')
    speed_ax.set_ylabel('speed [Sites / Iteration]')
    speed_ax.locator_params(nbins=8, axis='x')
    #speed_ax.errorbar(ths, speeds_1, yerr=dspeeds_1, color='c')
    #speed_ax.errorbar(ths, 0.5*(np.array(speeds_0) + np.array(speeds_1)), yerr=dspeeds_1)

    diff_ax.errorbar(ths, diffs_0, yerr=ddiffs_0, color=c, 
                     label=mode_name, fmt='o', ms = ms)
    diff_ax.legend(loc = 'upper right', fontsize=lsize)
    #diff_ax.set_ylim(bottom=0)
    #diff_ax.set_xlim([-10,99])
    diff_ax.set_xlabel('Phase Gate Angle [deg.]')
    diff_ax.set_ylabel('Diffusion Rate [Sites / Iteration]')
    diff_ax.locator_params(nbins=8, axis='x')
    diff_ax.locator_params(nbins=8, axis='y')
    #diff_ax.errorbar(ths, diffs_1, yerr=ddiffs_1)
    #diff_ax.errorbar(ths, 0.5*(np.array(diffs_0) + np.array(diffs_1)), yerr=ddiffs_1)

    speed_diff_ax.errorbar(speeds_0, diffs_0, xerr=dspeeds_0,
            yerr=ddiffs_0, color = c, label=mode_name, fmt='o', ms=ms)
    speed_diff_ax.legend(loc='upper left', fontsize=lsize)
    #speed_diff_ax.set_ylim(bottom=0)
    #speed_diff_ax.set_xlim(left=0)
    speed_diff_ax.set_xlabel('Speed [Sites / Iteration]')
    speed_diff_ax.set_ylabel('Diffusion Rate [Sites / Iteration]')
    speed_diff_ax.locator_params(axis='x', nbins=8)
    speed_diff_ax.locator_params(axis='y', nbins=8)



def make_speeds(fixed_params_dict, var_params_dict, **kwargs):
    params_list_list = io.make_params_list_list(fixed_params_dict, var_params_dict)
    fignum = 3
    for c, params_list in zip(['b', 'g', 'r'], params_list_list):
        speeds_0, dspeeds_0, diffs_0, ddiffs_0 = [], [] ,[], []
        speeds_1, dspeeds_1, diffs_1, ddiffs_1 = [], [] ,[], []
        ths = []
        transport = {'tmax_1' : None}
        for params in params_list:
            transport, fignum = add_speed(params, transport, fignum, speeds_0, dspeeds_0, diffs_0, ddiffs_0, 
                      speeds_1, dspeeds_1, diffs_1, ddiffs_1, ths, **kwargs)

        plot_speed_diffs(params, transport, fignum, speeds_0, dspeeds_0, diffs_0, ddiffs_0, 
                         speeds_1, dspeeds_1, diffs_1, ddiffs_1, ths, c)

    io.multipage(io.base_name(fixed_params_dict['output_dir'][0], 'plots') +
            'L' + str(params['L'])  + '_S6_speeds_R-L_90_gaussian.pdf')



if __name__ == '__main__':
    degs = range(0, 95,5)

    kwargs = dict(span         = [0, 61],
                  sds_frac     = 0.5,
                  tmin         = 0,
                  speed_f      = ft.flin,
                  diff_f       = ft.flin,
                  speed_Bs_est = [1.0, 0.0],
                  diff_Bs_est  = [1.0, 0.0])

    # outer params
    fixed_params_dict = {
                'output_dir' : ['Hphase_small'],
                'L'   : [15],
                'T'   : [60],
                'S'   : [6],
                'IC'  : ['f14'],
                'mode': ['sweep', 'block', 'alt'],
                'BC'  : ['1'],
                 }

    #inner params
    var_params_dict = {
                'V' : ['HP_'+str(deg) for deg in degs],
                 }

    make_speeds(fixed_params_dict, var_params_dict, **kwargs)

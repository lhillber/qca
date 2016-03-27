#!/usr/bin/python3

from os import environ
from itertools import product


import matplotlib.pyplot as plt
import matplotlib          as mpl
import matplotlib.gridspec as gridspec

import h5py
import numpy as np 
from math import pi, sin, cos
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize_scalar
from scipy import signal

import simulation.fio as io
import simulation.measures as ms
import simulation.plotting as pt
import simulation.matrix as mx
import simulation.fitting as ft
from sklearn.mixture import GMM

from scipy.stats.distributions import norm
from sklearn.neighbors import KernelDensity

def gmix(row, js, n=2):
    distrib = []
    mx = max(row)
    mn = min(row)
    for j, val in enumerate(row):
        distrib += [j]*int(1000*val)

    distrib = np.array(distrib)
    model = GMM(n).fit(distrib[:, np.newaxis])
    logprob, responsibilities = model.score_samples(js[:,np.newaxis])
    pdf = np.exp(logprob)
    pdf_individual = responsibilities * pdf[:, np.newaxis]
    means = model.means_
    sds = model.covars_

    if n == 1:
        means = np.array([means[0]]*2)
        sds = np.array([sds[0]]*2)
        pdf_individual = np.asarray([pdf_individual,
            pdf_individual]).reshape(len(pdf_individual),2)
    means = [means[0][0], means[1][0]]
    sds = [sds[0][0], sds[1][0]]
    return pdf, pdf_individual, means, sds, distrib

def sort_pdf_comps(pdf_individual, means, sds, prev_means, t, sds_frac=0.6):
    means = [means[0], means[1]]
    sds = [sds[0]**(0.5), sds[1]**(0.5)]
    pdf0 = pdf_individual[:,0]
    pdf1 = pdf_individual[:,1]
    pdfs = [pdf0, pdf1]

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


def pdf_intersection(pdf0, pdf1, tol=1e-15):
    sgn = np.sign(pdf0-pdf1)
    sgn[sgn<tol] = -1     # replace zeros with -1
    try:
        mnj = np.where(np.diff(sgn))[0][0]
    except:
        mnj = np.argmax(pdf0)
    return mnj


def plot_fits(row, js, pdf, pdf0, pdf1, jmx_0, jmx_1, mnj):
    #plt.scatter(js[[jmx_0, jmx_1]], pdf[[jmx_0, jmx_1]], c='g', marker='s')
    #plt.errorbar(js[mnj], pdf[mnj], yerr=0.05, c='k', marker='v')
    plt.plot(row, alpha=0.8)
    plt.plot(js, pdf0, '--k')
    plt.plot(js, pdf1, '-k')
    plt.plot(js, pdf)	


# Fit a function to rows of a grid
# --------------------------------
def fit_gmix(grid, n=2, sds_frac=0.6):
    M1_0, M2_0, M1_1, M2_1, intersection = [], [], [], [], []
    for t, row in enumerate(grid):
        L = len(row)
        row = row / sum(row)
        js = np.linspace(0, L-1, 100)
        pdf, pdf_individual, means, sds, distrib = gmix(row, js, n=n)

        if t<2:
            prev_means = [None, None]

        pdfs, means, sds, interaction = sort_pdf_comps(pdf_individual, means, sds,
                                            prev_means, t, sds_frac=sds_frac)
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

        slice_fit_plot = plot_fits(row, js, pdf, pdf0, pdf1, jmx_0, jmx_1, mnj)
        #plt.show()
    return map(np.array, (M1_0, M1_1, M2_0, M2_1, intersection))

def get_th(V):
    Vlist = V.split('_')
    return Vlist[1] if ( len(Vlist) == 2 ) else None

if __name__ == '__main__':

    # outer params
    fixed_params_dict = {
                'output_dir' : ['Hphase'],
                'L' : [21],
                'T' : [60],
                'S' : [6],
                'mode': ['alt'],
                'IC': ['f0'],
                'BC': ['1_00']
                 }

    #inner params
    var_params_dict = {
                'V' : ['HP_80'],
                 }

    params_list_list = io.make_params_list_list(fixed_params_dict, var_params_dict)

    for params_list in params_list_list:
        for params in params_list:
            res = h5py.File(io.default_file_name(params, 'data', '.hdf5'))
            mode = params['mode']
            S = params['S']
            V = params['V']
            T = params['T']
            L = params['L']
            th = get_th(V)

            n = 2
            sds_frac = 0.6

            grid = ms.get_diag_vecs(res['zz'][::])
            grid = 0.5 * (1.0 - grid)

            impact_t = np.argmax(grid[:,L-1])

            M1_0, M1_1, M2_0, M2_1, intersection = \
                    fit_gmix(grid, sds_frac=sds_frac, n=n)

            im = plt.imshow(grid, origin='lower', aspect=1, interpolation='none')

            #plt.errorbar(M1_0, range(T+1), xerr=M2_0, c='k', lw=1.5)

            plt.errorbar(M1_1, range(T+1), xerr=M2_1, c='r', lw=1.5)

            Bs, chi2, Bs_sd = ft.f_fits(ft.flin, [1.0, 0.0], range(4, cross_t),
                    M1_0[4:cross_t], xerr=M2_0[4:cross_t])

            plt.plot(ft.flin(Bs, range(impact_t+1)), range(impact_t+1), c='deeppink', lw=1.5)

            #plt.plot([j for j in range(L)], [cross_t]*(L), c='w')


            Bs, chi2, Bs_sd = ft.f_fits(ft.flin, [1.0, 0.0],
                    range(cross_t+1, impact_t), M1_0[cross_t+1:impact_t],
                    xerr=M2_0[cross_t+1:impact_t])

            plt.plot(ft.flin(Bs, range(impact_t+1)), 
                    range(impact_t+1), c='deeppink', lw=1.5)

            plt.plot([j for j in range(L)], [impact_t]*(L), c='w')


            plt.plot(intersection, range(len(intersection)), c='w')
            plt.show()

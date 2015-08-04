#!/usr/bin/python3

from math import sqrt
import numpy as np
import scipy.odr.odrpack as odrpack
import matplotlib.pyplot as plt
import matplotlib as mpl

font = {'family':'normal', 'weight':'bold', 'size':16}
mpl.rc('font',**font)



# Orthogonal distance regression
# ------------------------------
def flin(B, x):
    return B[0]*x + B[1]

def fpow(B, x):
    return B[0]*x**B[1] + B[2]

def fexp(B, x):
    return B[0]*B[1]**(x) +B[2]

def do_odr(f, x, xe, y, ye, estimates):
    model = odrpack.Model(f)
    data = odrpack.RealData(x, y, sx=xe, sy=ye)
    odr = odrpack.ODR(data, model, estimates)
    output = odr.run()
    return odr.run()

def chi2_calc(f, betas, x, y):
    chi2 = 0 
    for xval, yval in zip(x,y):
        chi2 += (yval - f(betas, xval))**2
    return chi2

def f_fits(func, beta_est, x_list, y_list, x_error = None, y_error = None): 
    x_error = [0.0000001] * len(x_list) if x_error is None else x_error
    y_error = [0.0000001] * len(y_list) if y_error is None else y_error
    fit = do_odr(func, x_list, x_error, y_list, y_error, beta_est)
    chi2 = chi2_calc(func, fit.beta, x_list, y_list)
    return fit.beta, chi2

def plot_f_fits(func, beta_est, x_list, y_list, ax, label, color, x_error = None, y_error = None): 
    x_error = [0.0000001] * len(x_list) if x_error is None else x_error
    y_error = [0.0000001] * len(y_list) if y_error is None else y_error

    fit_beta, chi2,  = \
            f_fits(func, beta_est, x_list, y_list, x_error = x_error, y_error = y_error)

    xs = np.linspace(min(x_list), max(x_list), 100)
    ax.plot(x_list, y_list, 'o', label = 'tmax = ' + str(label), color = color)
    ax.legend(loc = 'lower right')
    ax.set_yscale('log', basey=2)
    return


#!/usr/bin/python3

import time_evolve as te
import states as ss
import matrix as mx
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from os import environ
font = {'family':'serif','size':10}
mpl.rc('font',**font)


def make_dat(V):
    Us = np.zeros((16,8,8), dtype=complex)
    es = np.zeros((16,8), dtype=complex)
    for S in range(16):
        U = te.make_U(S, mx.listdot([ss.ops[k] for k in V]))[1]
        evals = np.linalg.eigvals(U)
        es[S][::] = evals[::]
        Us[S][::] = U[::]
    return Us, es

def plot_dat(V):
    fig = plt.figure(1, (14,7))
    Us, es = make_dat(V)
    gs = gridspec.GridSpec(4,4)
    gs.update(top=0.9, wspace=0.3, hspace=0.5)
    S = 0
    for i in range(4):
        for j in range(4):
            U = Us[S]
            evals = es[S]
            gs0 = gridspec.GridSpecFromSubplotSpec(1,2, subplot_spec=gs[i, j],
                    wspace=0.25)
            spec_ax = plt.subplot(gs0[0])
            mat_ax = plt.subplot(gs0[1])

            inds = np.argsort(evals.real)
            spec_ax.plot(np.angle(evals).real[inds])
            mat_ax.imshow(np.abs(U), interpolation='none')
            spec_ax.text(0.95, 1.05, r'$S={}$'.format(S), transform=spec_ax.transAxes)
            #ysmin = evals.min()*7/6
            #ysmax = evals.max()*7/6
            if S == 0:
                ysmin = 0.0
                ysmax = 1.3
                spec_ax.set_ylim([ysmin, ysmax])
            S = S +1
    plt.suptitle('arg(evals) of $U$ and Matrix element Magnitudes of $U$ \n' + r'$V = {}$'.format(V),
            fontsize=12)
    #plt.show()

V_list = ['X', 'H', 'HT','HXT']
for V in V_list:
    plot_dat(V)
    plt.savefig(environ['HOME'] + '/documents/qca/notebook_figs/Umats_V'+str(V)+'.pdf', 
            format='pdf', dpi=300, bbox_inches='tight')

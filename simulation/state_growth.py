#!/usr/bin/python3

import numpy    as np
import simulation.fio      as io
import simulation.matrix   as mx
import simulation.states   as ss
import simulation.measures as ms

import matplotlib.pyplot as plt
import matplotlib as mpl
import simulation.plotting as pt

font = {'size':12, 'weight' : 'normal'}
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rc('font',**font)

Lmin=1
Lmax = 26
Ls = np.linspace(Lmin, Lmax-1, 200)

def mem(L):
    return 2**(L+4)

nb_list = []
L_list = []
for L in range(Lmin, Lmax):
    state = ss.make_state(L,'f0')
    nb = state.nbytes
    del state
    nb_list.append(nb)
    L_list.append(L)

fig = plt.figure(1, (7,2.5))
ax = fig.add_subplot(111)
ax.plot(Ls, mem(Ls), '--c', label='theoretical', lw=2)
ax.plot(L_list, nb_list, 'ks', label='measured', ms=5)
ax.set_ylabel(r'$N_{\mathrm{bytes}}$')
ax.set_xlabel(r'system size [$L$]')
ax.set_yscale('log', basey=2)
ax.set_yticks([2**n for n in range(6,29,4)])
#ax.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
ax.legend(loc='lower right')
ax.grid('on')
ax.set_ylim([2**(Lmin+3), 2**(Lmax+4)])
ax.set_xlim([Lmin-0.5, Lmax-0.5])
#plt.show()
plots_fname = io.base_name('timing', 'plots')+'state_memory_use.pdf'
io.multipage(plots_fname)

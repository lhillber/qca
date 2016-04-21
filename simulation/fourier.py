#!/usr/bin/python3
from os import environ
import simulation.fio as io
import simulation.plotting as pt
import simulation.measures as ms
import simulation.matrix as mx
import matplotlib.pyplot as plt
import simulation.peakdetect as pd

import numpy as np
import h5py

import matplotlib          as mpl
import matplotlib.cm as cm

from itertools import product
from math import pi, sin, cos

# default plot font
# -----------------
font = {'size':12, 'weight' : 'normal'}
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rc('font',**font)


def main():
    output_dir = 'fock_IC'
    data_repo = '/mnt/ext0/qca_output/'+output_dir+'/data/' 
    #data_repo = None

    modes = ['alt']
    #uID = '_somethin_'
    uID = ''
    IC_label = 'fock'
    Ls = [17,20]
    degs = [0]
    Ss = [1,6,9,14]

    ICs = ['c3_f1']
    #ICs = ['G', 'W', 'c2_B0-1_0', 'c2_B0-1_1', 'c2_B0-1_2', 'c2_B0-1_3' ]
    #ICs = ['c3_f1', 'c3_f0-1', 'c3_f0-2', 'c3_f0-1-2']
    #ICs = ['r5-10', 'r5-20', 'r5-30']


    obs_list = ['z']

    # outer params
    fixed_params_dict = {
                'output_dir' : [output_dir],
                'T'   : [1000],
                'BC'  : ['1_00'],
                 }

    #inner params
    var_params_dict = {
                'L'   : Ls,
                'V' : ['HP_'+str(deg) for deg in degs],
                'IC'  : ICs,
                'S'   : Ss,
                'mode': modes
                 }

    D = 10
    # define the colormap
    cmap = cm.rainbow
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

    params_list_list = io.make_params_list_list(fixed_params_dict, var_params_dict)
    M, N = params_list_list.shape
    J = len(obs_list)
    fignum=0

    sim_pk_freqs = np.zeros((M*N,J, D))
    sim_pk_freqs = dict(zip(obs_list,[[]]*len(obs_list)))
    for m, params_list in enumerate(params_list_list):
        for n, params in enumerate(params_list):
            L = params['L']
            T = params['T']

            i = N*m + n
            print(m,n,i)
            for j, obs_key in enumerate(obs_list):
                if data_repo is not None:
                    sname = io.sim_name(params)
                    res_path = data_repo + sname + '_v0.hdf5'
                elif data_repo is None:
                    res_path = io.default_file_name(params, 'data', '.hdf5')
                res = h5py.File(res_path)

                if obs_key in ['ND', 'CC', 'Y']:
                    FT = res['F'+obs_key][::]
                    freqs = res['freqs'][::]

                if obs_key in ['s']:
                    dat = res[obs_key][::]
                if obs_key in ['z']:
                    dat = ms.get_diag_vecs(res[(obs_key+obs_key)][::])

                    #dat = np.mean(dat, axis=1)
                    #print(dat)
                    #freqs, FT, rn = ms.make_ft(dat)
                #ind = np.argpartition(FT, -D)[-D:]
                #frq = freqs[ind]
                #amp = FT[ind]
                #mx, mn = pd.peakdetect(FT, freqs, lookahead=1, delta=0.005)
                #frq = [m[0] for m in mx]
                #amp = [m[1] for m in mx]
                #sim_pk_freqs[obs_key].append(freqs[ind])

                #plt.semilogy(freqs, FT)
                #plt.scatter(frq, amp, c='r')
                #plt.ylim(bottom=1e-6)
                #plt.imshow(board[0:60,::], interpolation='none', origin='lower')


                tmn = 40
                Dt = 100
                for tmx in range(tmn+Dt,1000,Dt):
                    board = dat[tmn:tmx,::]
                    Fkw = np.fft.fftshift(np.fft.fft2(board))
                    Fkw = np.log(np.abs(Fkw))
                    fig = plt.figure(1, figsize=(3,3))
                    fax = fig.add_subplot(121)
                    cfax = fax.imshow(Fkw, interpolation="none", origin='lower',
                            aspect='auto')
                    cbar=fig.colorbar(cfax)

                    tax = fig.add_subplot(122)
                    ctax = tax.imshow(board, interpolation="none", origin='lower',
                            aspect='auto')

                    ctax.set_title(r'$S = $' + str())
                    cbar=fig.colorbar(ctax)
                    plt.show()


    bn = io.base_name(fixed_params_dict['output_dir'][0], 'plots/scatters') 
    fname = IC_label + '_'+'-'.join(modes) + '_L'+'-'.join(map(str,Ls)) + uID + 'fourier.pdf'
    print(fname)

    for bins in [10]:
        plt.hist(np.array(sim_pk_freqs['s']).flatten(), bins=bins)
        #plt.show()
    
    io.multipage(bn+fname, clip=clip)


if __name__ == '__main__':
    main()

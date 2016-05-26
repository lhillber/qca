#!/usr/bin/python3
from os import environ
import simulation.fio as io
import simulation.plotting as pt
import simulation.measures as ms
import simulation.matrix as mx
import matplotlib.pyplot as plt
import simulation.peakdetect as pd

from scipy import interpolate
import scipy.fftpack       as spf

import numpy as np
import h5py

import matplotlib          as mpl
import matplotlib.cm as cm
from matplotlib import ticker
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator

from itertools import product
from math import pi, sin, cos

# default plot font
# -----------------
font = {'size':12, 'weight' : 'normal'}
#mpl.rcParams['mathtext.fontset'] = 'stix'
#mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rc('font',**font)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def init_params_1d():
    output_dir = 'fock_IC'
    data_repo = '/mnt/ext0/qca_output/'+output_dir+'/data/' 
    #data_repo = None
    #uID = '_somethin_'
    uID = '_ICc3_f1_'
    IC_label = 'fock'

    modes = ['alt']
    Ss = [1,2,3,4,5,6,7,9,10,11,12,13,14]
    degs = [90]
    Vs = ['HP_'+str(deg) for deg in degs]
    Ls = [17]
    Ts = [1000]
    ICs = ['c3_f1']
    #ICs = ['G', 'W', 'c2_B0-1_0', 'c2_B0-1_1', 'c2_B0-1_2', 'c2_B0-1_3' ]
    #ICs = ['c3_f1', 'c3_f0-1', 'c3_f0-2', 'c3_f0-1-2']
    #ICs = ['r5-10', 'r5-20', 'r5-30']

    obs_list = ['ND','CC','Y','z']
    # outer params
    fixed_params_dict = {
                'output_dir' : [output_dir],
                'L'   : Ls,
                'T'   : Ts,
                'BC'  : ['1_00'],
                 }
    #inner params
    var_params_dict = {
                'V' : Vs,
                'IC'  : ICs,
                'S'   : Ss,
                'mode': modes
                 }
    params_list_list = io.make_params_list_list(fixed_params_dict, var_params_dict)

    bn = io.base_name(fixed_params_dict['output_dir'][0], 'plots/fourier')
    def plots_out_pathf(L):
        return bn + IC_label + '_'+'-'.join(modes) + '_L'+str(L)+ uID +\
    'DCYZ.pdf'
    return params_list_list, obs_list, data_repo, plots_out_pathf

def init_params_2d():
    output_dir = 'fock_IC'
    data_repo = '/mnt/ext0/qca_output/'+output_dir+'/data/' 
    #data_repo = None
    #uID = '_somethin_'
    uID = '_ICc3_f1_VHP_0_'
    IC_label = 'fock'

    modes = ['alt']
    Ss = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    degs = [0]
    Vs = ['HP_'+str(deg) for deg in degs]
    Ls = [17]
    Ts = [1000]
    ICs = ['c3_f1']
    #ICs = ['G', 'W', 'c2_B0-1_0', 'c2_B0-1_1', 'c2_B0-1_2', 'c2_B0-1_3' ]
    #ICs = ['c3_f1', 'c3_f0-1', 'c3_f0-2', 'c3_f0-1-2']

    obs_list = ['z']
    # outer params
    fixed_params_dict = {
                'output_dir' : [output_dir],
                'L'   : Ls,
                'T'   : Ts,
                'BC'  : ['1_00'],
                 }
    #inner params
    var_params_dict = {
                'V' : Vs,
                'IC'  : ICs,
                'S'   : Ss,
                'mode': modes
                 }
    params_list_list = io.make_params_list_list(fixed_params_dict, var_params_dict)
    bn = io.base_name(fixed_params_dict['output_dir'][0], 'plots/fourier')
    def plots_out_pathf(L):
        return bn + IC_label + '_'+'-'.join(modes) + '_L'+str(L)+ uID +\
               'ft2d.pdf'
    return params_list_list, obs_list, data_repo, plots_out_pathf

def obs_label(obs_key, _2d=False):
    pn = {'s': r's^{\mathrm{vN}}',
            'ND': r'\mathcal{D}',
            'CC': r'\mathcal{C}',
             'Y': r'\mathcal{Y}',
             'z': r'\langle \overline{\sigma_j^z} \rangle',
             'x': r'\langle \overline{\sigma^x} \rangle',
             'y': r'\langle \overline{\sigma^y} \rangle'
            }
    if _2d:
        pn = {'s': r's^{\mathrm{vN}}',
                'ND': r'\mathcal{D}',
                'CC': r'\mathcal{C}',
                 'Y': r'\mathcal{Y}',
                 'z': r'\langle \sigma_j^z \rangle',
                 'x': r'\langle \sigma_j^x \rangle',
                 'y': r'\langle \sigma_j^y \rangle'
                }
    return pn[obs_key]

# Autocorrelation function of vector x with lag h
# -----------------------------------------------
def autocorr(x, h=1):
    N = len(x)
    mu = np.mean(x)
    acorr = sum( (x[j] - mu) * (x[j+h] - mu) for j in range(N-h))
    denom = sum( (x[j] - mu)**2 for j in range(N) )
    if denom > 1e-14:
        acorr = acorr/denom
    else:
        print('auto correlation less than', 1e-1, 1e-144)
    return acorr

# Red noise as a function of frequency for power spectrum amps
# ------------------------------------------------------------
def red_noise(time_series, dt=1, h=1):
    a1 = autocorr(time_series, h=1)
    a2 = np.abs(autocorr(time_series, h=2))
    a = 0.5 * (a1 + np.sqrt(a2))

    def RN(f):
        rn = 1 - a**2
        rn = rn / (1 - 2*a*np.cos(2*pi*f/dt) + a**2)
        return rn
    return RN

# make Fourier transform of time series data
# ------------------------------------------
def make_ft_1d(time_series, dt=1, h=1):
    # set nan's to 0
    time_series = np.nan_to_num(time_series)
    time_series = time_series - np.mean(time_series)
    amps = np.abs(spf.rfft(time_series))**2
    amps = amps/sum(amps)
    ws = spf.rfftfreq(len(amps), d=dt)
    rn = red_noise(time_series, dt=dt, h=h)(ws)
    rn = rn * sum(amps)/sum(rn)
    return ws, amps, rn

def make_ft_2d(board, dt=1, dx=1):
    # set nan's to 0
    T = len(board)
    end = int((T-1)/2)
    if T%2 == 0:
        end = int(T/2)


    L = len(board[0])
    endk = int((L-1)/2)
    if T%2 == 0:
        endk = int(L/2)
    board = np.nan_to_num(board)
    ny, nx = board.shape
    # NOTE: The slice at the end of amps is required to cut negative freqs from
    # axis=0 of #the rfft. This appears to be a but. Report it.
    fraw = np.fft.fft2(board)
    amps = (np.abs(fraw)**2).real
    iboard = np.fft.ifft2(fraw).real

    amps = amps[0:end+1,0:endk+1]
    amps = amps/np.sum(amps)
    ws = np.fft.rfftfreq(ny, d=dt)[0:end+1]
    ks = np.fft.rfftfreq(nx, d=dx)[0:endk+1]
    return ws, ks, amps, iboard, board

def make_tick_d(mx, mn, n_ticks):
    i=0
    d = round((mx - mn)/n_ticks, 1)
    while d<=0:
        d = round((mx - mn)/n_ticks, i)
        i+=1
    return d

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N 


def get_ft_1d(res, obs_key):
    if obs_key in ['ND', 'CC', 'Y']:
        dat = res[obs_key][::]
    if obs_key in ['s']:
        dat = res[obs_key][::]
        dat = np.mean(dat, axis=1)
    if obs_key in ['z', 'x']:
        dat = ms.get_diag_vecs(res[(obs_key+obs_key)][::])
        dat = np.mean(dat, axis=1)
    freqs, FT, rn = make_ft_1d(dat)
    return freqs, FT, rn


def get_ft_2d(res, obs_key, tmn, tmx):
    if obs_key in ['s']:
        dat = res[obs_key][tmn:tmx,::]
    if obs_key in ['z', 'x', 'y']:
        dat = ms.get_diag_vecs(res[(obs_key+obs_key)][::])[tmn:tmx,::]
    ws, ks, amps, iboard, board = make_ft_2d(dat)
    return ws, ks, amps, iboard, board

def plot_ft_1d(freqs, amps, rn, params, row, nrows, obs_key,
        interval_factor=5.991, fignum=1):
    fpks = params['fpks'][obs_key]
    pks = params['pks'][obs_key]
    freqs_trunc = freqs[0:len(amps)]
    fig = plt.figure(fignum, figsize=(3,3.25))
    ax = fig.add_subplot(nrows, 1,row+1)
    ax.semilogy(freqs_trunc, amps, '-k', lw=0.9, zorder=1)
    ax.semilogy(freqs, rn, 'b', lw=0.9, zorder=2)
    ax.semilogy(freqs, rn*interval_factor, '--b', lw=0.9, zorder=2)
    ax.scatter(fpks, pks, c='r', marker='*', linewidth=0,
            s=35, zorder=2)

    #ax.scatter(minfrq, minamp, c='g', marker='v', linewidth=0,
    #        s=25, zorder=2)



    ax.yaxis.set_major_locator(mpl.ticker.LogLocator(numticks=3))
    if params['S'] == 14 and obs_key == 'CC':
        print('enter')
        ax.yaxis.set_major_locator(mpl.ticker.LogLocator(numticks=2))

    ax.set_ylim([np.min(amps[10::])/5, np.max(amps)*5])
    ax.set_xlim([0,0.5])
    ax.minorticks_off()
    ax.get_xaxis().tick_bottom()   # remove unneeded ticks 
    ax.get_yaxis().tick_left()

    #ax.xaxis.set_ticks_position('none')
    #ax.yaxis.set_ticks_position('none')
    #ax.grid('on')
    fticks = False
    if row == nrows-1:
        fticks = True
        ax.set_xlabel('Frequency')
    plt.setp([ax.get_xticklabels()], visible=fticks)
    ax.set_ylabel(r'$\mathcal{F}\big ('+obs_label(obs_key)+r'\big )$')
    letters = {14:'(a)', 6:'(b)'}
    if params['S'] in letters.keys():
        title = letters[params['S']]
        fig.suptitle(title)
    plt.subplots_adjust(hspace=0.3, top=0.93)

def get_ft_peaks(freqs, FT, rn, win = 5, interval_factor=5.991):
    # smoothing spectrum with win-point moving average
    FT = running_mean(FT, win)
    freqs_trunc = freqs[0:len(FT)]
    # iterpolate to make a red noise function
    RNf = interpolate.interp1d(freqs, rn)
    # extract peaks
    mx, mn = pd.peakdetect(FT, freqs_trunc, lookahead=4,
            delta=np.std(FT))
    frq = np.array([m[0] for m in mx])
    amp = np.array([m[1] for m in mx])
    # keep peaks above 95% conf level of red noise
    pk_ind = RNf(frq) * interval_factor < amp
    fpks = frq[pk_ind]
    pks = amp[pk_ind]
    return fpks, pks, FT, freqs_trunc

def get_data_in_path(data_repo, params):
    if data_repo is not None:
        sname = io.sim_name(params)
        data_in_path = data_repo + sname + '_v0.hdf5'
    elif data_repo is None:
        data_in_path = io.default_file_name(params, 'data', '.hdf5')
    return data_in_path

def set_id_position(params, id_by='S', id_pos=0, id_dict={}):
    idd = params[id_by]
    if idd in id_dict:
        x = id_dict[idd]
    else:
        id_pos += 1
        id_dict[idd] = id_pos
        x = id_pos
    return x, id_pos, id_dict


def plot_agg_peaks(params_res_list, obs_list, id_by ='S'):
    fig2 = plt.figure(100,figsize=(6,4))
    ax2 = fig2.add_subplot(111)
    colors = [ 'r', 'g', 'b', 'orange']
    id_dict = {}
    id_pos = 0
    labeled = False
    for jj, params in enumerate(params_res_list):
        L = params['L']
        for ii, (c, obs_key) in enumerate(zip(colors, obs_list)):
            # get peak frequencies, ignor simulations with no peaks
            plot_freqs = params['fpks'][obs_key]
            if len(plot_freqs)==0:
                continue

            # set x coordinate for unique id_by
            x, id_pos, id_dict = set_id_position(params, id_by=id_by, id_pos=id_pos,
                                                 id_dict=id_dict)
            xs = np.array([x]*len(plot_freqs)) + ii/7 - len(obs_list)/14


            # label scatter plot only once, may have to change j0 in jj == j0
            label=None
            if params['S'] == 9 and not labeled:
                label = label='$'+obs_label(obs_key)+'$'
                if ii == len(obs_list)-1:
                    labeled = True

            # plot the peak frequencies vs id
            ax2.scatter(xs, plot_freqs, c=c, linewidth=0.1, alpha=0.7,
                    label=label, s=50)

    # plot vertical partitions
    for x0 in range(2, len(id_dict.keys())+1):
        ax2.axvline(x=x0-0.5, ymin=-0.1, ymax = 1, linewidth=1, alpha=0.5,
                color='k', linestyle='--')

    # plot the 'bouncing frequency' assuming a speed of 1 site/iteration
    # try higher harmonics with the substitution 1/L -> k/L for
    ax2.axhline(y=1/L, xmin=0.0, xmax = 1, linewidth=1.5,
            color='k', linestyle=':')

    ax2.set_xticks(range(1, len(id_dict.keys())+1))
    ax2.set_xticklabels(list(id_dict.keys()))
    ax2.legend(bbox_to_anchor=[1.31, 1.03], loc='upper right', scatterpoints=1)
    ax2.set_xlabel('Rule number [$S$]')
    ax2.set_ylabel('Peak frequency')
    ax2.set_ylim([-0.05,0.55])

def run_1d_ft():
    params_list_list, obs_list, data_repo, plots_out_pathf = init_params_1d()
    M, N = params_list_list.shape
    J = len(obs_list)
    interval_factor=5.991
    for m, params_list in enumerate(params_list_list):
        params_res_list = []
        for n, params in enumerate(params_list):
            data_in_path = get_data_in_path(data_repo, params)
            params['fpks'] = {}
            params['pks'] = {}
            L = params['L']
            T = params['T']
            i = N*m + n
            for j, obs_key in enumerate(obs_list):
                res = h5py.File(data_in_path)
                freqs, FT, rn = get_ft_1d(res, obs_key)
                fpks, pks, FT, freqs_trunc = get_ft_peaks(freqs, FT, rn,
                        interval_factor=interval_factor)
                # save peaks to params dict
                params['fpks'][obs_key] = fpks
                params['pks'][obs_key] = pks
                params_res_list.append(params)
                plot_ft_1d(freqs, FT, rn, params, j, J, obs_key, fignum=n,
                        interval_factor=interval_factor)
                j += 1
        plot_agg_peaks(params_res_list, obs_list, id_by ='S')
        # save all plots
        plots_out_path = plots_out_pathf(L)
        print(plots_out_path)
        io.multipage(plots_out_path, clip=True)

def run_2d_ft(fs=12):
    params_list_list, obs_list, data_repo, plots_out_pathf = init_params_2d()
    M, N = params_list_list.shape
    interval_factor=5.991
    letters = ['(c)', '(a)', '(b)', '(c)']
    labeled_rules = [1, 4 ,6, 14]
    letter_dict = dict(zip(labeled_rules, letters))


    for m, params_list in enumerate(params_list_list):
        params_res_list = []
        for n, params in enumerate(params_list):
            data_in_path = get_data_in_path(data_repo, params)
            params['fpks'] = {}
            params['pks'] = {}
            L = params['L']
            T = params['T']
            S = params['S']
            V = params['V']
            th = pt.get_th(V)
            i = N*m + n
            for j, obs_key in enumerate(obs_list):
                res = h5py.File(data_in_path)
                tmn = 300
                tmx = 1000
                ws, ks, amps, iboard, board = get_ft_2d(res, obs_key, tmn, tmx)
                fig = plt.figure(i, figsize=(1.5,4))
                tfig = plt.figure(i+100, figsize=(2,6))
                fax = fig.add_subplot(111)
                tax = tfig.add_subplot(111)
                amps[0,0]=np.nan
                try:
                    cax = fax.imshow(amps, interpolation="none", origin='lower',
                            aspect='auto', norm=LogNorm() )
                    cbar=fig.colorbar(cax)
                    tcax = tax.imshow(iboard, interpolation="none", origin='lower',
                            aspect='auto')
                    tcbar=tfig.colorbar(tcax)
                except:
                    cax = fax.imshow(amps, interpolation="none", origin='lower',
                            aspect='auto')
                    tcax = tax.imshow(iboard, interpolation="none", origin='lower',
                            aspect='auto')
                    tcbar=tfig.colorbar(tcax)


                cbar.ax.tick_params(labelsize=fs)
                cbar.set_label(r'$\mathcal{F}\big('+obs_label(obs_key,
                    _2d=True)+r'\big)$', fontsize=fs, rotation=0, y=.75,
                    labelpad=-0.1)
                cbar.ax.yaxis.set_major_locator(mpl.ticker.AutoLocator())
                cbar.ax.locator_params(nbins=4)
                #cbar.update_ticks()
                tcbar.set_label(r'$' + obs_label(obs_key, _2d=True) + r'$',
                        fontsize=fs)

                wtick_lbls = ws[::int(len(ws)/5)]
                iw = np.in1d(ws, wtick_lbls)
                wticks = np.arange(0,len(amps)+1)[iw]
                ktick_lbls = ks[::int(len(ks)/2)+1]
                ik = np.in1d(ks, ktick_lbls)

                kticks = np.arange(0,len(amps[0]))[ik]

                fax.set_yticks(wticks)
                fax.set_yticklabels(np.around(wtick_lbls, decimals=2),
                        fontsize=fs)
                fax.set_xticks([0,L/4,L/2])
                fax.set_xticklabels([0, 0.25, 0.5], fontsize=fs)

                fax.set_ylabel(r'$f$', fontsize=fs)
                fax.set_xlabel(r'$k$', fontsize=fs)
                tax.set_ylabel(r'$Iteration$', fontsize=fs)
                tax.set_xlabel(r'$Site$', fontsize=fs)

                if S == 1:
                    title = r'$'+str(th)+r'^{\circ}$'
                    if th == 0:
                        title = r'$\theta='+str(th)+r'^{\circ}$'
                if S != 1:
                    title = r'${}$'.format(S)
                    if S == 6:
                        title = r'$S={}$'.format(S)

                if S in labeled_rules:
                    panel_label = letter_dict[S]
                    #fax.text(0.5, -0.17, panel_label,
                    #verticalalignment='bottom', horizontalalignment='center',
                    #transform=fax.transAxes, fontsize=fs)
                fax.set_title(' ' + title, fontsize=fs+1)
                tax.set_title(title, fontsize=fs+1)
                plt.subplots_adjust(top=0.85, bottom=0.15, wspace=0.6)
        plots_out_path = plots_out_pathf(L)
        print(plots_out_path)
        io.multipage(plots_out_path, clip=True)


def main():
    #run_1d_ft()
    run_2d_ft()

if __name__ == '__main__':
    main()

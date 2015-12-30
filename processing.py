#!/usr/bin/python

from cmath  import sqrt
from math   import fabs, sin, pi
from collections import OrderedDict
from itertools import product
import copy
import csv
import numpy as np
import information as im
import fio as io
import networkmeasures as nm
import measures as ms
import states as ss
import matplotlib        as mpl
import matplotlib.pyplot as plt
import plotting as pt
import matplotlib.gridspec as gridspec

# default plot font to bold and size 16
# -------------------------------------
font = {'family':'serif', 'size':9}
mpl.rc('font',**font)

eq = 1.0/sqrt(2.0)

def get_diag_vecs(mats):
    return np.array([mat.diagonal() for mat in mats])

def get_offdiag_mats(mats):
    L = len(mats[0])
    mats_out = copy.deepcopy(mats)
    for t, mat in enumerate(mats_out): 
        mat[np.arange(L), np.arange(L)] = 0.0
        mats_out[t] = mat
    return mats_out

# make single site von Neuman entropy in space and time
def make_local_vn_entropy(results):
    return get_diag_vecs(results['sjk'])

# make single site density matricies fom si results
# -------------------------------------------------
def make_rjt_mat(results):
    return results['sdr'] + 1.0j*results['sdi'] 

# exp vals of op wrt all single site density matricies in space and time
# ----------------------------------------------------------------------
def local_exp_vals(rjt_mat, op):
    return np.asarray([ [ np.trace(r_jt.dot(op)).real for r_jt in r_jlist]\
               for r_jlist in rjt_mat ])


# apply network measures to list of networks; 
# typ='avg' for time series, typ='st' for spatialy resolved measure
# -----------------------------------------------------------------
def measure_networks(nets, typ='avg', cut_first=0):
    if typ=='avg':
        measure_tasks=['ND', 'CC', 'Y']

    if typ=='st':
        measure_tasks=['EV', 'CC', 'Y']

    measures = {} 
    for task in measure_tasks:
        measures[task] = np.asarray([ms.NMcalc(net, typ=typ,
                                    tasks=measure_tasks)[task] for net in
                                    nets[cut_first::]])
    return measures





# Specific functions
# ==================
def make_net_dict(results, net_types=['nz', 'nx', 'mi']):
    net_dict = {}
    for net_typ in net_types:
        if net_typ == 'mi':
            network = results['mi']
        else: 
            network = spatialnetworksQ(results, net_typ)
            #network = get_offdiag_mats(results[net_typ])
            #network = correlator_calc(results[net_typ])
        net_dict[net_typ] = network
    return net_dict

def spatialnetworksQ(results, typ):
    n_mats = results[typ]
    corr, loc = get_offdiag_mats(n_mats), get_diag_vecs(n_mats)
    return im.spatialnetworksQ(corr, loc)

def make_comp_data(Cparams, Qparams, avg_tasks=['ND','CC','Y']):
    Cres = io.read_results(Cparams, typ='C')
    Qres = io.read_results(Qparams, typ='Q')
    
    times = Qres['t'] 
    
    Qmi_dict = make_net_dict(Qres)
    Qmi_nz = Qmi_dict['nz']
    Qmi    = Qmi_dict['mi']
    
    Cmi   = Cres['mi'] 
    
    Qnz_avg_net_measures  = measure_networks(Qmi_nz, 
                                            tasks=avg_tasks, typ='avg')
    Qmi_avg_net_measures  = measure_networks(Qmi, 
                                            tasks=avg_tasks, typ='avg')
    Cmi_avg_net_measures  = measure_networks(Cmi, 
                                            tasks=avg_tasks, typ='avg')

    measures_dict = {'Qnz_m' : Qnz_avg_net_measures,
                     'Qmi_m' : Qmi_avg_net_measures,
                     'Cmi_m' : Cmi_avg_net_measures}
    
    avg_measures_dict = {} 
    for typ in measures_dict.keys():
        measures = measures_dict[typ] 
        avg_measures_dict[typ] = {}
        for task in Qnz_avg_net_measures.keys():
            measure = measures[task] 
            r_avg = running_average(measure)
            avg_measures_dict[typ][task] = r_avg

    return avg_measures_dict

def make_long_time_avgs(Cparams_list, Qparams_list, win_len=10):
    stats = {'avg' : np.mean, 'std' : np.std}
    N = len(Cparams_list)
    data = []
    for n, (Cparams, Qparams) in enumerate(zip(Cparams_list, Qparams_list)):
        avg_meas_dict = make_comp_data(Cparams, Qparams)
        #running_avg_plots(avg_measures_dict, fignum=fignum)
        #plt.show() 
        #plt.close('all')
        dat = {} 
        for typ in avg_meas_dict.keys():
            dat[typ] = {}
            avg_measures = avg_meas_dict[typ] 
            
            for task in avg_measures.keys():
                dat[typ][task] = {} 
                for stat in ('avg', 'std'):
                    dat[typ][task][stat] = {}
                    r_avg = avg_measures[task]
                    d = stats[stat](r_avg[-win_len:])
                    dat[typ][task][stat] = d
        data.append(dat)

    res = {}
    for typ in avg_meas_dict.keys():
        res[typ] ={}
        for task in avg_measures.keys():
            res[typ][task] = {}
            for stat in ('avg', 'std'):
                res[typ][task][stat] = [data[n][typ][task][stat] for n in range(N)]

    return res


def long_time_avg_plots(long_time_avgs, fignum=0):
    fignum=fignum
    for typ in long_time_avgs.keys():
        Tavg_measures = long_time_avgs[typ] 
        for task in Tavg_measures.keys():
            fignum = (fignum+1)%3
            dat = Tavg_measures[task]
            
            fig = plt.figure(fignum)
            plt.errorbar(range(1,len(dat['avg'])+1), dat['avg'],
                    yerr=dat['std'], label = typ) 
            plt.legend()
            plt.title(task)
            plt.tight_layout()
            plt.xlabel(r'$|i-j|$')
            plt.ylabel('avg measure')
    
def running_avg_plots(avg_measures_dict, fignum=0):
    fignum=fignum
    for typ in avg_measures_dict.keys():
        avg_measures = avg_measures_dict[typ] 
        for task in avg_measures.keys():
            fignum = (fignum+1)%3
            r_avg = avg_measures[task]
            pt.plot_time_series(r_avg, typ[:3], label= typ[:3] + ' ' + task,
                    fignum=fignum)
            plt.xlabel('Time')
            plt.ylabel('Measure average')

def correlator_calc(mats):
    one_corr = get_diag_vecs(mats)
    two_corr = get_offdiag_mats(mats)
    tmax = len(mats)
    L = len(mats[0]) 
    corr = np.zeros( (tmax + 1, L) )
    dist = []
    i = 3
    for t in range(tmax):
        for j in range( L):
            corr[t, j] = two_corr[t, i, j] - one_corr[t, i] * one_corr[t, j]
            if t==0: 
                dist.append(fabs(i-j))
    return corr, dist



# Data comparisons
# ----------------

def make_params(output_name, mode, center_op, R, IC, L, tmax):
        params = OrderedDict( [ 
                ('output_name', output_name), 
                ('mode', mode),
                ('center_op', center_op),
                ('R', R), 
                ('IC', IC), 
                ('L', L), 
                ('tmax', tmax) 
                ] )
        return params

def concat_dicts(d1, d2):
    d = d1.copy()
    d.update(d2)
    return d

def dict_product(dicts):
        return (dict(zip(dicts, x)) for x in product(*dicts.values()))

def make_params_list_list(fixed_params_dict, var_params_dict):
    unordered_params_dict_list = [0]*len(list(dict_product(fixed_params_dict)))
    
    for index, fixed_dict in enumerate(dict_product(fixed_params_dict)):
        unordered_params_dict_sublist = []
        
        for var_dict in dict_product(var_params_dict):
            unordered_params_dict_sublist = unordered_params_dict_sublist + \
                            [concat_dicts(fixed_dict, var_dict)]
        
        unordered_params_dict_list[index] = unordered_params_dict_sublist

    params_list_list = [[make_params( unordered_params_dict['output_name'],
                                unordered_params_dict['mode'],
                                unordered_params_dict['center_op'],
                                unordered_params_dict['R'],
                                unordered_params_dict['IC'],
                                unordered_params_dict['L'],
                                unordered_params_dict['tmax'])
                    for unordered_params_dict in
                    unordered_params_dict_sublist ]
                    for unordered_params_dict_sublist in
                    unordered_params_dict_list ] 
    return params_list_list


def inner_title(ax, title, l=0.1, u=0.95):
    return ax.text(l, u, str(title), horizontalalignment='left', transform=ax.transAxes)

def make_measures_dict_list(res_list):
    mi_nets_list = [res['mi'] for res in res_list]
    ipr_list = [res['ipr'] for res in res_list]
    measures_dict_list = [ 
            concat_dicts(measure_networks(mi_nets, typ='avg', cut_first=0), {'ipr':ipr})
            for (mi_nets, ipr) in zip(mi_nets_list, ipr_list)
            ]
    return measures_dict_list

# compute running average through time of a time series
# -----------------------------------------------------
def equib_calc(time_series, tol=0.1):
    tmax = len(time_series)
    n_equib = tmax*3/4.0
    
    eq_flag = 0
    tot = 0
    r_avg = [0]*tmax
    for n, d in enumerate(time_series):
        tot += d
        r_avg[n] = tot/(n+1)
        if n>1 and eq_flag==0:
            if abs(r_avg[n-2] - r_avg[n-1]) < tol\
                and abs(r_avg[n-1] - r_avg[n]) < tol:
                n_equib = n
                eq_flag=1 

    val_equib = np.mean(time_series[n_equib::])

    dval_equib = np.std(time_series[n_equib::])
    return r_avg, n_equib, val_equib, dval_equib


def measures_comp_plot(res_list, var_params_dict, suptitle = 'suptitle',
        fignum=1, FT='True', meas_tasks = ['CC', 'Y', 'ipr']):
    start_t = 0 
    end_t = 1000
    dt = 1
    cut_first = 100
    
    meas_labels = {'ND':'ND', 'CC':'CC', 'Y':'Y', 'ipr':'IPT'}
    
    colors = ["0.0", "0.3", ".6"]
    curve_labels = [''.join(var_params_dict['center_op'][i])
            for i in range(len(var_params_dict['center_op']))]

    nrow = len(meas_tasks)

    if FT == 'True':
        ncol = 2

    elif FT == 'False' or FT == 'Only':
        ncol = 1

    measures_dict_list = make_measures_dict_list(res_list)
    
    fig = plt.figure(fignum) 
    gs = gridspec.GridSpec(nrow, 1)
    gs.update(left=0.1, hspace = 0.1)

    gs_list = np.asarray([gridspec.GridSpecFromSubplotSpec(1, ncol,
        subplot_spec=gs[k], wspace=0.25) for k in range(nrow)])

    ymax = 0 
    ymin = 1E10 
    ax_list = []
    for meas_ind, (pair_gs, meas_task) in enumerate(zip(gs_list, meas_tasks)):
        for ind, (measures_dict, color, curve_label) in \
            enumerate(zip(measures_dict_list, colors, curve_labels)):
            if ind == 0:

                if FT == 'True': 
                    time_meas_ax = fig.add_subplot(pair_gs[0])
                    freq_meas_ax = fig.add_subplot(pair_gs[1])
                    ax_pair = (time_meas_ax, freq_meas_ax)
                    ax_list.append(ax_pair) 

                elif FT == 'False':
                    time_meas_ax = fig.add_subplot(pair_gs[0])
                    ax_pair = [time_meas_ax]
                    ax_list.append(ax_pair) 

                elif FT == 'Only':
                    freq_meas_ax = fig.add_subplot(pair_gs[0])
                    ax_pair = [freq_meas_ax]
                    ax_list.append(ax_pair) 

             
            meas_vals = measures_dict[meas_task][start_t : end_t : dt]
            meas_times = np.arange(start_t,end_t,dt)

            
            if FT in ('True', 'Only'):
                meas_freqs, meas_amps = pt.make_ft(meas_vals[cut_first::], 1) 
                freq_meas_ax.semilogy(meas_freqs, meas_amps, linewidth=0.1,
                        color=color, label = curve_label) 
                freq_meas_ax.set_ylabel(r'$\mathcal{F}$'+'('+meas_labels[meas_task]+')')
                if meas_ind == nrow-1:
                    freq_meas_ax.set_xlabel('frequency')

                #Nyquist criterion
                high_freq = 1.0/(2.0*dt * (2.0*pi))
                low_freq = 1.0/(dt*len(meas_amps) * (2.0*pi))
                freq_meas_ax.set_xlim([low_freq, high_freq])
                freq_meas_ax.set_ylim([np.mean(meas_amps)/1000., 10.*meas_amps.max()])


            if FT in ('True', 'False'):
                time_meas_ax.plot(meas_times, meas_vals, linewidth=0.1, color=color, label = curve_label)
                time_meas_ax.set_ylabel(meas_labels[meas_task])
                if meas_ind == nrow-1:
                    time_meas_ax.set_xlabel('iterations')

                curr_ymax = meas_vals[cut_first::].max()
                curr_ymin = meas_vals[cut_first::].min()
                if ymin > curr_ymin:
                    ymin = curr_ymin
                if ymax < curr_ymax:
                    ymax = curr_ymax
                time_meas_ax.set_xlim([start_t, end_t])
                time_meas_ax.set_ylim(ymin*(1-1/10), ymax*(1+1/10))
                if meas_task == 'ipr':
                    time_meas_ax.set_ylim(3.0, ymax*(1+1/50))

    flat_ax_list = (a for pair_ax in ax_list for a in pair_ax)
    #plt.setp([a.xaxis.set_ticks(range(0, 0.1, 0.01) for a in flat_ax_list])
    #plt.setp([a.yaxis.set_ticks(range(0, 4, .1)) for a in flat_ax_list])
    no_xlabel_ax = [ax_list[k][j] for j in range(ncol) for k in range(nrow-1)]
    plt.setp([a.get_xticklabels() for a in no_xlabel_ax], visible=False)
    plt.setp([a.grid( 'on' ) for a in flat_ax_list])
    
    
    if FT == 'Only':
        freq_meas_ax.legend(loc='lower right', ncol=3)
    else:
        time_meas_ax.legend(loc='lower right', ncol=3)
    
    plt.suptitle(str(suptitle))

    #time_meas_ax.legend(bbox_to_anchor=(1.05, 0), loc='lower left', borderaxespad=0.)
    #plt.setp([inner_title(a, title) for (a, title) in title_ax], visible=True)

def space_time_comp_plot(res_list, fignum=1):
    plot_tmax = 60
    
    rjt_mat_list = [make_rjt_mat(res)[0:plot_tmax:] for res in res_list]
    x_grid_list  = [local_exp_vals(rjt_mat, ss.ops['X']) 
                    for rjt_mat in rjt_mat_list ]
    y_grid_list  = [local_exp_vals(rjt_mat, ss.ops['Y'])
                    for rjt_mat in rjt_mat_list ]
    z_grid_list  = [local_exp_vals(rjt_mat, ss.ops['Z'])
                    for rjt_mat in rjt_mat_list ]

    grid_axarr=[]
    for j, grid_data_list in enumerate(zip(x_grid_list, y_grid_list, z_grid_list)):

        im, im_axlist = pt.plot_projections(grid_data_list, fignum=fignum,
                ax=231+j, cmap=plt.cm.gray)

        grid_axarr.append(im_axlist)

    grid_axarr=np.asarray(grid_axarr)

    no_xlabel_ax = [grid_axarr[k,j] for j in [0, 1, 2] for k in [0, 1, 2]]
    no_ylabel_ax = [grid_axarr[k,j] for j in [0, 1, 2] for k in [1, 2, 4, 5]]
    title_ax = [(grid_axarr[k,j], tit) for (j, tit) in zip([0, 1, 2],
        [r'$\sigma_x$', r'$\sigma_y$', r'$\sigma_z$']) for k in [0, 1, 2]]

    cax = plt.figure(fignum).add_axes([0.915, 0.117, 0.015, 0.765])
    plt.colorbar(im, cax=cax)

    plt.setp([a.get_xticklabels() for a in no_xlabel_ax], visible=False)
    plt.setp([a.get_yticklabels() for a in no_ylabel_ax], visible=False)
    plt.setp([inner_title(a, title) for (a, title) in title_ax], visible=True)
    plt.figure(fignum).subplots_adjust(hspace = -0.04, wspace=0.06, left=0.05)


def equib_comp_plots(res_list, fignum=1, fmt='--s', label=''):
    start_t = 500
    end_t = 1000
    dt = 1
    
    fig = plt.figure(fignum) 
    meas_task_list = ['ND', 'CC', 'Y', 'ipr']

    measures_dict_list = make_measures_dict_list(res_list)
    params_list = [res['meta']['params'] for res in  res_list]
   
    val_dict = {}
    for meas_task in meas_task_list:
        val_dict[meas_task] = []
        for params, measures_dict in zip(params_list, measures_dict_list):
            center_op = params['center_op'] 
            R = params['R']
            mode = params['mode']
            meas_vals = measures_dict[meas_task][start_t : end_t : dt]
            r_avg, n_equib, val_equib, dval_equib = equib_calc(meas_vals)
            val_dict[meas_task].append((val_equib, dval_equib, center_op, R, mode))

    for ind, meas_task in enumerate(meas_task_list):
        ax = fig.add_subplot(221+ind)
        vals = [val_dict[meas_task][j][0] for j in range(len(val_dict[meas_task]))]
        dvals = [val_dict[meas_task][j][1] for j in range(len(val_dict[meas_task]))]
        xs = np.arange(len(val_dict[meas_task]))

        ax.errorbar(xs, vals , yerr=dvals, fmt=fmt, label=label)
        
        ax.set_xlim([-0.3, 2+0.3])
        ax.set_ylim([0.0, 0.2])
        plt.setp(ax, xticks=[0, 1, 2], xticklabels=['H', 'HT', 'HXT'])
        plt.setp(ax.grid( 'on' ) )

        if ind != 3 and ind != 2:
            plt.setp(inner_title(ax, r'$'+str(meas_task)+'$', l=0.02, u=0.93), visible=True)

        if ind == 2:
            plt.setp(inner_title(ax, r'$'+str(meas_task)+'$', l=0.045, u=0.93), visible=True)

        if ind == 3:
            ax.set_ylim([3.5, 4.05])
            plt.setp(inner_title(ax, r'$IPT$', l=0.02, u=0.93), visible=True)
            #ax.yaxis.tick_right()

        #if ind == 1:
            #plt.setp(ax.get_yticklabels(), visible=False)

        if ind == 0 or ind == 1:
            plt.setp(ax.get_xticklabels(), visible=False)
        plt.subplots_adjust(wspace=0.2, hspace=0.1)

def R_meas_comp_plots(res_list, fignum=1, label=''):
    start_t = 0
    end_t = 1000
    dt = 1

    meas_times = range(start_t, end_t)
    fig = plt.figure(fignum) 
    meas_tasks = ['CC','ipr']

    measures_dict_list = make_measures_dict_list(res_list)
    params_list = [res['meta']['params'] for res in  res_list]

    for meas_ind, meas_task in enumerate(meas_tasks):
        for ind, measures_dict in enumerate(measures_dict_list):
            meas_vals = measures_dict[meas_task][start_t : end_t : dt]
            meas_times = np.arange(start_t,end_t,dt)

            ax = fig.add_subplot(221+ind)

            ax.plot(meas_times, meas_vals, linewidth=0.1)
            ''' 
            ax.set_xlim([-0.3, 2+0.3])
            ax.set_ylim([0.0, 0.2])
            plt.setp(ax, xticks=[0, 1, 2], xticklabels=['H', 'HT', 'HXT'])
            plt.setp(ax.grid( 'on' ) )

            if ind != 3 and ind != 2:
                plt.setp(inner_title(ax, r'$'+str(meas_task)+'$', l=0.02, u=0.93), visible=True)

            if ind == 2:
                plt.setp(inner_title(ax, r'$'+str(meas_task)+'$', l=0.045, u=0.93), visible=True)

            if ind == 3:
                ax.set_ylim([3.5, 4.05])
                plt.setp(inner_title(ax, r'$IPT$', l=0.02, u=0.93), visible=True)
                #ax.yaxis.tick_right()

            #if ind == 1:
                #plt.setp(ax.get_yticklabels(), visible=False)

            if ind == 0 or ind == 1:
                plt.setp(ax.get_xticklabels(), visible=False)
            plt.subplots_adjust(wspace=0.2, hspace=0.1)
            '''
def make_R_meas_comp_plot(name):
    plt.clf
    plt.close('all')
    fixed_params_dict =  OrderedDict( [ 
                                        ('output_name', ['sweep_block4']),
                                        ('IC', ['s18']),
                                        ('L', [19]),
                                        ('tmax', [1000]),
                                        ('mode', ['sweep', 'block']),
                                        ] )

    var_params_dict = OrderedDict( [ 
                                    ('R', [150, 102]),
                                    ('center_op', [['H'], ['H','T'], ['H','X','T']]),
                                     ] )

    params_list_list = make_params_list_list(fixed_params_dict, var_params_dict)
    fignum=10
    
    for ind, params_list in enumerate(params_list_list):
        res_list = [io.read_results(params) for params in params_list]
        R_meas_comp_plots(res_list, fignum)

    output_name = params_list[0]['output_name']
    plt.savefig(io.file_name(output_name, 'plots/comps', name, '.pdf'))

def make_measures_comp_plot(name, FT='True', meas_tasks = ['ND', 'CC', 'Y', 'ipr']):
    plt.clf
    plt.close('all')
    fixed_params_dict =  OrderedDict( [ 
                                        ('output_name', ['sweep_block4']),
                                        ('IC', ['s18']),
                                        ('L', [19]),
                                        ('tmax', [1000]),
                                        ('R', [150, 102]),
                                        ('mode', ['sweep', 'block']),
                                        ] )

    var_params_dict = OrderedDict( [ 
                                     ('center_op', [['H'], ['H','T'], ['H','X','T']])
                                     ] )

    params_list_list = make_params_list_list(fixed_params_dict, var_params_dict)

    fignum=3
    for plot_num, params_list in enumerate(params_list_list):
            suptitle = params_list[0]['mode']+' '+str(params_list[0]['R'])
            res_list = [io.read_results(params) for params in params_list]

            measures_comp_plot(res_list, var_params_dict, fignum=fignum,
                    suptitle = suptitle, FT=FT, meas_tasks=meas_tasks)

            fignum=fignum + 1

    output_name = params_list[0]['output_name']
    io.multipage(io.file_name(output_name, 'plots/comps', name, '.pdf'))    


def make_equib_comp_plot(name):
    plt.clf
    plt.close('all')
    fixed_params_dict =  OrderedDict( [ 
                                        ('output_name', ['sweep_block4']),
                                        ('IC', ['s18']),
                                        ('L', [19]),
                                        ('tmax', [1000]),
                                        ('R', [150, 102]),
                                        ('mode', ['sweep', 'block'])
                                        ] )

    var_params_dict = OrderedDict( [ 
                                     ('center_op', [['H'], ['H','T'], ['H','X','T']]),
                                     ] )
    params_list_list = make_params_list_list(fixed_params_dict, var_params_dict)
    fignum=10
    label_list = ['sweep 150', 'block 150', 'sweep 102', 'block 102']
    for ind, params_list in enumerate(params_list_list):
        fmt_list = ['-^k','--^k','-sk','--sk']
        res_list = [io.read_results(params) for params in params_list]
        equib_comp_plots(res_list, fignum, fmt=fmt_list[ind],
                label=label_list[ind])

    plt.figure(fignum).subplots_adjust(top=.97)
    plt.legend(bbox_to_anchor=(0.0, 0.0), loc='lower left', borderaxespad=0.,
            ncol=2)
    output_name = params_list[0]['output_name']
    plt.savefig(io.file_name(output_name, 'plots/comps', name, '.pdf'))




if __name__ == '__main__':
    #make_space_time_comp_plot()
    #make_measures_comp_plot('all_meas_and_FT', FT='True', meas_tasks=['ND','CC','Y','ipr'])
    make_measures_comp_plot('all_FT', FT='Only', meas_tasks=['ND','CC','Y','ipr'])
    #make_equib_comp_plot('equib')

    #make_R_meas_comp_plot('102_150102_150_CC_IPT')

    
    
    #name = str(mode)+'_measures_R'+str(R)+'_L19'
    ''' 
    for i in range(3):
        data_fname = io.file_name(output_name, 'data/comps', name, '.'+task, V=0)
        with open(data_fname, 'w') as outfile:
            writer = csv.writer(outfile)
            writer.writerows(zip(['H'], ['HT'], ['HXT']))
            writer.writerows(zip(d0, d1, d2))
    '''
            






    '''
    import run_ham as run 
    Qparams_list = run.params_list
    Qparams = Qparams_list[0]
    L = Qparams['L']
    tmax = Qparams['tmax']
    Qres = io.read_results(Qparams, typ='Q')
    Zcorr = Qres['nz'] 
    corr, dist = correlator_calc(Zcorr)
    for t in range(tmax-1, tmax):
        corr0j = [corr[t,j] for j in range(L)]
        plt.plot(dist, corr0j, 'x')
    plt.show()

    import run_sweep as run 
    Cparams_list = run.Cparams_list
    Qparams_list = run.Qparams_list
    output_name, R, IC, L, tmax = Cparams_list[0]
     
    LT = make_long_time_avgs(Cparams_list, Qparams_list)
    long_time_avg_plots(LT)
    
    io.multipage(io.file_name(output_name, \
        'plots','LT_avg_'+'R'+str(R)+'_0011_seperation', '.pdf')) 
    ''' 


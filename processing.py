#!/usr/bin/python

from cmath  import sqrt
from math   import fabs, sin, pi
from collections import OrderedDict
import copy
import csv
import numpy as np
import information as im
import fio as io
import networkmeasures as nm
import measures as ms
import matplotlib        as mpl
import matplotlib.pyplot as plt
import plotting as pt


# default plot font to bold and size 16
# -------------------------------------
font = {'weight':'bold', 'size':16}
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


# compute running average through time of a time series
# -----------------------------------------------------
def running_average(time_series, tol=0.1):
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

def make_comp_title(params1, params2):
    params = {}
    labels = {}
    title = ''
    measures = {'ND':'network density', 'Y':'disparity',
                'CC':'clustering coefficient', 'ipr':'inverse participation ratio'} 
    
    for key in params1.keys():
        if params1[key] == params2[key]:
            title = title + params1[key]
        else:
            labels[key] = (params1[key], params2[key])
         



if __name__ == '__main__':

    for mode in ['sweep', 'block']:
        for R in [150, 102]:
            params_list = [make_params('sweep_block4', mode, center_op, R, 's18', 19, 1000)
                        for center_op in [['H'],['H','T'],['H','X','T']]]

            name_list = [io.sim_name(params) for params in params_list]
            
            output_name = params_list[0]['output_name']

            task_name = {'ND':' network density', 'CC': ' clustering coefficient',
                    'Y':' disparity', 'ipr': ' inverse participation ratio' }

            res_list = [io.read_results(params) for params in params_list]

            mi_nets_list = [res['mi'] for res in res_list]

            ipr_list = [res['ipr'] for res in res_list]

            nm_time_series_list = [measure_networks(mi_nets, typ='avg', cut_first=0)
                    for mi_nets in mi_nets_list]

            fignum = 1
            for task in ['ND','CC','Y', 'ipr']:
                if task != 'ipr':
                    d0 = nm_time_series_list[0][task] 
                    d1 = nm_time_series_list[1][task] 
                    d2 = nm_time_series_list[2][task] 

                elif task == 'ipr':
                    d0 = ipr_list[0]
                    d1 = ipr_list[1]
                    d2 = ipr_list[2]

                d0_freqs, d0_amps = pt.make_ft(d0, 1) 
                d1_freqs, d1_amps = pt.make_ft(d1, 1) 
                d2_freqs, d2_amps = pt.make_ft(d2, 1) 
                
                max_index = 100 + np.argmax(d2_amps[100::])
                max_amp = d2_amps[max_index]
                max_freq = d2_freqs[max_index]

                print(max_freq)

                pt.plot_time_series(d0, '', cut_first=0,
                        fignum=fignum, label='H', color='B',
                        marker='o')
                
                pt.plot_time_series(d1, '', cut_first=0,
                        fignum=fignum, label='HT', color='G',
                        marker='^')

                title = mode+' update rule '+str(R) + task_name[task]
                pt.plot_time_series(d2, title, 
                        cut_first=0, fignum=fignum, label='HXT', color='R',
                        marker='s', loc='upper right')

                pt.plot_ft(d0_freqs, d0_amps, 1, '', fignum = fignum+1, color='B')
                pt.plot_ft(d1_freqs, d1_amps, 1, '', fignum = fignum+1, color='G')
                pt.plot_ft(d2_freqs, d2_amps, 1, task_name[task]+' spectrum', fignum = fignum+1, color='R')

                #plt.scatter(max_freq, max_amp, color='B')
                fignum = fignum + 10


                name = str(mode)+'_measures_R'+str(R)+'_L19'
                for i in range(3):
                    data_fname = io.file_name(output_name, 'data/comps', name, '.'+task, V=0)
                    with open(data_fname, 'w') as outfile:
                        writer = csv.writer(outfile)
                        writer.writerows(zip(['H'], ['HT'], ['HXT']))
                        writer.writerows(zip(d0, d1, d2))

            io.multipage(io.file_name(output_name, 'plots/comps', name, '.pdf'))    
            

with open('some.csv', 'wb') as f:

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


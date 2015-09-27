#!/usr/bin/python

from cmath  import sqrt
import copy
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
    mats = np.asarray(mats)
    return np.array([mat.diagonal() for mat in mats])

def get_offdiag_mats(mats):
    L = len(mats[0])
    mats_out = copy.deepcopy(mats)
    for t, mat in enumerate(mats_out): 
        mats_out[np.arange(L), np.arange(L)] = 0.0
        mats_out[t] = mat
    return mats_out

def spatialnetworksQ(results, typ):
    n_mats = results[typ]
    corr, loc = get_offdiag_mats(n_mats), get_diag_vecs(n_mats)
    return im.spatialnetworksQ(corr, loc)

def measure_networks(nets, tasks=['Y','CC'], typ='avg'):
    measures = {} 
    for task in tasks:
        measures[task] = np.asarray([ms.NMcalc(net, typ=typ,
                                    tasks=tasks)[task] for net in nets])
    return measures

def make_net_dict(results, net_types=['nz', 'nx', 'mi']):
    net_dict = {}
    for net_typ in net_types:
        if net_typ == 'mi':
            network = results['mi']
        else: 
            network = spatialnetworksQ(results, net_typ)
        net_dict[net_typ] = network
    return net_dict


def running_average(data):
    tot = 0
    r_avg = [0]*len(data)
    for n, d in enumerate(data):
        tot += d
        r_avg[n] = tot/(n+1)
    return r_avg


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

if __name__ == '__main__':
    
    import run_sweep as run 
    Cparams_list = run.Cparams_list
    Qparams_list = run.Qparams_list
    output_name, R, IC, L, tmax = Cparams_list[0]
     
    LT = make_long_time_avgs(Cparams_list, Qparams_list)
    long_time_avg_plots(LT)
    
    io.multipage(io.file_name(output_name, \
        'plots','LT_avg_'+'R'+str(R)+'_0011_seperation', '.pdf')) 
    


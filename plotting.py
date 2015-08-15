#!/usr/bin/python

import numpy             as np
import processing        as pp
import matplotlib        as mpl
import matplotlib.pyplot as plt
import fio               as io


# default plot font to bold and size 16
# -------------------------------------
font = {'family':'normal', 'weight':'bold', 'size':16}
mpl.rc('font',**font)


# Plotting utilities
# ==================

# plot space time grids
# ---------------------
def board_plot(board_res, cmap, norm):
    vmin, vmax = 0.0, 1.0
    if np.max(board_res) > 1.0:
        vmax = np.max(board_res)
    if np.min(board_res) < 0.0:
        vmin = np.min(board_res)
    plt.imshow(board_res,
                    vmin = vmin,
                    vmax = vmax,
                    cmap = cmap,
                    norm = norm,
                    interpolation = 'none',
                    aspect = 'auto',
                    rasterized = True)

def plot_time_series(data, label, title, fignum=1, ax=111):
    fig = plt.figure(fignum)
    fig.add_subplot(ax)
    plt.plot(range(len(data)), data, label=label)
    plt.title(title)
    plt.legend(loc='lower right')
    plt.tight_layout() 

# plot probability, discretized probability, and single-site entropy
# ------------------------------------------------------------------
def board_plots(results, suptitle, fignum=1, axs=(131, 132, 133)):
    #xtick_locs = range(0,L,4)
    #xtick_lbls = xtick_locs
     
    n_res      = pp.get_diag_vecs(results['nz'])
    sj_res     = pp.get_diag_vecs(results['sr'])
    disc_n_res = [ [0 if p < 0.5 else 1 for p in p_vec] for p_vec in n_res]
    
    disc_cmap = mpl.colors.ListedColormap([plt.cm.jet(0.), plt.cm.jet(1.)])
    bounds = [0,0.5,1]
    disc_norm = mpl.colors.BoundaryNorm(bounds, disc_cmap.N)
    
    titles = [ r'$ \langle n_j \rangle $', \
               r'$ \lfloor \langle n_j \rangle + 0.5 \rfloor $', \
               r'$ s_j $' ]
    results = [n_res, disc_n_res, sj_res]
    mycmaps = [plt.cm.jet, disc_cmap, plt.cm.jet]
    norms = [None, disc_norm, None]
    fig = plt.figure(fignum)
    for title, res, cmap, norm, ax in zip(titles, results, mycmaps, norms, axs):
        fig.add_subplot(ax)
        board_plot(res, cmap, norm)
        plt.title(title)
        plt.xlabel('site number')
     #   plt.xticks(xtick_locs, xtick_lbls)
        plt.ylabel('time')
        plt.colorbar()
    plt.suptitle(suptitle) 
    plt.tight_layout() 


    

# plot entropy of all bi-partite cuts
# -----------------------------------
def cut_entropy_plots(results, L, suptitle, fignum=1, axs=(121, 122)):

#    xtick_locs = range(0,L-1,2)
#    xtick_lbls = xtick_locs
     
    EC_res         = results['ec']
    center_cut_res = (np.array(EC_res).transpose()[int(L/2)]).transpose()

    fig =  plt.figure(fignum)

    fig.add_subplot(axs[0]) 
    board_plot(EC_res, plt.cm.jet, None)
    plt.title('Entropy of cut')
    plt.xlabel('cut number')
#    plt.xticks(xtick_locs, xtick_lbls)
    plt.ylabel('time')
    
    fig.add_subplot(axs[1])
    plot_time_series(center_cut_res,'', 'Entropy of center cut', fignum=fignum,
            ax = axs[1])
    plt.xlabel('time')
    plt.ylabel('Entropy')
    plt.suptitle(suptitle) 
    plt.tight_layout() 

# plot network measures
# ---------------------
def network_plots(results,  types = ['nz', 'nx', 'mi'], tasks
        = ['ND', 'CC','Y', 'IHL'], fignum=10):
    net_measures_dict = pp.make_net_measures_dict(results, tasks) 
    fignum = fignum
    for typ in types:
        data = net_measures_dict[typ]
        title = 'Network Measures on ' + typ 
        for task in tasks:
            dat = data[task]
            label = task + ' ' + typ
            plot_time_series(dat, label, title, fignum=fignum)
        fignum+=1
        plt.tight_layout

# plot eigenvector centrality
# ---------------------------
def evec_centrality_plot(results, params, fignum=1, ax=111):
    output_name, R, IC, L, tmax = params
    fig = plt.figure(fignum) 
    fig.add_subplot(ax)
    board_plot(results['EV'], plt.cm.jet, None)
    plt.title('R ' + str(R) + 'Eigenvector Centrality')
    plt.xlabel('site number')
    plt.ylabel('time')
    plt.tight_layout()

# call for time series plots
# --------------------------
def plot_main(params, name=None):
    output_name, R, IC, L, tmax = params
    if name is None:
        name = io.sim_name(R, IC, L, tmax)
    else:
        name = name
    print('Importing results...')
    results = io.read_results(params)
    board_plots(results, 'R ' + str(R), fignum=1)
    cut_entropy_plots(results, L, 'R ' + str(R), fignum=2) 
    #evec_centrality_plot(results, params, fignum=3) 
    network_plots(results) 
    io.multipage(io.file_name(output_name, 'plots', 'Q'+name, '.pdf'))    
    return


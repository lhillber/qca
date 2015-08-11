#!/usr/bin/python

import numpy             as np
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

# plot probability, discretized probability, and single-site entropy
# ------------------------------------------------------------------
def board_plots(results, params, fignum=1, axs=(131, 132, 133)):
    output_name, R, IC, L, tmax = params 
    xtick_locs = range(0,L,4)
    xtick_lbls = xtick_locs
     
    prob_res      = results['prob'] 
    sse_res       = results['sse'] 
    disc_prob_res = [ [0 if p < 0.5 else 1 for p in p_vec] for p_vec in prob_res]
    
    disc_cmap = mpl.colors.ListedColormap([plt.cm.jet(0.), plt.cm.jet(1.)])
    bounds = [0,0.5,1]
    disc_norm = mpl.colors.BoundaryNorm(bounds, disc_cmap.N)
    
    titles = [' probability', ' projected prob.', ' s.s. entropy']
    results = [prob_res, disc_prob_res, sse_res]
    mycmaps = [plt.cm.jet, disc_cmap, plt.cm.jet]
    norms = [None, disc_norm, None]
    fig = plt.figure(fignum)
    for title, res, cmap, norm, ax in zip(titles, results, mycmaps, norms, axs):
        fig.add_subplot(ax)
        board_plot(res, cmap, norm)
        plt.title(title)
        plt.xlabel('site number')
        plt.xticks(xtick_locs, xtick_lbls)
        plt.ylabel('time')
        plt.colorbar()
    plt.suptitle('R ' + str(R)) 
    plt.tight_layout() 

# plot entropy of all bi-partite cuts
# -----------------------------------
def entropy_of_cut_plots(res, params, fignum=1, axs=(121, 122)):
    output_name, R, IC, L, tmax = params 

    xtick_locs = range(0,L-1,2)
    xtick_lbls = xtick_locs
     
    EC_res         = res['ec']
    center_cut_res = (np.array(EC_res).transpose()[int(L/2)]).transpose()

    fig =  plt.figure(fignum)

    fig.add_subplot(axs[0]) 
    board_plot(EC_res, plt.cm.jet, None)
    plt.title('Entropy of cut')
    plt.xlabel('cut number')
    plt.xticks(xtick_locs, xtick_lbls)
    plt.ylabel('time')
    
    fig.add_subplot(axs[1]) 
    plt.plot(range(tmax), center_cut_res)
    plt.title('Entropy of center cut')
    plt.xlabel('time')
    plt.ylabel('Entropy')
    plt.suptitle('R ' + str(R)) 
    plt.tight_layout() 

# plot network measures
# ---------------------
def nm_plot(res, params, fignum=1, ax=111):
    output_name, R, IC, L, tmax = params
    nm_keys = ['ND', 'CC', 'Y', 'IHL'] 

    fig = plt.figure(fignum)
    fig.add_subplot(ax)
    for key in nm_keys: 
        measure = res[key]
        plt.plot(range(tmax), measure, label = key)
    plt.title('R '+str (R) + ' Network measures')
    plt.legend()
    plt.tight_layout() 

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
def plot_time_series(params, name=None):
    output_name, R, IC, L, tmax = params
    if name is None:
        name = io.sim_name(R, IC, L, tmax)
    else:
        name = name
    print('Importing results...')
    results = io.read_results(params)
    board_plots (results, params)
    nm_plot     (results, params, fignum=2) 
    evec_centrality_plot(results, params, fignum=3) 
    entropy_of_cut_plots(results, params, fignum=4) 
    io.multipage(io.file_name(output_name, 'plots', name, '.pdf'))    
    return


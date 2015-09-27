#!/usr/bin/python

import numpy             as np
import processing        as pp
import matplotlib        as mpl
import matplotlib.pyplot as plt
import fio               as io


# default plot font to bold and size 16
# -------------------------------------
font = {'family':'normal', 'weight':'bold', 'size':14}
mpl.rc('font',**font)


# Plotting utilities
# ==================

# plot time series
# ---------------0
def plot_time_series(data, title, label='', loc='lower right', fignum=1, ax=111):
    fig = plt.figure(fignum)
    fig.add_subplot(ax)
    plt.plot(range(len(data)), data, label=label)
    plt.title(title)
    plt.legend(loc=loc)
    plt.tight_layout() 

# plot space time grids
# ---------------------
def plot_spacetime_grid(data, title, cmap=plt.cm.jet, norm=None, fignum=1, ax=111):
    vmin, vmax = 0.0, 1.0
    if np.max(data) > 1.0:
        vmax = np.max(data)
    if np.min(data) < 0.0:
        vmin = np.min(data)
    
    fig = plt.figure(fignum)
    fig.add_subplot(ax)
    plt.imshow(data,
                    vmin = vmin,
                    vmax = vmax,
                    cmap = cmap,
                    norm = norm,
                    interpolation = 'none',
                    aspect = 'auto',
                    rasterized = True)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout() 

# plot probability, discretized probability, and single-site entropy
# ------------------------------------------------------------------
def board_plots(results, fignum=1, axs=(131, 132, 133)):
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
        kwargs = {'cmap': cmap, 'norm': norm, 'fignum' : fignum, 'ax' : ax} 
        fig.add_subplot(ax)
        plot_spacetime_grid(res, title, **kwargs)
        plt.title(title)
        plt.xlabel('Site number')
     #   plt.xticks(xtick_locs, xtick_lbls)
        plt.ylabel('Time')
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
    plot_spacetime_grid(EC_res, 'Entropy of cut', 
                         cmap=plt.cm.jet, norm=None, fignum=fignum, ax=axs[0])
    plt.xlabel('Cut number')
#    plt.xticks(xtick_locs, xtick_lbls)
    plt.ylabel('Time')
    
    fig.add_subplot(axs[1])
    plot_time_series(center_cut_res, 'Entropy of center cut', 
                     fignum=fignum, ax = axs[1])
    plt.xlabel('time')
    plt.ylabel('Entropy')
    
    plt.suptitle(suptitle) 
    plt.tight_layout() 

# plot network measures
# ---------------------


def st_nm_plots(st_data, title, tasks = ['EV', 'CC', 'Y'], fignum=1):
    ax = 131 
    for i, task in enumerate(tasks):
        dat = st_data[task]
        label = task
        plot_spacetime_grid(dat, title, fignum=fignum, ax=ax)
        plt.ylabel('Time')
        plt.xlabel('Site number')
        ax += 1


def avg_nm_plots(avg_data, title, tasks = ['ND', 'CC','Y'], fignum=1):
    for i, task in enumerate(tasks):
        dat = avg_data[task]
        label = task
        plot_time_series(dat, title, label=label, fignum=fignum)

def st_nm_plots(st_data, title, tasks = ['EV', 'CC', 'Y'], fignum=1):
    ax = 131 
    for i, task in enumerate(tasks):
        title = task
        dat = st_data[task]
        label = task
        plot_spacetime_grid(dat, title, fignum=fignum, ax=ax)
        plt.ylabel('Time')
        plt.xlabel('Site number')
        ax += 1

# call for time series plots
# --------------------------
def plot_main(params, 
        st_tasks =['EV', 'CC', 'Y' ], avg_tasks=['ND', 'CC', 'Y'],
        net_types=['nz', 'mi'], name=None):
    
    R    = params[ 'R'   ]
    L    = params[ 'L'   ]
    tmax = params[ 'tmax']
    output_name = params['output_name']
    
    if name is None:
        name = io.sim_name(params)
    else:
        name = name
    
    print('Importing results...')
    results = io.read_results(params, typ='Q')
    
    net_dict = pp.make_net_dict(results, net_types=net_types)
    fignum = 0
    board_plots(results, fignum=fignum)
    for net_typ in net_types:
         title = net_typ + ' network measures'
         fignum += 10
         nets = net_dict[net_typ]
         
         st_net_measures  = pp.measure_networks(nets, 
                                                tasks=st_tasks , typ='st')
         avg_net_measures  = pp.measure_networks(nets, 
                                                tasks=avg_tasks, typ='avg')
         
         avg_nm_plots( avg_net_measures, title, tasks=avg_tasks, fignum=fignum  )
         st_nm_plots ( st_net_measures,  title,  tasks=st_tasks , fignum=fignum+1 )
    
    plot_time_series(results['st'], r'$S^t_{topo}$', fignum=fignum+2)
    
    cut_entropy_plots(results, L, 'R ' + str(R), fignum=fignum+3) 
    
    io.multipage(io.file_name(output_name, 'plots', 'Q'+name, '.pdf'))    
    return


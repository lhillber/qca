import numpy as np
import matplotlib.pyplot as plt
from matrix import ops
import matplotlib.gridspec as gridspec
import os
from scipy.optimize import curve_fit
from matplotlib import rc
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from figures import letters

fontstyle = {'pdf.fonttype': 42,
'text.usetex': True,
'text.latex.preamble': '\\usepackage{amsfonts}',
'font.family': 'serif',
'axes.labelsize': 9,
'font.size': 9,
'legend.fontsize': 9,
'xtick.labelsize': 9,
'ytick.labelsize': 9}
plt.rcParams.update(fontstyle)
rc("mathtext", default='regular')

# where is the raw data wrt this file?
# also where processed data will be stored
DATA_DER = "data/qeb_decay"


def main():
    # system paremters, correlated with file name
    # error values (percent)
    errs = [0.4, 0.54, 0.728, 0.983, 1.33, 1.79, 2.41, 3.26, 4.39, 5.93, 8.0]
    L = 17  # true system size
    dt = 0.1  # time step
    load = True  # load processed data? (generates/saves if not found)
    show = False  # show fits while processing?

    # processed data save location
    proc_fname = f"{L+4}-site_admix-processed.npy"
    proc_fname = os.path.join(DATA_DER, proc_fname)

    # plot save location
    plot_fname = f"{L+4}-site_admix_lifetime_V5.pdf"
    # generate or load processed data
    popts, perrs, errs, ts, ys_list = run(
        load, errs, L, dt, proc_fname, show)
    # make a plot
    plot(popts, perrs, errs, ts, L, ys_list, plot_fname)


# isolate all the ugly plotting code
def plot(popts, perrs, errs, ts, L, ys_list, plot_fname):
    # which curves to plot?
    plot_errs = [4.39, 5.93, 8.0]
    # shift each curve by
    shift = 0.1
    # calculate time step from time axis
    dt = ts[1] - ts[0]

    # set up the figure, one section for the curves/fits
    fig = plt.figure(figsize=(3.375, 5.5))
    gs1 = gridspec.GridSpec(1, 1)
    gs1.update(left=0.2, right=0.9, bottom=0.65, top=0.95)
    tax = fig.add_subplot(gs1[0, 0])

    # one section for the spacetime grids
    gs2 = gridspec.GridSpec(1, 3)
    gs2.update(left=0.15, right=0.85, bottom=0.1, top=0.53, wspace=0.1)
    gaxs = [fig.add_subplot(gs2[0, i]) for i in range(3)]

    # for each data set (in descending order)
    for i, err in enumerate(plot_errs[::-1]):
        # select the grid axis
        gax = gaxs[i]
        # select the data
        idx = errs.index(err)
        ys = ys_list[idx]
        popt = popts[idx]
        tau = popt[1]
        print(f"error {err} has lifetime {tau}")

        # line plots for panel A
        zorder = 100 - 5*i
        timax = int(400/dt)
        tax.plot(ts[10:timax], ys[10:timax]+i*shift, zorder=zorder,
                 label=(r"$\varepsilon=%03.2f$" % round(err, 2))+r"$\%$")
        tax.plot(ts[:timax], lifetime_func(ts[:timax], *popt)+i*shift, c="k", zorder=zorder+1)
        handles, labels = tax.get_legend_handles_labels()
        tax.legend(handles[::-1], labels[::-1], loc="lower left",
                   frameon=False, handlelength=0.5, handletextpad=0.25,
                   bbox_to_anchor=(0.55, 0.5))
        tax.text(0.1, 0.9, letters[0], transform=tax.transAxes)
        #tax.set_yscale("log")
        taxyticks = [0.4, 0.6, 0.8, 1.0]
        tax.set_yticks(taxyticks)
        tax.set_yticklabels(taxyticks)
        tax.set_xlabel(r"Time, $t$")
        tax.set_xticks([0, 100, 200, 300, 400])
        tax.set_ylabel(r"$\big \langle \hat P^{(1)}_{\lfloor L/2 \rfloor-1}  \big \rangle$")
        # grid plots for panels B, C, D
        grid = make_grid(L, err)
        grid  = (1 - grid) / 2
        im = gax.imshow(grid[::int(1/dt)][:L*4+1],
            origin="lower",
            interpolation="none",
            cmap="inferno",
            vmin=0,
            vmax=1)
        gax.axhline(tau, c=f"C{i}", lw=4)
        gax.axhline(tau, c="k", lw=2)
        gax.set_xticks([0, (L-1)//2, (L-1)])
        gax.set_xticklabels([])
        gax.set_yticks([i*(L-1) for i in range(5)])
        gax.set_yticklabels([])
        gax.text(0.1, 0.93, letters[i+1],
                 transform=gax.transAxes, color="w")
        if i == 0:
            gax.set_xticklabels([0, (L-1)//2, (L-1)])
            gax.set_yticklabels(i*(L-1) for i in range(5))
            gax.set_xlabel(r"Site, $j$")
            gax.set_ylabel("Time, $t$")
        if i == 1:
            box = gax.get_position()
            box.x0 = box.x0 - 0.01
            box.x1 = box.x1 - 0.01
            gax.set_position(box)
        if i == 2:
            divider = make_axes_locatable(gax)
            cax = divider.append_axes("right", size="15%", pad=0.075)
            fig.colorbar(im, cax=cax, ticks=[0, 1])
            cax.text(1.2, 0.5, r"$\big \langle \hat P^{(1)}_j  \big \rangle$",
                     transform=cax.transAxes)
    plt.savefig(plot_fname)
    print("figure saved to")
    print(plot_fname)


# raw data file name
def make_data_fname(L, err):
    fname = f"{L+4}-site_rho_{err}-err.npy"
    return os.path.join(DATA_DER, fname)

# fitting function
def lifetime_func(x, A, tau, B):
    return A*np.exp(-x / tau) + B


def make_grid(L, err):
    fname = make_data_fname(L, err)
    # load single site density matrix data: size = (T, L, 2, 2)
    rhojs = np.load(fname)
    # calculate <Z> grid
    grid = np.array([get_expectation(rhoj, ops["Z"]) for rhoj in rhojs])
    # clip boundary sites
    grid = grid[:, 2:-2]
    return grid


def process(errs, L, dt, proc_fname, show):
    popts = []
    perrs = []
    ys_list = []
    for err in errs:
        grid = make_grid(L, err)
        # extract one-from center time series
        # and rescale: <n> = (1 - <Z>)/2
        ys = (1 - grid[:, L//2 + 1]) / 2
        # construct time axis
        ts = dt * np.arange(len(ys))
        # fit life time with a good guess,
        # comes from iterating analysis sequence
        # popt = [A, tau, B] = [scale, lifetime, offset]
        tau0 = 2.55 * err**(-1.25)
        popt, pcov = curve_fit(lifetime_func, ts, ys,
                               p0=[0.5, tau0, 0.5],
                               bounds=((0, 0, 0), (np.inf, np.inf, np.inf)))
        # estimate parameter errorbars
        perr = np.sqrt(np.diag(pcov))
        # collect parameters and their errors
        popts.append(popt)
        perrs.append(perr)
        # collect processed data
        ys_list.append(ys)

        # optionally inspect on the fly
        if show:
            plt.plot(ts, ys)
            plt.plot(ts, lifetime_func(ts, *popt))
            plt.show()

    # multi dim. lists to numpy arrays
    popts = np.array(popts)  # size = (len(errs), 3)
    perrs = np.array(perrs)  # size = (len(errs), 3)
    ys_list = np.array(ys_list)  # size = (len(errs), T)

    # put all the good stuff in a dict
    summary = {"popts": popts,
               "perrs": perrs,
               "errs": errs,
               "ts": ts,
               "ys_list": ys_list}
    # save out and return the good stuff
    np.save(proc_fname, summary)
    print("Processed data saved to")
    print(proc_fname)
    return summary


def run(load, errs, L, dt, proc_fname, show):
    if load:
        try:
            summary = np.load(proc_fname, allow_pickle=True).item()
            print("Processed data loaded from")
            print(proc_fname)
        except FileNotFoundError:
            print("Processed data not found, generating now...")
            summary = process(errs, L, dt, proc_fname, show)
    else:
        print("Processing data now...")
        summary = process(errs, L, dt, proc_fname, show)

    keys = ["popts", "perrs", "errs", "ts", "ys_list"]
    popts, perrs, errs, ts, ys_list = [summary[k] for k in keys]
    return popts, perrs, errs, ts, ys_list

# methods taken from my measures.py scipt
# pasted here for imporved portability
def expectation(state, A):
    if len(state.shape) == 2:
        exp_val = np.real(np.trace(state.dot(A)))
    else:
        if len(state.shape) == 1:
            exp_val = np.real(np.conjugate(state).dot(A.dot(state)))
        else:
            raise ValueError("Input state not understood")
    return exp_val

def get_expectation(rhos, A):
    return np.asarray([expectation(rho, A) for rho in rhos])



if __name__ == "__main__":
    main()

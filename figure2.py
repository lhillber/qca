import numpy as np
import matplotlib.pyplot as plt
import os
from figures import names, letters
from matrix import ops
from measures import get_expectation, get_entropy
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.colorbar import Colorbar
from qca import QCA
import matplotlib as mpl
from matplotlib import rc

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


if __name__ == "__main__":
    # figure rows and their color map
    measures = ["exp_Z", "s_2"]
    cmaps = ["inferno_r", "inferno"]

    # setup plots
    plot_fname = "figures/figure2/figure2_V13.pdf"
    fig = plt.figure(figsize=(3.375 * 2, 6.4))

    # QCA specifications
    Skey = ["3.6", "3.14", "3.13", "5.4", "5.26"]
    IC = "c3_f1"
    L = 19
    BC = "1-00"
    T = (L - 1) * 2 + 2  # plot ylim
    T2 = (L - 1) * 1 + 2  # plot ylim

    # panel labels and colors
    letts1 = [
        letters[0],
        letters[2],
        letters[4],
        letters[6],
        letters[8],
    ]
    letts2 = [
        letters[1],
        letters[3],
        letters[5],
        letters[7],
        letters[9],
    ]
    clett1 = ["w", "w", "w", "w", "w"]
    clett2 = ["k", "k", "k", "w", "k"]
    letts = [letts1, letts2]
    cletts = [clett1, clett2]

    # loop for plot rows
    for row, (meas, letti, cli) in enumerate(zip(measures, letts, cletts)):

        # image grid for each row of plots
        grid = ImageGrid(
            fig,
            int("21" + str(1 + row)),
            nrows_ncols=(2, len(Skey)),
            direction="row",
            axes_pad=[0.25, 0.04],
            cbar_mode="single",
            cbar_location="right",
            cbar_size="10%",
            cbar_pad=0.2,
        )

        # loop for plot columns
        for col, (S, lett, cl) in enumerate(zip(Skey, letti, cli)):
            N, S = map(int, S.split("."))  # N=3 vs 5 site, S=rule number
            # T type
            if N == 3:
                # load qca data
                Q = QCA(
                    dict(
                        L=L,
                        T=1000,
                        dt=1,
                        R=S,
                        r=1,
                        V="H",
                        IC=IC,
                        BC=BC,
                        E=0,
                        N=1,
                        totalistic=False,
                        hamiltonian=False,
                        trotter=True,
                        symmetric=False,
                    )
                )
                # Q.check_repo(test=True)
                d = Q.get_measure(meas)
                Q.close()

                # magnetization vs entropy colorbar labels
                if meas[0] == "e":
                    ticks = [-1, 1]
                    ticklabels = [
                        r"$\downarrow$",
                        r"$\uparrow$",
                    ]
                    xaxticks = [0, (L - 1) // 2, L - 1]
                    if "sbipart_2" in measures:
                        xaxticklabels = xaxticks
                        xaxlabel = r"Site, $j$"
                        hspace = 0.3
                    else:
                        xaxticklabels = []
                        xaxlabel = r""
                        hspace = 0.15
                    # ticklabels = ["↑", "↓"]
                elif meas.split("_")[0] == "sbipart":
                    ticks = [0, 8]
                    ticklabels = [r"$0$", r"$8$"]
                    xaxticks = [0, (L - 2) // 2, L - 2]
                    xaxticklabels = xaxticks
                    xaxlabel = r"Cut, $\ell$"

                elif meas.split("_")[0] == "s":
                    ticks = [0, 1]
                    ticklabels = [r"$0$", r"$1$"]
                    xaxticks = [0, (L - 1) // 2, L - 1]
                    xaxticklabels = xaxticks
                    xaxlabel = r"Site, $j$"
                vmin, vmax = ticks

            # For MPS data, load sperate from my QCA implementation
            # F type
            elif N == 5:
                # der = "/home/lhillber/documents/research/cellular_automata/qeca/qops"
                # der = os.path.join(
                #    der, f"qca_output/hamiltonian/rule{S}/rho_i.npy")
                # one_site = np.load(der)
                # one_site = one_site.reshape(2000, 22, 2, 2)
                # one_site = one_site[::, 2:-2, :, :]
                # T5, L5, *_ = one_site.shape
                # d = np.zeros((T5, L5))
                # ti = 0
                # for t, rhoi in enumerate(one_site):
                #    if t % 10 == 0:
                #        if meas == "exp_Z":
                #            d[ti, :] = get_expectation(rhoi, ops["Z"])
                #        elif meas == "s_2":
                #            d[ti, :] = get_entropy(rhoi, order=2)
                #        ti += 1

                Q = QCA(
                    dict(
                        L=19,
                        T=1000.0,
                        dt=0.1,
                        R=S,
                        r=2,
                        V="X",
                        IC="c3_f0-2",
                        BC="1-0000",
                        E=0.0,
                        N=1,
                        totalistic=True,
                        hamiltonian=True,
                        trotter=True,
                        symmetric=True,
                    )
                )
                #Q.run(tasks=["rhoj", "rhojk", "ebipart"])
                d = Q.get_measure(meas)
                Q.close()
                d = d[::10, :]

            # plot the data
            ax = grid[col + len(Skey)]
            ax2 = grid[col]
            I = ax.imshow(
                d[0:T],
                origin="lower",
                interpolation=None,
                cmap=cmaps[row],
                vmin=vmin,
                vmax=vmax,
            )
            I2 = ax2.imshow(
                d[1000 - T2 : 1000],
                origin="lower",
                interpolation=None,
                cmap=cmaps[row],
                vmin=vmin,
                vmax=vmax,
            )

            # colorbar
            cb = plt.colorbar(I, cax=ax.cax)
            cb.set_ticks(ticks)
            cb.set_ticklabels(ticklabels)

            if meas == "exp_Z":
                cb.ax.invert_yaxis()

            # axis ticks
            ax.set_yticks([i * (L - 1) for i in range(3)])
            ax.set_yticklabels([])
            ax2.set_yticks([(L - 1) // 2])
            ax2.set_yticklabels([])
            ax.set_xticks(xaxticks)
            if col == 0:
                ax.set_xticklabels(xaxticklabels)
                ax.set_xlabel(xaxlabel, labelpad=5)
            else:
                ax.set_xticklabels([])

            if col == 0:
                ax.set_xticklabels(xaxticklabels)

            # panel letter labels
            ax2.text(
                0.03,
                0.83,
                lett,
                color=cl,
                weight="bold",
                transform=ax2.transAxes,
            )

            # color bar label
            if col == len(Skey) - 1:
                ax.cax.text(
                    1.6,
                    0.78,
                    names[meas],
                    rotation=0,
                    transform=ax.transAxes,
                    ha="left",
                    va="center",
                )

            # panel titles
            if N == 3 and row == 0:
                ax2.set_title(r"$T_{%d}$" % S)
            elif N == 5 and row == 0:
                ax2.set_title(r"${F_{%d}}$" % S)
            ax2.tick_params(axis="x", direction="in")

        grid[0].set_yticklabels([r"$950$"])
        # axis labels
        grid[len(Skey)].set_ylabel("Time, $t$")
        grid[len(Skey)].yaxis.set_label_coords(-0.5, 0.8)
        # axis tick la5s
        grid[len(Skey)].set_yticklabels(
            [r"$" + str(i * (L - 1)) + "$" for i in range(3)]
        )

    # make it fit
    fig.subplots_adjust(hspace=hspace, left=0.05, right=0.97)

    # save out
    plt.savefig(plot_fname, dpi=300, bbox_inches="tight")
    print("plot saved to ", plot_fname)

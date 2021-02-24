import argparse
from qca import main_from_params_list, defaults, QCA
from copy import copy
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import matplotlib.animation as animation
from measures import get_bitstring_fidelity
import matplotlib.gridspec as gridspec
from figures import names

params = copy(defaults)

params["T"] = 100.0
params["V"] = "H"
params["totalistic"] = True

evos = [{"tris": "13_24"}, {"rods": "1234"}]
BCs = ["1-00", "1-0"]
Rs = [4, 2]

Ls = [8, 9, 12, 16]
Lxs = [2, 3, 3, 4]
ICs = ["f1-2-5-6", "f1-3-5-7", "f1-3-8-10", "f1-4-11-14"]

Es = [0.0, 0.02]
N = 1000

params_list = []
for E in Es:
    for IC, L, Lx in zip(ICs, Ls, Lxs):
        for IC_aug in [IC, "f0"]:
            for R, BC, evo in zip(Rs, BCs, evos):
                if R == 4 and IC_aug == "f0":
                    continue
                p = copy(params)
                if E == 0.0:
                    p["N"] = 1
                else:
                    p["N"] = N
                p["L"] = L
                p["Lx"] = Lx
                p["BC"] = BC
                p.update(evo)
                p["R"] = R
                p["IC"] = IC_aug
                p["E"] = E
                params_list.append(p)



def process_Lscaling(E, N):
    for IC, L, Lx in zip(ICs, Ls, Lxs):
        for IC_aug in [IC, "f0"]:
            for R, BC, evo in zip(Rs, BCs, evos):
                if R == 4 and IC_aug == "f0":
                    continue
                fig, axs = plt.subplots(1,2)
                p = copy(params)
                Ly = L // Lx
                p["L"] = L
                p["Lx"] = Lx
                p["BC"] = BC
                p.update(evo)
                p["R"] = R
                p["IC"] = IC_aug
                p["E"] = 0.0
                p["N"] = 1
                p2 = copy(p)
                p2["E"] = E
                p2["N"] = N
                Q = QCA(p)
                Q2 = QCA(p2)
                d = Q.get_measure("C_2")
                d2 = Q2.get_measure("C_2")
                y = 1 - np.abs(d[2:] - d2[2:])/d[2:]
                F = Q2.F()
                axs[0].plot(y, label="C")
                axs[0].plot(F, label="F")
                axs[0].legend()
                axs[1].plot(d[2:])
                axs[1].plot(d2[2:])
                axs[0].set_yscale("log")
                axs[1].set_yscale("log")
                if evo == {"tris": "13_24"}:
                    fig.suptitle(f"'F12', R={R}, IC={IC}, Lx,Ly = {Lx,Ly}")
                else:
                    fig.suptitle(f"'F10', R={R}, IC={IC}, Lx,Ly = {Lx,Ly}")
                plt.show()
                Q.close()
                Q2.close()




def process(fig, R, ICs, evos, Es, measures):
    count = 0
    ims = np.zeros((len(evos), len(ICs), len(measures),
        len(Es)), dtype="object")
    Qs = np.zeros((len(ICs), len(evos), len(Es)), dtype="object")
    for J, evo in enumerate(evos):
        for K, IC in enumerate(ICs):
            if K != len(ICs) - 1:
                cbar_mode = None
            else:
                cbar_mode = "edge"

            count += 1
            grid = ImageGrid(
                fig,
                int(str(len(evos)) + str(len(ICs)) + str(count)),
                nrows_ncols=(len(evos), len(Es)),
                direction="row",
                axes_pad=[-0.1, 0.1],
                label_mode="L",
                add_all=True,
                cbar_mode="each",
                cbar_location="right",
                cbar_size="15%",
                cbar_pad=0.2,
            )

            inner_count = 0
            for j, meas in enumerate(measures):
                vmax = 1
                if meas[0] == "s":
                    vmin = 0
                if meas.split("_")[0] == "exp":
                    vmin = -1

                for k, E in enumerate(Es):
                    ax = grid[inner_count]
                    inner_count += 1

                    p = copy(params)
                    if E == 0.0:
                        p["N"] = 1
                    else:
                        p["N"] = N
                    p.update(evo)
                    p["R"] = R
                    p["IC"] = IC
                    p["E"] = E
                    Q = QCA(p)
                    data = Q.get_measure(meas)
                    data = Q.to2d(data)
                    I = ax.imshow(data[0], vmin=vmin, vmax=vmax, cmap="inferno")
                    ims[J, K, j, k] = I
                    Qs[J, K, k] = Q
                    cb = plt.colorbar(I, cax=ax.cax)
                    ax.set_xticks(np.arange(Q.Lx))
                    ax.set_yticks(np.arange(Q.Ly))

                    if K == len(ICs) - 1 and k == len(Es)-1:
                        ax.cax.text(
                            1.85,
                            0.5,
                            names[meas],
                            rotation=0,
                            transform=ax.transAxes,
                            ha="left",
                            va="center",
                        )
                    else:
                        cb.ax.clear()
                        cb.set_ticks([])

    def update(t):
        fig.suptitle(f"t={t}")
        for J, evo in enumerate(evos):
            for K, IC in enumerate(ICs):
                for j, meas in enumerate(measures):
                    if meas[0] == "s":
                        vmin = 0
                    if meas.split("_")[0] == "exp":
                        vmin = -1
                    for k, E in enumerate(Es):
                        Q = Qs[J, K, k]
                        data = Q.to2d(Q.get_measure(meas))
                        ims[J,K,j,k].set_array(data[t])

    anim = animation.FuncAnimation(fig, update, frames=100)
    #plt.show()
    anifname = "figures/animation/noise_effect.mp4"
    anim.save(anifname)
    print("animation saved to")
    print(anifname)

    plt.close("all")
    fig = plt.figure()
    gs0 = gridspec.GridSpec(2, 2, figure=fig)
    for J, evo in enumerate(evos):
        for K, IC in enumerate(ICs):
            gs = gs0[J, K].subgridspec(3, 1)
            ax0 = fig.add_subplot(gs[0])
            ax1 = fig.add_subplot(gs[1])
            ax2 = fig.add_subplot(gs[2])

            p1 = copy(params)
            p1.update(evo)
            p1["R"] = R
            p1["IC"] = IC
            p1["N"] = 1
            p2 = copy(p1)
            p2["E"] = Es[1]
            p2["N"] = N
            Q1 = QCA(p1)
            Q2 = QCA(p2)
            ps = Q2.bitstring  # measured
            qs = Q1.bitstring  # expected
            F = np.array([get_bitstring_fidelity(p, q) for p, q in zip(ps, qs)])
            tmin=0
            ax2.plot(Q.ts[tmin:], F[tmin:], c="k")
            for Q, label, in zip([Q1, Q2], ["ideal", f"E={Q2.E*100}\%"]):
                ax0.plot(Q.ts[tmin:], Q.C(2)[tmin:], label=label)
                ax1.plot(Q.ts[tmin:], Q.Y(2)[tmin:], label=label)

            if J == 0 and K == 0:
                ax0.legend(ncol=2, loc="center", bbox_to_anchor=[0.5, 1.4])

            if K == 0:
                ax0.set_ylabel("$\mathcal{C}$")
                ax1.set_ylabel("$\mathcal{Y}$")
                ax2.set_ylabel("$\mathcal{F}$")
            if J ==  len(evos) - 1:
                ax2.set_xlabel("Time, $t$")
    plt.savefig("figures/noise_effect.pdf")


main_from_params_list(params_list, tasks=[
                      "rhoj", "rhojk", "bitstring"], recalc=False)

#process(plt.figure(figsize=(7, 7)), 2, ICs,
#        evos, Es, measures=["exp_Z", "s_2"])
#
#
#process_Lscaling(Es[-1], N)

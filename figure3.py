from qca import QCA
import numpy as np
import measures as ms
from states import make_state
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from networkviz import netviz
from figures import (
    names,
    colors,
    markers,
    lines,
    moving_average,
    firstdiff,
    multipage,
    letters,
)
from matplotlib import rc

rc("text", usetex=True)
font = {"size": 12, "weight": "normal"}
mpl.rc(*("font",), **font)
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}\usepackage{sansmath}\sansmath"

recalc_list = []

def network_measures_scatter(
    Skey,
    L=18,
    BC="1-00",
    ICkey=[
        "c3_f1",
        "c3_f0-1",
        "c3_f0-2",
        "c3_f0-1-2",
        "d3",
        "d4",
        "d5",
        "d6",
        "d7",
        "R123",
    ],
    V="H",
    T=1000,
    axs=None,
    statestrs=["R123", "W", "GHZ", "C6-3"],
    meas_axs=[["sbisect_2", "C"], ["sbisect_2", "Y"]],
    order=2,
):
    if axs is None:
        fig, axs = plt.subplots(2, 1, figsize=(1.5, 2.4))
    for col, measures in enumerate(meas_axs):
        d = {meas: {"avg": [], "std": [], "c": []} for meas in measures}
        d["colors"] = []
        ax = axs[col]
        for S in Skey:
            for IC in ICkey:
                d["colors"] += [colors[S]]
                Q = QCA(
                    dict(
                        L=L,
                        T=T,
                        dt=1,
                        R=S,
                        r=1,
                        V=V,
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
                # Q.check_repo(test=False)
                for meas in measures:
                    try:
                        dat = Q.get_measure(meas, save=True)[500:]
                    except:
                        print(meas, Q.available_tasks)
                        recalc_list.append(Q.params)
                        dat = np.zeros(len(Q.ts[500:]))
                    d[meas]["avg"] += [np.mean(dat)]
                    d[meas]["std"] += [np.std(dat)]
                Q.close()
        # ax.errorbar(
        #   d[measures[0]]["avg"],
        #   d[measures[1]]["avg"],
        #   xerr=d[measures[0]]["std"],
        #   yerr=d[measures[1]]["std"],
        #   fmt = ".k",
        # )

        ax.scatter(
            d[measures[0]]["avg"],
            d[measures[1]]["avg"],
            c=d["colors"],
            cmap=cmap,
            linewidths=0.1,
            edgecolor="k",
            alpha=0.8,
            vmin=0,
            vmax=len(Skey) - 1,
            s=40,
        )

        for statestr in statestrs:
            m = markers[statestr[0]]
            state = make_state(L, statestr)
            rhoj = ms.get_rhoj(state)
            rhojk = ms.get_rhojk(state)
            bisect = ms.get_bisect(state)
            ds1 = ms.get_entropy(rhoj, order)
            ds2 = ms.get_entropy2(rhojk, order)
            dMI = ms.get_MI(ds1, ds2)
            dsc = ms.renyi_entropy(bisect, order=2)
            dC = ms.network_clustering(dMI)
            dY = ms.network_disparity(dMI)
            statedata = {"C": dC, "Y": dY, "sbisect": dsc}
            if m in ("x", "."):
                facecolors = "k"
            else:
                facecolors = "none"
            ax.scatter(
                statedata[measures[0].split("_")[0]],
                statedata[measures[1].split("_")[0]],
                facecolors=facecolors,
                edgecolors="k",
                marker=m,
                s=40,
            )

        ax.minorticks_off()
        ax.set_xticks([1, 3, 5, 7, 9])
        if meas[0] == "C":
            ax.set_yscale("log")
            ax.set_yticks([1e-5, 1e-3, 1e-1])
            ax.set_yticklabels([])
            ax.set_ylim([1e-6, 1.8])
            ax.set_xticklabels([])
        elif meas[0] == "Y":
            ax.set_yticks([0.0, 0.2, 0.4])
            ax.set_yticklabels([])
            ax.set_ylim([-0.001, 0.58])
            ax.set_xticklabels(["$1$", "$3$", "$5$", "$7$", "$9$"])
            ax.set_xlabel(names[measures[0] + "avg"])


def network_measures_timeseries(
    Skey,
    L=19,
    BC="1-00",
    ICkey=["c3_f1", "R123"],
    Vkey=["H"],
    T=1000,
    axs=None,
    measures=["C", "Y"],
    randline=False,
):
    if axs is None:
        fig, axs = plt.subplots(2, 1, figsize=(3, 2.3), sharex=True)
    for col, meas in enumerate(measures):
        ax = axs[col]
        Rstack = []
        for V in Vkey:
            ls = lines[V]
            for IC in ICkey:
                for S in Skey:
                    c = colors[S]
                    if S == 14:
                        Vs = Vkey
                    else:
                        Vs = ["H"]
                    Q = QCA(
                        dict(
                            L=L,
                            T=T,
                            dt=1,
                            R=S,
                            r=1,
                            V=V,
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
                    print(Q.params)

                    # Q.check_repo(test=False)

                    d = Q.get_measure(meas, save=True)[3:]
                    Q.close()

                    if IC[0] == "R":
                        Rstack += [d]
                    else:
                        zorder = None
                        if S == 1:
                            zorder = 10
                        if V in Vs:
                            # md = moving_average(d, n=3)
                            (line,) = ax.plot(
                                np.arange(len(d)), d, c=c, lw=1, ls="-", zorder=zorder,
                            )

                            if ls == "--":
                                line.set_dashes([2, 2, 10, 2])

                    if meas[0] == "C":
                        ax.set_yscale("log")
                        ax.set_yticks([1e-5, 1e-3, 1e-1])
                        ax.set_ylim([1e-6, 1.8])
                    elif meas[0] == "Y":
                        ax.set_yticks([0.0, 0.2, 0.4])
                        ax.set_ylim([-0.001, 0.45])
                    ax.set_ylabel(names[meas])
                    ax.set_xlabel("Time, $t$")
                    ax.label_outer()
                    ax.set_xticks([0, 250, 500, 750, 1000])
        if randline:
            Rstack = np.array(Rstack)
            Ravg = np.mean(Rstack)
            Rstd = np.std(Rstack)
            print()
            print("rand", meas, Ravg, Rstd)
            print()
            ax.plot(np.arange(len(d)), Ravg * np.ones(len(d)), c="k", lw=1)
            ax.fill_between(
                [0, 1001],
                [Ravg + Rstd],
                [Ravg - Rstd],
                facecolor="k",
                alpha=1.0,
                zorder=10,
            )


def cluster_angle_scaling(order=2):
    fig, axs = plt.subplots(3, 1, figsize=(3, 7), sharex=True)
    phs = np.linspace(10, 360, 100)
    L = 18
    bases = ["6-3", "3-6", "9-2", "2-9"]
    for base in bases:
        statedata = {"C": [], "Y": [], "sc": []}
        statestrs = [f"C{base}_{ph}" for ph in phs]
        for statestr in statestrs:
            state = make_state(L, statestr)
            rhoj = ms.get_rhoj(state)
            rhojk = ms.get_rhojk(state)
            bisect = ms.get_bisect(state)
            ds1 = ms.get_entropy(rhoj, order)
            ds2 = ms.get_entropy2(rhojk, order)
            dMI = ms.get_MI(ds1, ds2)
            dsc = ms.renyi_entropy(bisect, order=2)
            dC = ms.network_clustering(dMI)
            dY = ms.network_disparity(dMI)

            statedata["C"].append(dC)
            statedata["Y"].append(dY)
            statedata["sc"].append(dsc)
        axs[0].plot(phs, statedata["C"])
        axs[1].plot(phs, statedata["Y"])
        axs[2].plot(phs, statedata["sc"], label=base)
    axs[2].legend()
    axs[0].set_ylabel(names["C"])
    axs[1].set_ylabel(names["Y"])
    axs[2].set_ylabel(names["sbisect_2"])
    axs[2].set_xlabel("$\phi$ [deg]")
    axs[2].label_outer()


def QI_measures(Lkey, measures, order=2):
    statestrs = ["R123", "W", "GHZ"]
    markers = ["x", "d", "*", "^"]
    QI = np.zeros((len(statestrs), len(Lkey), len(measures)))
    for j, statestr in enumerate(statestrs):
        for k, L in enumerate(Lkey):
            state = make_state(L, statestr)
            rhoj = ms.get_rhoj(state)
            rhojk = ms.get_rhojk(state)
            bisect = ms.get_bisect(state)
            s1 = ms.get_entropy(rhoj, order)
            s2 = ms.get_entropy2(rhojk, order)
            MI = ms.get_MI(s1, s2)
            sc = ms.renyi_entropy(bisect, order=2)
            C = ms.network_clustering(MI)
            Y = ms.network_isparity(MI)
            QI[j, k, 0] = sc
            QI[j, k, 1] = C
            QI[j, k, 2] = Y
    for m, measure in enumerate(["sbisect_2", "C", "Y"]):
        fig, ax = plt.subplots(1, 1, figsize=(1.3, 1.0))
        for j, (mk, statestr) in enumerate(zip(markers, statestrs)):
            if mk in ("x", "."):
                facecolors = "k"
            else:
                facecolors = "none"
            for k, L in enumerate(Lkey):
                ax.scatter(
                    Lkey,
                    QI[j, :, m],
                    facecolors=facecolors,
                    edgecolors="k",
                    marker=mk,
                    s=40,
                )
        ax.set_ylabel(names[measures[1] + "avg"])
        ax.set_xlabel(names[measures[0] + "avg"])
        # ax.set_yscale("log")
        fig.subplots_adjust(left=0.18, bottom=0.22, top=0.9)


# Lscaling of states, long time averages
def measure_Lscaling(
    Skey,
    Lkey=[6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    BC="1-00",
    ICkey=["c3_f1", "R123"],
    V="H",
    T=1000,
    axs=None,
    measures=["C", "Y"],
):
    if axs is None:
        fig, axs = plt.subplots(2, 1, figsize=(1.3, 1.0), sharex=True)

    lta = np.zeros((len(Skey), len(ICkey), len(Lkey), len(measures)))
    dlta = np.zeros((len(Skey), len(ICkey), len(Lkey), len(measures)))
    for i, S in enumerate(Skey):
        for j, IC in enumerate(ICkey):
            for k, L in enumerate(Lkey):
                Q = QCA(
                    dict(
                        L=L,
                        T=T,
                        dt=1,
                        R=S,
                        r=1,
                        V=V,
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
                # Q.check_repo(test=False)
                for m, meas in enumerate(measures):
                    if meas[0] == "d":
                        d = Q.get_measure(meas[1:], save=True)
                        d = np.abs(firstdiff(d, acc=2, dx=1))

                    else:
                        d = Q.get_measure(meas, save=True)
                    d = d[500:]
                    lta[i, j, k, m] = np.mean(d)
                    dlta[i, j, k, m] = np.std(d)
                Q.close()
    assert IC[0] == "R"
    initrand = np.mean(lta[:, 1, :, :], axis=0)
    dinitrand = np.mean(dlta[:, 1, :, :], axis=0)

    for m, measure in enumerate(measures):
        ax = axs[m]
        for i, S in enumerate(Skey):
            c = colors[S]
            for j, IC in enumerate(ICkey):
                r = initrand[:, m]
                dr = dinitrand[:, m]
                y = lta[i, j, :, m]
                dy = dlta[i, j, :, m]
                x = Lkey
                ax.fill_between(x, y + dy, y - dy, facecolor=c, alpha=0.2)
                if j != 1:
                    ax.scatter(x, y, marker="o", s=8, c=c)
                    if measure in ("C", "C_1", "C_2", "Dsbisect_2"):
                        mm, bb = np.polyfit(x, np.log(y), 1)
                        xs = np.linspace(x[0], x[-1], 100)
                        ys = np.e ** (bb + mm * xs)
                        ax.plot(xs, ys, c=c,
                                label=f"$\lambda = {round(mm, 2)}$")
                        print(f"clustering slope ** V={V}, S={S}: lambda={mm}")
        ax.scatter(x, r, marker="o", s=8, c="k")
        ax.fill_between(x, r + dr, r - dr, facecolor="k", alpha=0.2)
        if measure in ("C", "C_1", "C_2", "Dsbisect_2"):
            mm, bb = np.polyfit(x, np.log(r), 1)
            xs = np.linspace(x[0], x[-1], 100)
            rs = np.e ** (bb + mm * xs)
            ax.plot(xs, rs, c="k", label=f"$\lambda = {round(mm, 2)}$")
            print(f"clustering slope ** random throw: lambda={mm}")

            # ax.legend(bbox_to_anchor=(1,1))

        if measure[0] == "C":
            ax.set_yscale("log")
            ax.set_yticks([1e-5, 1e-3, 1e-1])
            ax.set_ylim([1e-6, 1.8])
            ax.set_xticks([6, 10, 14, 18])
            ax.set_xticklabels([])
            ax.set_ylabel(names[measure + "avg"], labelpad=5)
        elif measure[0] == "Y":
            ax.set_yticks([0.0, 0.2, 0.4])
            ax.set_yticklabels([0.0, 0.2, 0.4])
            ax.set_xticks([6, 10, 14, 18])
            ax.set_ylim([-0.001, 0.58])
            ax.set_ylabel(names[measure + "avg"], labelpad=5)
            ax.set_xlabel("Size, $L$")


if __name__ == "__main__":

    Skey = [6, 1, 14, 13]
    L = 19
    order = 1
    BC = "1-00"
    statestrs = ["R123", "W", "GHZ", f"C{L}-1_45"]
    plot_fname = f"figures/figure3/figure3_L{L}_BC{BC}_V12.pdf"

    cmap = mcolors.ListedColormap(colors.values())

    fig = plt.figure(figsize=(3.375 * 2, 3.4))
    gs1 = gridspec.GridSpec(1, 8)
    gs1.update(left=0.07, right=0.99, bottom=0.78, top=0.98)

    gs2 = gridspec.GridSpec(2, 2)
    gs2.update(left=0.13, right=0.48, bottom=0.18, top=0.66)

    gs3 = gridspec.GridSpec(2, 2)
    gs3.update(left=0.6, right=0.98, bottom=0.18, top=0.66)

    netaxs = [fig.add_subplot(gs1[0, i]) for i in range(8)]

    Cax1 = fig.add_subplot(gs2[0, 0:2])
    Yax1 = fig.add_subplot(gs2[1, 0:2])

    Cax2 = fig.add_subplot(gs3[0, 0])
    Cax3 = fig.add_subplot(gs3[0, 1])

    Yax2 = fig.add_subplot(gs3[1, 0])
    Yax3 = fig.add_subplot(gs3[1, 1])

    #network_measures_scatter(
    #    Skey,
    #    L=L,
    #    BC=BC,
    #    ICkey=[
    #        "c3_f1",
    #        "c3_f0-1",
    #        "c3_f0-2",
    #        "c3_f0-1-2",
    #        "d3",
    #        "d4",
    #        "d5",
    #        "d6",
    #        "d7",
    #        "R123",
    #    ],
    #    V="H",
    #    meas_axs=[[f"sbisect_{2}", f"C_{order}"], [f"sbisect_{2}", f"Y_{order}"]],
    #    statestrs=statestrs,
    #    axs=[Cax3, Yax3],
    # )

    print("enter")
    print(recalc_list)
    #from qca import main_from_params_list
    #main_from_params_list(recalc_list, tasks=["bipart"], recalc=False)

    netviz(
        Skey,
        t=500,
        L=L,
        IC="c3_f1",
        V="H",
        layout="spring",
        statestrs=statestrs,
        BC=BC,
        axs=netaxs,
        order=order,
    )

    network_measures_timeseries(
        Skey,
        L=L,
        BC=BC,
        ICkey=["c3_f1", "R123"],
        Vkey=["H", "HP_45"],
        measures=[f"C_{order}", f"Y_{order}"],
        randline=True,
        axs=[Cax1, Yax1],
     )

    measure_Lscaling(
        Skey,
        Lkey=[6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        BC=BC,
        ICkey=["c3_f1", "R123"],
        V="H",
        measures=[f"C_{order}", f"Y_{order}"],
        axs=[Cax2, Yax2],
     )

    netaxs[0].text(
        -0.45, 0.63, letters[0], weight="bold", transform=netaxs[0].transAxes
     )

    Cax1.text(0.01, 0.2, letters[1], weight="bold", transform=Cax1.transAxes)
    Cax2.text(0.02, 0.08, letters[3], weight="bold", transform=Cax2.transAxes)
    Cax3.text(0.02, 0.08, letters[5], weight="bold", transform=Cax3.transAxes)

    Yax1.text(0.92, 0.76, letters[2], weight="bold", transform=Yax1.transAxes)
    Yax2.text(0.85, 0.76, letters[4], weight="bold", transform=Yax2.transAxes)
    Yax3.text(0.85, 0.76, letters[6], weight="bold", transform=Yax3.transAxes)

    # fig.subplots_adjust(left=0.175, bottom=0.16, top=0.98, right=0.97, hspace=0.1)
    #deltaS_bond_timmeseries(Skey)
    #cluster_angle_scaling()

    multipage(plot_fname, clip=False, dpi=300)
    plt.close("all")
    print("plot saved to ", plot_fname)

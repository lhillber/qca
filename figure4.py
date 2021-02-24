import matplotlib.gridspec as gridspec
import numpy as np
from numpy.linalg import eigvalsh
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from qca import QCA
from figures import (
    select,
    colors,
    lines,
    names,
    firstdiff,
    moving_average,
    brody_fit,
    page_fit,
    multipage,
    letters,
)
from matplotlib.patches import Patch
from matplotlib import rc
import matplotlib as mpl

rc("text", usetex=True)
font = {"size": 12, "weight": "normal"}
mpl.rc(*("font",), **font)
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}\usepackage{sansmath}\sansmath"


def Dscenter_timeseries(
    Skey, L=19, T=1000, BC="1-00", Vkey=["H"], ICkey=["c3_f1", "R123"], ax=None, t0=10
):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(3, 1.5), sharey=True, sharex=True)

    Rstack = []
    for IC in ICkey:
        for S in Skey:
            c = colors[S]
            if S == 14:
                Vs = [Vkey]
            else:
                Vs = ["H"]
            for V in Vkey:
                ls = lines[V]

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
                d = Q.get_measure("sbisect_2", save=True)[t0:]

                d = np.abs(firstdiff(d, acc=8, dx=1))

                ad = moving_average(d, n=L)
                ts = np.arange(t0, len(d) + t0)
                if IC[0] == "R":
                    Rstack += [d]

                else:
                    (line,) = ax.plot(
                        ts[: -L + 1],
                        ad,
                        c=c,
                        lw=1,
                        marker="o",
                        markersize=0.5,
                        linestyle=ls,
                        markeredgecolor="none",
                    )
                    (line,) = ax.plot(ts[::L], d[::L], c=c, lw=1, ls=ls, alpha=0.4)
                if ls == "--":
                    line.set_dashes([2, 2, 15, 2])
                ax.set_yscale("log")
                ax.minorticks_off()
                ax.set_ylabel(names["Dsbisect_2"])
                # ax.yaxis.set_label_coords(-0.05, 0.43)
                # ax.xaxis.set_label_coords(0.5, -0.2)
                xticks = [0, 250, 500, 750, 1000]
                ax.set_xticks(xticks)
                ax.set_xticklabels([f"${i}$" for i in xticks])
                ax.set_xlabel("Time, $t$")
                ax.set_yticks([1e-4, 1e-2, 1e-0])
                ax.set_ylim([1e-5, 10])

    Rstack = np.array(Rstack)
    # ax.plot(Rstack.T)
    Ravg = np.mean(Rstack)
    Rstd = np.std(Rstack)
    ax.fill_between(
        [0, 1001 - L], [Ravg + Rstd], [Ravg - Rstd], facecolor="k", alpha=0.2, zorder=10
    )
    # ax.text(0.045, 2 * 0.055, "Random initializations", transform=ax.transAxes)


def simple_page(Skey, Lkey=[19], T=1000, BC="1-00", V="H", ICkey=["c3_f1"], ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(1.5, 1.3))
    for o, S in enumerate(Skey):
        color = colors[S]
        for IC in ICkey:
            sb_avg = []
            sb_std = []
            sb_res = []
            ns = []
            for L in Lkey:
                ells = np.arange(L - 1)
                dells = np.linspace(0, L - 2, 100)

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
                sb = Q.sbipart_2
                sba = np.mean(sb[500:], axis=0)
                sbd = np.std(sb[500:], axis=0)

                func, popt, pcov = page_fit(sba, sbd)

                ax.fill_between(
                    ells, sba + sbd, sba - sbd, facecolor=color, alpha=0.4, zorder=0
                )
                ax.scatter(
                    ells,
                    sba,
                    edgecolors=color,
                    linewidths=0.4,
                    marker="o",
                    facecolors=color,
                    s=6,
                    zorder=o,
                )
                ax.plot(
                    dells,
                    func(dells),
                    color=color,
                    lw=1,
                    label=r"$T_{%s}$" % S,
                    zorder=o,
                )
                ax.set_xlabel(r"Cut, $\ell$")
                ax.set_ylabel(names["sbipart_2avg"])
                #ax.yaxis.set_label_coords(-0.05, 0.5)
                #ax.xaxis.set_label_coords(0.5, -0.15)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.set_yticks([0, 4, 8])
                xticks = [1,8,16]
                ax.set_xticks(xticks)
                ax.set_xticklabels([f"${x}$" for x in xticks])
                sb_avg += [sba[L // 2]]
                sb_std += [sbd[L // 2]]
                sb_res += [(func(ells) - sba) ** 2]
                print(Q.R, np.mean(sb_res))

    # fig.subplots_adjust(left=0.3, bottom=0.3, top=0.98, right=0.98)
    # ax.legend(
    #    loc=2,
    #    fontsize=8,
    #    handlelength=1,
    #    labelspacing=0.2,
    #    handletextpad=0.2,
    #    frameon=False,
    # )


#def plot(plot_fname, T=1000):
#    fig = plt.figure(figsize=(3.4, 2))
#    gs = gridspec.GridSpec(2, 3)
#    ax = fig.add_subplot(gs[:, 0:2])
#    ax2 = fig.add_subplot(gs[0, 2])
#    ax3 = fig.add_subplot(gs[1, 2])
#
#    axins1 = ax.inset_axes((1 - 0.335, 1 - 0.335, 0.3, 0.3))
#    cs = ["r", "g", "b", "k"]
#    ICkey = "C1_f0"
#    Lkey = 18
#    for color, S in zip(cs, Skey):
#        for IC in ICkey:
#            sb_avg = []
#            sb_std = []
#            sb_res = []
#            ns = []
#            for L in Lkey:
#                ells = np.arange(L - 1)
#                dells = np.linspace(0, L - 2, 100)
#                sim = select(T, L, S, IC, "H", "1-00")
#                if sim is None:
#                    print("No sim!")
#                    continue
#                S = sim["S"]
#                L = sim["L"]
#                IC = sim["IC"]
#                h5file = sim["h5file"]
#                sb = h5file["sbond-2"][:]
#                sba = np.mean(sb[500:], axis=0)
#                sbd = np.std(sb[500:], axis=0)
#                sb_avg += [sba[L // 2]]
#                sb_std += [sbd[L // 2]]
#                ax.set_xlabel("$\\ell$")
#                # ax.label_outer()
#                func, popt, pcov = fit_page(sba)
#                sb_res += [(func(ells) - sba) ** 2]
#                if L in (18,):
#                    ls = lss[L]
#                    ax.fill_between(ells, sba + sbd, sba - sbd, color=color, alpha=0.3)
#                    ax.scatter(ells, sba, color=color, marker="o", s=6)
#
#                    ax.plot(
#                        dells,
#                        func(dells),
#                        color=color,
#                        ls=ls,
#                        lw=1,
#                        label="$R={}$".format(S),
#                    )
#                    ax.set_xlabel("$\\ell$")
#                    ax.set_ylabel("$\\overline{s11~}_{\\ell}$")
#                    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
#                    if L == 18:
#                        axins1.semilogx(
#                            sb[:, int(L / 2)], color=color, lw=1, label=names[IC]
#                        )
#            sb_avg = np.array(sb_avg)
#            sb_std = np.array(sb_std)
#            ax2.errorbar(
#                Lkey,
#                sb_avg,
#                yerr=sb_std,
#                color=color,
#                fmt="-o",
#                capsize=1,
#                elinewidth=1,
#                markersize=2,
#                lw=1,
#            )
#            sb_rss = [np.sum(res) for res in sb_res]
#            ax3.plot(Lkey, sb_rss, "-o", color=color, lw=1, markersize=2)
#
#    plt.subplots_adjust(wspace=0.7, hspace=0.1, bottom=0.22, right=0.97, top=0.97)
#
#    ax2.set_ylabel("$\\overline{s}_{:L/2}$", fontsize=9, labelpad=-2)
#    ax2.set_yticks([2.0, 8.0])
#    ax2.set_xticks([])
#
#    ax3.set_ylabel("RSS", fontsize=9, labelpad=-11)
#    ax3.set_xlabel("$L$")
#    ax3.set_yticks([0, 0.07])
#    ax3.set_xticks([10, 14, 18])
#
#    ax.set_yticks([0, 4, 8])
#    ax.set_xticks([0, 4, 8, 12, 16])
#    ax.set_xlim(right=17)
#    ax.set_ylim(top=11)
#    ax.legend(
#        loc=2,
#        fontsize=9,
#        handlelength=1,
#        labelspacing=0.2,
#        handletextpad=0.2,
#        frameon=False,
#    )
#
#    axins1.set_xticks([1, 1000])
#    axins1.set_xticklabels([0, 3])
#    axins1.set_yticks([0, 8])
#    axins1.set_xlabel(r" $\log_{10}(t)$", fontsize=9, labelpad=-6)
#    axins1.set_ylabel("$s_{:L/2}$", fontsize=9, labelpad=-2)
#    axins1.tick_params(axis="both", labelsize=9)
#    # axins1.axvline(100, color="k", lw=1)
#    axins1.patch.set_alpha(1)


def spectrum_statistics(
    Skey,
    L=19,
    T=1000,
    BC="1-00",
    IC="c3_f1",
    V="H",
    t0=11,
    tol=1e-10,
    order=10,
    bins=15,
    clip=10,
    axs=None,
    intermediate_plots=True,
):

    if axs is None:
        fig, axs = plt.subplots(1, 1, figsize=(3, 2), sharex=True)
    if intermediate_plots:
        hist_figs = []
        spec_figs = []
        spec_figs2 = []

    legend_elements = [
        Patch(facecolor=colors[S], edgecolor=None, label=r"$T_{%d}$" % S)
        for S in [6, 1, 14, 13]
    ]

    for j, S in enumerate(Skey):
        if S == 6:
            bins0 = bins
            order0 = 2
            clip0 = 0
        else:
            bins0 = bins
            order0 = order
            clip0 = clip
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
        try:
            espec = Q.espec
        except:
            d = Q.bisect
            espec = np.zeros((d.shape[0], d.shape[1]))
            for ti, rho in enumerate(d):
                espec[ti, :] = eigvalsh(rho)
            Q.h5file["espec"] = espec

        etas = []
        detas = []
        ii = 0
        if intermediate_plots:
            hist_fig, hist_axs = plt.subplots(
                3, 3, figsize=(4, 4), sharex=True, sharey=True
            )
            spec_fig, spec_axs = plt.subplots(
                3, 3, figsize=(4, 4), sharex=True, sharey=True
            )
            spec_fig2, spec_ax2 = plt.subplots(1, 1, figsize=(4, 2))

            esort = np.array([np.sort(espec[t])[-10:] for t in range(len(espec[:, 0]))])
            spec_ax2.semilogy(np.arange(t0, len(esort)), esort[t0:], lw=1)
            spec_ax2.set_xlabel("t")
            spec_ax2.set_ylabel(r"$\lambda_i$")
            spec_fig2.suptitle(r"$T_{%d}$" % S)

        for ti, es0 in enumerate(espec[t0:1000]):
            es0 = np.sort(es0)
            es0 = es0[es0 > tol]
            if clip0 > 0:
                es0 = es0[clip0:-clip0]
            coeffs = np.polyfit(es0, np.arange(len(es0)), order0)
            unrollf = np.poly1d(coeffs)
            es = unrollf(es0)
            es = np.sort(es)
            s = es[1:] - es[:-1]
            s = s[s < 7]

            n, bin = np.histogram(s, density=True, bins=bins0)
            # x = (bin[1:] + bin[:-1]) / 2.0
            x = bin[:-1] + np.diff(bin) / 2

            xs = np.linspace(x[0], x[-1], 100)
            func, popt, pcov = brody_fit(x, n)
            detas.append(np.sqrt(np.diag(pcov)[0]))
            etas.append(popt[0])

            if intermediate_plots:
                if (ti + t0) % 100 == 0:
                    row, col = ii // 3, ii % 3
                    hist_ax = hist_axs[row, col]
                    spec_ax = spec_axs[row, col]

                    # dx = x[1] - x[0]
                    # n = np.insert(n, 0, 0)
                    # n = np.insert(n, len(n), 0)
                    # x = np.insert(x, 0, x[0] - dx / 2)
                    # x = np.insert(x, len(x), x[-1] + dx / 2)
                    hist_ax.step(x, n, where="mid")
                    # if S == 14:
                    #    hist_ax.set_xlim(0, 3)
                    hist_ax.plot(xs, func(xs))
                    hist_ax.set_title(f"t={t0+ti}", pad=-13)
                    hist_fig.suptitle(r"$T_{%d}$" % S)

                    spec_ax.plot(es, marker="o", c="k", ms=5)
                    spec_ax.set_title(f"t={t0+ti}", pad=-13)
                    spec_fig.suptitle(r"$T_{%d}$" % S)

                    ii += 1

                    if col == 1 and row == 2:
                        hist_ax.set_xlabel("$d^{\prime}$")
                        spec_ax.set_xlabel("$i$")
                    if col == 0 and row == 1:
                        hist_ax.set_ylabel(r"$\mathcal{D}$")
                        spec_ax.set_ylabel("$\lambda^{\prime}_i$")
                    hist_ax.tick_params(direction="inout")

        if intermediate_plots:
            hist_figs.append(hist_fig)
            spec_figs.append(spec_fig)
            spec_figs2.append(spec_fig2)

        etas = np.array(etas)
        detas = np.array(detas)

        ts = np.arange(t0, len(etas) + t0)
        mask = detas < 1
        etas = etas[mask]
        detas = detas[mask]
        ts = ts[mask]

        aetas = moving_average(etas, n=L)
        axs.plot(
            ts[: -L + 1],
            aetas,
            marker="o",
            markersize=0.5,
            linestyle="-",
            markeredgecolor="none",
            color=colors[S],
            lw=1,
            zorder=j,
        )
        avgerr = np.mean(detas)
        # axs.plot(ts, etas, c=colors[S], alpha=0.3)
        axs.errorbar(
            ts[: -L + 1],
            aetas,
            yerr=detas[: -L + 1],
            color=colors[S],
            # errorevery=5,
            elinewidth=0.4,
            linestyle="none",
            alpha=0.2,
            zorder=(j + 1) * 4,
        )

    # axs.legend(loc="lower right")
    # axs.set_xlabel("$t$")
    axs.set_ylabel("$\eta$", labelpad=1)
    # axs.yaxis.set_label_coords(-0.2, 0.5)
    xticks = [0, 250, 500, 750, 1000]
    axs.set_xticks(xticks)
    axs.set_xticklabels([])

    # axs.set_xticklabels([f"${i}$" for i in xticks])
    axs.set_yticks([0.0, 0.5, 1.0])
    axs.set_ylim(top=1.6)
    axs.legend(
        handles=legend_elements,
        loc="lower left",
        bbox_to_anchor=[-0.01, 0.91],
        frameon=False,
        markerfirst=False,
        ncol=4,
        columnspacing=1.5,
        handletextpad=0.3,
        handlelength=0.7,
    )
    if intermediate_plots:
        plt.gcf().tight_layout()
        plot_fname = f"figures/figure4/spectrum_statistics_L{L}_BC{BC}_IC{IC}_tol{tol}_clip{clip0}_order{order0}_bins{bins0}.pdf"
        multipage(plot_fname, clip=True, dpi=10 * plt.gcf().dpi)
        print("plot saved to ", plot_fname)


def QEB_plot(ax, dt=0.1):
    # load data from text file
    # Matt fitting results
    # d1 = np.loadtxt("data/qeb_decay/rule_admix_lifetime_17.txt", skiprows=1, delimiter=",")
    d2 = np.loadtxt(
        "data/qeb_decay/schmidt_trunc_lifetime_17.txt", skiprows=1, delimiter=","
    )
    cs = ["purple", "green"]
    # my fitting results
    d1 = np.load("data/qeb_decay/21-site_admix-processed.npy", allow_pickle=True).item()
    for i, (d, c) in enumerate(zip([d1], cs)):
        # unpack columns into variables,
        # convert to raw data by exponentiating base 10

        # x, y, dy = d.T
        if i == 0:
            keys = ["popts", "perrs", "errs", "ts", "ys_list"]
            popts, perrs, errs, ts, ys_list = [d[k] for k in keys]
            marker = "s"
            x = errs[2:]
            y = popts[2:, 1]
            dy = perrs[2:, 1]
            print(x)
            print(y)

        m, b = np.polyfit(np.log10(x), np.log10(y), deg=1)
        m = -1.3
        b = 2.5

        def admix_func(x):
            return 10 ** b * x ** m

        print("admix m, b", m, b)

        # plotting
        xs = np.linspace(x[0], x[-1], 10)
        ax.plot(xs, admix_func(xs), c=c, lw=2)
        ax.errorbar(
            x,
            y,
            yerr=dy,
            marker=marker,
            ms=4,
            ls="none",
            mec="k",
            ecolor="k",
            mfc="none",
            label="",
        )
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlabel(r"Admixture, $\varepsilon~(\%)$")
        ax.set_ylabel(r"Lifetime, $\tau$")
        ax.text(
            0.51, 0.61, f"slope: \n{round(m,2)}", fontsize=9, transform=ax.transAxes
        )
        #ax.yaxis.set_label_coords(-0.2, 0.47)
        #ax.xaxis.set_label_coords(0.45, -0.12)

        xticks = [0.5, 1.5, 5]
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"${x}$" for x in xticks])
        yticks = [50, 150, 500]
        ax.set_ylim(top=1600)
        # ax.set_ylim(top=150)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"${y}$" for y in yticks])
        ax.minorticks_off()


if __name__ == "__main__":
    # spectrum_statistics([13, 14, 1], IC="c3_f1", axs=None, intermediate_plots=True)
    # spectrum_statistics([13, 14, 1, 6], IC="R123", axs=None, intermediate_plots=True)

    plot_fname = "figures/figure4/figure4_V8.pdf"

    fig = plt.figure(figsize=(3.375, 4.125))
    gs1 = gridspec.GridSpec(2, 1)
    gs1.update(left=0.22, right=0.95, bottom=0.5, top=0.92, hspace=0.15)

    gs2 = gridspec.GridSpec(1, 2)
    gs2.update(left=0.13, right=0.98, bottom=0.11, top=0.35, wspace=0.65)

    Dscenter_ax = fig.add_subplot(gs1[1, 0])
    brody_ax = fig.add_subplot(gs1[0, 0])
    page_ax = fig.add_subplot(gs2[0, 0])
    breather_ax = fig.add_subplot(gs2[0, 1])

    brody_ax.text(0.01, 0.75, letters[0], transform=brody_ax.transAxes)
    Dscenter_ax.text(0.01, 0.75, letters[1], transform=Dscenter_ax.transAxes)
    page_ax.text(0.03, 0.85, letters[2], transform=page_ax.transAxes)
    breather_ax.text(0.03, 0.85, letters[3], transform=breather_ax.transAxes)

    Skey = [13, 14, 1, 6]
    Dscenter_timeseries(Skey, ax=Dscenter_ax)
    spectrum_statistics([13, 14, 1], axs=brody_ax, intermediate_plots=True)
    simple_page(Skey, ax=page_ax)
    QEB_plot(ax=breather_ax)
    multipage(plot_fname, clip=False)
    print("plot saved to ", plot_fname)
    # plot("figures/figure4/pagecurves_V1.pdf")
    # plt.close("all")
    # simple_page("figures/figure4/pagecurves-simple_V1.pdf")

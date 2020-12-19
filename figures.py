import os
import glob
import h5py
import numpy as np
from copy import copy
import networkx as nx
from scipy.special import gamma
from scipy.optimize import curve_fit
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_pdf import PdfPages


def ket(x):
    return r"$\vert " + x + r"\rangle$"


def exp(x):
    return r"$\langle " + x + r"\rangle$"


names = {
    0: r"$T_{0}$",
    1: r"$T_{1}$",
    6: r"$T_{6}$",
    13: r"$T_{13}$",
    14: r"$T_{14}$",
    4: r"$F_{4}$",
    26: r"$F_{26}$",
    "c1_f0": ket("010"),
    "R": ket("R"),
    "R123": ket("R"),
    "c3_f1": ket("010"),
    "exp_Z": exp("\hat{\sigma}^{z}_j"),
    "exp_X": exp("\hat{\sigma}^{x}_j"),
    "s_2": r"$s_j$",
    "X": r"XEB",
    "Hx": r"$H_x$",

    "sbipart_2": r"$s^{\mathrm{bond}(2)}_{\ell}$",
    "sbipart_2avg": r"$\overline{s}^{\mathrm{bond}(2)}_{\ell}$",
    "sbisect_2avg": r"$\overline{s}^{\mathrm{bond}(2)}_{L/2}$",
    "sbisect_2": r"$s^{\mathrm{bond}(2)}_{L/2}$",
    "Dsbisect_2": r"$\Delta s^{\mathrm{bond}(2)}_{L/2}$",
    "Dsbisect_2avg": r"$\overline{\Delta s}^{\mathrm{bond}(2)}_{L/2}$",

    "sbipart_1": r"$s^{\mathrm{bond}(1)}_{\ell}$",
    "sbipart_1avg": r"$\overline{s}^{\mathrm{bond}(1)}_{\ell}$",
    "sbisect_1avg": r"$\overline{s}^{\mathrm{bond}(1)}_{L/2}$",
    "sbisect_1": r"$s^{\mathrm{bond}(1)}_{L/2}$",
    "Dsbisect_1": r"$\Delta s^{\mathrm{bond}(1)}_{L/2}$",
    "Dsbisect_1avg": r"$\overline{\Delta s}^{\mathrm{bond}(1)}_{L/2}$",


    "sbipart": r"$s^{\mathrm{bond}}_{\ell}$",
    "sbipartavg": r"$\overline{s}^{\mathrm{bond}}_{\ell}$",
    "sbisectavg": r"$\overline{s}^{\mathrm{bond}}_{L/2}$",
    "sbisect": r"$s^{\mathrm{bond}}_{L/2}$",
    "Dsbisect": r"$\Delta s^{\mathrm{bond}}_{L/2}$",
    "Dsbisectavg": r"$\overline{\Delta s}^{\mathrm{bond}}_{L/2}$",

    "Cavg": r"$\overline{\mathcal{C}}$",
    "Davg": r"$\overline{\mathcal{D}}$",
    "Yavg": r"$\overline{\mathcal{Y}}$",
    "C_2avg": r"$\overline{\mathcal{C}}$",
    "D_2avg": r"$\overline{\mathcal{D}}$",
    "Y_2avg": r"$\overline{\mathcal{Y}}$",
    "C_1avg": r"$\overline{\mathcal{C}}^{(1)}$",
    "D_1avg": r"$\overline{\mathcal{D}}^{(1)}$",
    "Y_1avg": r"$\overline{\mathcal{Y}}^{(1)}$",

    "C": r"$\mathcal{C}$",
    "D": r"$\mathcal{D}$",
    "Y": r"$\mathcal{Y}$",
    "DY": "$\Delta \mathcal{Y}$",
    "DYavg": r"$\overline{\Delta \mathcal{Y}}$",
    "C_2": r"$\mathcal{C}$",
    "D_2": r"$\mathcal{D}$",
    "Y_2": r"$\mathcal{Y}$",
    "C_1": r"$\mathcal{C}^{(1)}$",
    "D_1": r"$\mathcal{D}^{(1)}$",
    "Y_1": r"$\mathcal{Y}^{(1)}$",

    "time": r"Time $t$",
    "size": r"Size $L$",
    "site": r"Site $j$",
    "cut":  r"Cut $\ell$"
}

colors = {6: "crimson",
          1: "darkturquoise",
          14: "darkorange",
          13: "olivedrab",
          4: "darkgoldenrod",
          26: "olivedrab",
          "R": "k"}

markers = {"G": "s", "W": "*", "R": "x", "C": "d"}
lines = {"H": "-", "HP_45": "--"}
letters = [r"$\mathrm{\bf{(%s)}}$" % lett for lett in "abcdefghijklmnopqrstuvwxyz"]
#letters = [r"$\bf{%s}$" % lett for lett in "abcdefghijklmnopqrstuvwxyz".upper()]


def lettering(ax, x, y, num):
    ax.text(x, y, letters[num], weight="bold", fontsize=9, transform=ax.transAxes,
            horizontalalignment="center", verticalalignment="center")


def multipage(fname, figs=None, clf=True, dpi=300, clip=True, extra_artist=None):
    """
    Save multi-page pdfs. One page per matplotlib figure object.
    """
    pp = PdfPages(fname)
    if figs is None:
        figs = [plt.figure(fignum) for fignum in plt.get_fignums()]
    for fig in figs:
        if clip is True:
            bbox_inches = "tight"
        else:
            bbox_inches = None
        fig.savefig(
            pp,
            format="pdf",
            bbox_inches=bbox_inches,
            dpi=dpi,
        )
        if clf == True:
            fig.clf()
    pp.close()


def shade_color(color, amount=0.5):
    """
    Lightens/darkens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> shade_color('g', 0.3)
    >> shade_color('#F034A3', 0.6)
    >> shade_color((.3,.55,.1), 0.5)
    """
    import colorsys

    try:
        c = mcolors.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mcolors.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def select(T, L, S, IC, V, BC, v=None, master="master"):
    maxoverT = False
    if T is None:
        T = "*"
        maxoverT = True

    name = f"L{L}_T{T}_V{V}_r1_S{S}_M2_IC{IC}_BC{BC}"
    if v is not None:
        name += f"_v{v}"
    name += ".hdf5"
    data_dir_glob = f"/home/lhillber/documents/research/cellular_automata/qeca/qops/qca_output/{master}/data/{name}"
    # print(data_dir_glob)
    sims = [
        dict(
            L=int(os.path.basename(f).split("L")[1].split("T")[0][:-1]),
            T=int(os.path.basename(f).split("T")[1].split("V")[0][:-1]),
            V=os.path.basename(f).split("V")[1].split("r")[0][:-1],
            r=int(os.path.basename(f).split("r")[1].split("S")[0][:-1]),
            S=int(os.path.basename(f).split("S")[1].split("M")[0][:-1]),
            M=int(os.path.basename(f).split("M")[1].split("IC")[0][:-1]),
            IC=os.path.basename(f).split("IC")[1].split("BC")[0][:-1],
            BC=os.path.basename(f).split("BC")[1].split(".")[0],
            h5file=h5py.File(f, "r"),
        )
        for f in glob.glob(data_dir_glob)
    ]
    if len(sims) == 0:
        print("No sim:", name)
    if maxoverT:
        sim = sims[np.argmax(np.array([s["T"] for s in sims]))]
    else:
        sim = sims[0]

    namefound = "L{}_T{}_V{}_r1_S{}_M2_IC{}_BC{}.hdf5".format(
        *[sim[k] for k in ["L", "T", "V", "S", "IC", "BC"]]
    )
    return sim


def cmap_discretize(cmap, N):
    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)
    colors_i = np.concatenate((np.linspace(0, 1.0, N), (0.0, 0.0, 0.0, 0.0)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1.0, N + 1)
    cdict = {}
    for ki, key in enumerate(("red", "green", "blue")):
        cdict[key] = [
            (indices[i], colors_rgba[i - 1, ki], colors_rgba[i, ki])
            for i in range(N + 1)
        ]
    # Return colormap object.
    return mcolors.LinearSegmentedColormap(cmap.name + "_%d" % N, cdict, 1024)


def colorbar(label, ncolors, cmap):
    cmap = cmap_discretize(cmap, ncolors)
    mappable = cm.ScalarMappable(cmap=cmap)
    mappable.set_array([])
    ax_divider = make_axes_locatable(plt.gca())
    cax = ax_divider.append_axes("right", size="7%", pad="8%")
    colorbar = plt.colorbar(mappable, cax=cax)
    colorbar.set_label(label, rotation=0, y=0.55, labelpad=1.8)
    return cax, colorbar


def brody_func(x, eta):
    b = (gamma((eta + 2) / (eta + 1))) ** (eta + 1.0)
    return b * (eta + 1.0) * x ** eta * np.exp(-b * x ** (eta + 1.0))


def exp_fit(x, y):
    m, b = np.polyfit(x, np.log(y), 1)

    def func(x):
        return np.e**(b + m * x)
    return func, m, b


def exp2_fit(x, y):
    m, b = np.polyfit(x, np.log2(y), 1)

    def func(x):
        return 2**(b + m * x)
    return func, m, b


def powerlaw_fit(x, y):
    m, b = np.polyfit(np.log10(x), np.log10(y), 1)

    def func(x):
        return 10**b * x**m
    return func, m, b


def brody_fit(x, n, eta0=1.0):

    popt, pcov = curve_fit(brody_func, x, n, p0=[eta0], bounds=[0, 1])

    def func(x):
        return brody_func(x, *popt)

    return func, popt, pcov


def page_fit(sba, sbd):
    L = len(sba) + 1
    ells = np.arange(L - 1)

    def page_func(ell, a, logK):
        return (ell + 1) * np.log2(a) - np.log2(1 + a ** (-L + 2 * (ell + 1))) + logK

    popt, pcov = curve_fit(page_func, ells, sba,
                           sigma=sbd, absolute_sigma=True,
                        bounds=[(1e-15, -np.inf), (np.inf, np.inf)])

    def func(ell):
        return page_func(ell, *popt)

    return func, popt, pcov


def moving_average(a, n=3):
    return np.convolve(a, np.ones((n,)) / n, mode="valid")


coeffs = [
    [1.0 / 2],
    [2.0 / 3, -1.0 / 12],
    [3.0 / 4, -3.0 / 20, 1.0 / 60],
    [4.0 / 5, -1.0 / 5, 4.0 / 105, -1.0 / 280],
]


def firstdiff(d, acc, dx):
    assert acc in [2, 4, 6, 8]
    dd = np.sum(
        np.array(
            [
                (
                    coeffs[acc // 2 - 1][k - 1] * d[k * 2:]
                    - coeffs[acc // 2 - 1][k - 1] * d[: -k * 2]
                )[acc // 2 - k: len(d) - (acc // 2 + k)]
                / dx
                for k in range(1, acc // 2 + 1)
            ]
        ),
        axis=0,
    )
    return dd


def grid_animation(Qs, meas, cmap="inferno", nrows=1, tmin=0, tmax=60, vmin=None,
                   vmax=None, label="R", figsize=None):
    meas, arg = meas.split("_")

    if arg.isnumeric():
        arg = int(arg)

    # automate lower bound of color scale for entropy and expectation values
    if vmin is None:
        if meas == "s":
            vmin = 0
        elif meas == "exp":
            vmin = -1
    if vmax is None:
        vmax = 1

    # generate data
    data_list = [Q.to2d(getattr(Q, meas)(arg)) for Q in Qs]

    # initialize subplots
    ncols = int(np.ceil(len(Qs) / nrows))
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows == 1 and ncols == 1:
        axs = [[axs]]
    elif nrows == 1:
        axs = [axs]
    elif ncols == 1:
        axs = [[ax] for ax in axs]

    for i, Q in enumerate(Qs):
        j, k = np.unravel_index(i, (nrows, ncols))
        axs[j][k].imshow(data_list[i][0], vmin=vmin, vmax=vmax, cmap=cmap)
        axs[j][k].set_title(f"{label}={getattr(Q,label)}")

    for iprime in range(nrows * ncols - (i + 1)):
        idx = nrows * ncols - 1 - iprime
        j, k = np.unravel_index(idx, (nrows, ncols))
        axs[j][k].axis("off")
    plt.tight_layout()

    # figure update function
    def update(t):
        t += tmin
        fig.suptitle(f"t={t}")
        for i, data in enumerate(data_list):
            j, k = np.unravel_index(i, (nrows, ncols))
            axs[j][k].imshow(data[t], vmin=vmin, vmax=vmax, cmap=cmap)

    # generate animation
    anim = animation.FuncAnimation(fig, update, frames=tmax, fargs=())
    plt.close()
    return anim


# Network movies
def draw_MI(M_orig, ax, layout="spring", pos=None):
    M = copy(M_orig)
    M[np.abs(M) < 1e-6] = 0.0
    M[np.abs(M) < np.median(M)] = 0.0
    G = nx.from_numpy_matrix(M)
    if pos is None:
        if layout == "spring":
            pos = nx.spring_layout(G, k=1, iterations=200)
        elif layout == "bipartite":
            pos = nx.bipartite_layout(G, nodes=range(len(M) // 2))
        elif layout == "circular":
            pos = nx.circular_layout(G)
        elif layout == "spectral":
            pos = nx.spectral_layout(G)
        elif layout == "spiral":
            pos = nx.spiral_layout(G)
        elif layout == "shell":
            pos = nx.shell_layout(G)
        elif layout == "planar":
            pos = nx.planar_layout(G)
        elif layout == "fr":
            pos = nx.fruchterman_reingold_layout(G)
        elif layout == "kk":
            pos = nx.kamada_kawai_layout(G)
        if layout.split("_")[0] == "grid":
            try:
                Ly, Lx = [Li for Li in map(
                    int, layout.split("_")[1].split("-"))]
                ys = np.linspace(-1, 1, Ly)
                xs = np.linspace(-1, 1, Lx)
            except IndexError:
                ys = np.linspace(-1, 1, int(np.ceil(np.sqrt(len(M)))))
                xs = ys
            pos = {}
            i = 0
            for y in ys[::-1]:
                for x in xs:
                    if i < len(M):
                        pos[i] = np.array([x, y])
                    i += 1

    try:
        edges, weights = zip(*nx.get_edge_attributes(G, "weight").items())
    except ValueError:
        edges = [(0, 1)]
        weights = [1e-8]
    ws = np.array([w for w in weights])
    mx = max(ws)
    mn = min(ws)
    if mx != mn:
        ws = (ws - mn) / (mx - mn)
        ws *= 1.5
    nx.draw(
        G,
        pos,
        ax=ax,
        node_color="purple",
        node_size=9,
        alphs=0.5,
        edgelist=edges,
        #edge_color=ws,
        #edge_cmap=plt.cm.,
        edge_color="k",
        width=ws,
    )
    ax.collections[0].set_facecolor("none")
    ax.collections[0].set_edgecolor("k")
    ax.set_aspect("equal")


def network_animation(Qs, order=2, layout="grid", tmin=1, tmax=60, label="R"):
    # initialize subplots
    fig, axs = plt.subplots(2, len(Qs))
    data_list = [Q.MI(order) for Q in Qs]
    if len(Qs) == 1:
        axs = np.array([axs]).T
    layouts = []
    for i, Q in enumerate(Qs):
        if layout == "grid":
            layouts.append(f"grid_{Q.Ly}-{Q.Lx}")

        draw_MI(data_list[i][1], axs[0, i], layout=layouts[i])
        axs[1, i].imshow(data_list[i][tmin])
        axs[1, i].set_title(f"{label}={getattr(Q,label)}")
    plt.tight_layout()
    # figure update function

    def update(t):
        t += tmin
        [ax.clear() for ax in axs[0, :]]
        fig.suptitle(f"t={t}")
        for i, data in enumerate(data_list):
            draw_MI(data[t], axs[0, i], layout=layouts[i])
            axs[1, i].imshow(data[t])
    # generate animation
    anim = animation.FuncAnimation(fig, update, frames=tmax)
    plt.close()
    return anim

# Network measures


def network_measures_plot(Qs, axs=None, Cref=None, Yref=None, order=2, label="R", sublabel="", reflabel="PT", tmin=1, tmax=-1, logC=False, logY=False, logt=False, **plot_kwrgs):
    if axs is None:
        fig, axs = plt.subplots(2, 1, sharex=True)
    else:
        fig = plt.gcf()
    for Q in Qs:
        axs[0].plot(Q.ts[tmin:tmax], Q.C(order)[tmin:tmax], **plot_kwrgs)
        axs[0].set_ylabel(r"$\mathcal{C}$")
        axs[1].plot(Q.ts[tmin:tmax], Q.Y(order)[tmin:tmax],
                    label=f"{label}= {getattr(Q, label)} " + sublabel, **plot_kwrgs)
        axs[1].set_ylabel(r"$\mathcal{Y}$")
        axs[1].set_xlabel(r"Time, $t$")
    if Cref is not None:
        axs[0].axhline(Cref, label=reflabel, c="k")
    if Yref is not None:
        axs[1].axhline(Yref, label=reflabel, c="k")
    if logC:
        axs[0].set_yscale("log")
    if logY:
        axs[1].set_yscale("log")
    if logt:
        axs[0].set_xscale("log")
        axs[1].set_xscale("log")
    axs[1].legend(loc="center", bbox_to_anchor=(1.15, 1.15))
    return fig, axs

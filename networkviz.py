from matplotlib import animation
from copy import copy
import numpy as np
import matplotlib.pyplot as plt
from figures import ket, colors, markers, multipage
import matplotlib as mpl
from qca import QCA
import networkx as nx
from states import make_state
import measures as ms

from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from matplotlib import rc
rc("text", usetex=True)
font = {"size": 12, "weight": "normal"}
mpl.rc(*("font",), **font)
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["text.latex.preamble"] = [
    r"\usepackage{amsmath}" r"\usepackage{helvet}",  # set the normal font here
    r"\usepackage{sansmath}",  # load up the sansmath so that math -> helvet
    r"\sansmath",  # <- tricky! -- gotta actually tell tex to use it!
]

rc("text", usetex=True)


def draw_MI(M_orig, ax, layout="spring", pos=None):
    M = copy(M_orig)
    M[np.abs(M) < 1e-5] = 0.0
    M[np.abs(M) < np.median(M)] = 0.0
    G = nx.from_numpy_matrix(M)
    if pos is None:
        if layout == "spring":
            pos = nx.spring_layout(G, k=0.8 / np.sqrt(len(M)),
                                   iterations=74)
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
        if layout == "grid":
            Ggrid = nx.grid_2d_graph(*M.shape)
            xs = np.linspace(-1, 1, int(np.ceil(np.sqrt(len(M)))))
            pos = {}
            i = 0
            for x in xs:
                for y in xs:
                    if i < len(M):
                        pos[i] = np.array([x, y])
                    i += 1

    edges, weights = zip(*nx.get_edge_attributes(G, "weight").items())
    ws = np.array([w for w in weights])
    mx = max(ws)
    mn = min(ws)
    if mx != mn:
        ws = (ws - mn) / (mx - mn)
    nx.draw(
        G,
        pos,
        ax=ax,
        node_color="k",
        node_size=6,
        alphs=0.5,
        edgelist=edges,
        edge_color="k",
        width=ws,
    )
    ax.set_aspect("equal")

def netviz(
    Skey,
    L=19,
    t=250,
    statestrs=["R123", "W", "GHZ", "C19-1_45"],
    layout="spring",
    IC="c3_f1",
    V="H",
    BC="1-00",
    T=1000,
    axs=None,
    order=1,
):
    legend_elements = []
    if axs is None:
        fig, axs = plt.subplots(1, 8, figsize=(4.75, 0.8))
    for col, S in enumerate(Skey):
        Q = QCA(dict(L=L, T=T, dt=1, R=S, r=1, V=V, IC=IC, BC=BC, E=0, N=1,
                totalistic=False, hamiltonian=False, trotter=True, symmetric=False))
        Ms = Q.get_measure(f"MI_{order}", save=True)
        M = Ms[t]
        draw_MI(M, axs[col], layout=layout)
        # axs[col].set_title(r"$T_{%d}$" % S, pad=-0.3)
        legend_elements += [
            Patch(facecolor=colors[S], edgecolor=None, label=r"$T_{%d}$" % S)
        ]
    for col, statestr in enumerate(statestrs):
        state = make_state(L, statestr)

        rhoj = ms.get_rhoj(state)
        rhojk = ms.get_rhojk(state)
        s1 = ms.get_entropy(rhoj, order)
        s2 = ms.get_entropy2(rhojk, order)
        M = ms.get_MI(s1, s2)
        draw_MI(M, axs[len(Skey) + col], layout=layout)
        if statestr[0] in ["C", "R", "G"]:
            label = ket(statestr[0])
        else:
            label = ket(statestr)
        m = markers[statestr[0]]
        if m in ("x", "."):
            facecolor = "k"
        else:
            facecolor = "none"
        legend_elements += [
            Line2D(
                [0],
                [0],
                marker=m,
                color="k",
                markersize=7,
                markeredgecolor="k",
                markerfacecolor=facecolor,
                label=label,
                linestyle='None',
            )
        ]
        C = ms.network_clustering(M)
        Y = ms.network_disparity(M)
        print(statestr, " C:", C, " Y:", Y)

    axs[4].legend(
        handles=legend_elements,
        ncol=(len(Skey) + len(statestrs)),
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(-0.1, 0.1),
        markerfirst=False,
        handlelength=0.71,
        handletextpad=0.1,
        columnspacing=2.7,
    )
    # fig.subplots_adjust(wspace=0.4, top=0.7, left=0.05, right=1, bottom=0)


def update(num, Ms, fig, axs, layout):
    [ax.clear() for ax in axs]
    axs[0].imshow(Ms[num])
    draw_MI(Ms[num], axs[1], layout=layout, pos=None)
    fig.suptitle(f"t={1+num}")
    axs[1].set_xlim([-1, 1])
    axs[1].set_ylim([-1, 1])


def netmovie(Q, order=2, layout="spring", tmax=30):
    Ms = Q.MI(order=order)[1:]
    tmax = min(tmax, len(Ms))
    fig, axs = plt.subplots(1, 2)
    ani = animation.FuncAnimation(
        fig, update, frames=tmax-1, fargs=(Ms, fig, axs, layout))
    return ani


def main():
    Skey = [1, 6, 13, 14]
    layout = "spring"
    t = 250
    plot_fname = f"figures/figure3/t{t}_{layout}_layout_medclip.pdf"
    netviz(t=t, Skey=Skey, layout=layout, order=1)
    print(plot_fname)
    multipage(plot_fname, clip=True)


if __name__ == "__main__":
    netmovie()

import numpy as np
from qca import QCA
import matplotlib.pyplot as plt
from classical import iterate, Crule
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Patch
from figures import letters, names
import matplotlib.gridspec as gridspec
from PIL import Image


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


def ket(x):
    return "$\\vert " + x + "\\rangle$"


def exp(x):
    return "$\\langle " + x + "\\rangle$"


defaults = {
    "Ls": [],
    "Ts": [],
    "Vs": [],
    "rs": [],
    "Ss": [],
    "Ms": [],
    "ICs": [],
    "BCs": [],
    "tasks": ["s-2", "exp-z"],
    "sub_dir": "default",
    "thread_as": "zip",
}

L = 19
T = 1000
r = 1
Rkey = [1]
#Rkey = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
V = "H"
M = 2
BC = "1-00"
IC = "c3_f1"

for R in Rkey:
    plt.close("all")
    # classical simulation
    Rc = Crule(R)
    ICc = np.zeros(L, dtype=np.int32)
    ICc[(L - 1) // 2] = 1
    C = iterate(L, T, Rc, ICc, BC)
    Q = QCA(dict(L=L, T=1000, dt=1, R=R, r=1, V=V, IC=IC, BC=BC, E=0, N=1,
            totalistic=False, hamiltonian=False, trotter=True, symmetric=False))
    # Q.check_repo(test=False)
    z = Q.exp("Z")
    s = Q.s(2)
    Q.close()
    T = (L - 1) * 3 + 1  # plot ylim

    # spin
    w, h = mpl.figure.figaspect(z[0:T])
    #fig, axs = plt.subplots(1, 3, figsize=(2.25, 2), sharey=True)

    fig = plt.figure(figsize=(3.375, 3.7))
    gs0 = gridspec.GridSpec(2, 3)
    gs1 = gridspec.GridSpec(2, 3)
    gs0.update(top=0.92, bottom=0.12, left=0.13, right=0.8, wspace=0.0)
    gs1.update(left=0.01, right=0.98, bottom=0.0, top=1)

    ax0 = fig.add_subplot(gs0[0, 0])
    ax1 = fig.add_subplot(gs0[0, 1])
    ax2 = fig.add_subplot(gs0[0, 2])
    ax3 = fig.add_subplot(gs1[1, :])
    axs = [ax0, ax1, ax2]

    ax = axs[1]
    im1 = ax.imshow(
        z[0:T],
        interpolation=None,
        origin="lower",
        cmap="inferno_r",
        vmin=-1,
        vmax=1,
    )
    ticks = [-1, 1]
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="11%", pad=0.05)
    # cax.text(
    #    1.3,
    #    0.5,
    #    names["exp-z"]["name"],
    #    rotation=0,
    #    transform=ax.transAxes,
    #    ha="left",
    #    va="center",
    # )
    cax.axis("off")
    #cbar = fig.colorbar(im1, cax=cax, ticks=ticks)
    # cbar.set_ticks(ticks)
    #
    #ax.set_xlabel("$j$", labelpad=0)
    ax.set_xticks([0, (L - 1) // 2, L - 1])
    ax.set_xticklabels([])

    # ax.set_yticks([i*(L-1) for i in range(4)])
    ax.set_yticklabels([])
    # ax.set_ylabel("$t$", labelpad=0)
    # ax.set_title("$R = %d $" % R)
    ax.text(
        1.15, 1.1, "$T_{%d} $" % R, transform=ax.transAxes, ha="center", va="center"
    )

    # entropy
    ax = axs[2]
    im2 = ax.imshow(
        s[0:T],
        interpolation=None,
        origin="lower",
        cmap="inferno",
        vmin=0,
        vmax=1,
    )
    ticks = [0, 1]
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="11%", pad=0.05)
    # cax.text(
    #    1.3,
    #    0.5,
    #    names["s-2"]["name"],
    #    rotation=0,
    #    transform=ax.transAxes,
    #    ha="left",
    #    va="center",
    # )

    cbar = fig.colorbar(im2, cax=cax, ticks=ticks)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(["↑" + ", " + str(ticks[0]),
                        "↓" + ", " + str(ticks[1])])

    ax.set_yticklabels([])
    ax.set_xticks([0, (L - 1) // 2, L - 1])
    ax.set_xticklabels([])
    #ax.set_xlabel("$j$", labelpad=0)
    # ax.set_ylabel("$t$", labelpad=0)
    # ax.set_title("$R = %d $" % R)
    cax.text(
        1.4,
        0.5,
        names["exp_Z"] + ", " + names["s_2"],
        rotation=0,
        transform=ax.transAxes,
        ha="left",
        va="center",
    )

    # classical
    ax = axs[0]
    im3 = ax.imshow(
        C[0:T],
        interpolation=None,
        origin="lower",
        cmap="Greys",
        vmin=0,
        vmax=1,
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="11%", pad=0.05)
    cax.axis("off")

    legend_elements = [
        Patch(facecolor="k", edgecolor="k", label="1"),
        Patch(facecolor="w", edgecolor="k", label="0"),
    ]
    ylim = ax.get_ylim()
    ax.plot(
        [0], [1000], linestyle="none", marker="s", markeredgecolor="k", c="k", label="1"
    )
    ax.plot(
        [0], [1000], linestyle="none", marker="s", markeredgecolor="k", c="w", label="0"
    )
    # ax.legend(
    #    # handles=legend_elements,
    #    bbox_to_anchor=(2.2, 0.76),
    #    handlelength=0.7,
    #    markerscale=1.2,
    #    frameon=False,
    # )
    ax.set_ylim(ylim)
    ax.set_xticks([0, (L - 1) // 2, L - 1])

    ax.set_yticks([i * (L - 1) for i in range(4)])

    ax.set_yticklabels(["$" + str(i * (L - 1)) + "$" for i in range(4)])

    ax.set_xlabel("Site $j$", labelpad=1)
    ax.set_ylabel("Time $t$", labelpad=2)
    ax.text(
        0.5,
        1.1,
        "$C_{%d} $" % Rc,
        transform=ax.transAxes,
        ha="center",
        va="center",
    )
    #plt.subplots_adjust(left=0.155, right=0.86, wspace=0.06)
    # zoom=0.5
    #w, h = fig.get_size_inches()
    #fig.set_size_inches(w * zoom, h * zoom)
    img = Image.open("figures/figure1/figure1_d-e.png")
    ax3.imshow(np.asarray(img))
    ax3.axis("off")
    axs[0].text(0.1, 50, letters[0], color="w", weight="bold")
    axs[1].text(0.1, 50, letters[1], color="w", weight="bold")
    axs[2].text(0.1, 50, letters[2], color="w", weight="bold")
    plt.savefig("figures/figure1/figure1_V7.pdf", dpi=800)

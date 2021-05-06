# classical.py
#
# Classical elementary cellular automata are simulated
# and their correlation network properties are studeied.
#
#
# By Logan Hillberry


# import modules
import numpy as np
from numpy.linalg import matrix_power
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
from matplotlib import rc
from figures import letters
import matplotlib as mpl
from PIL import Image
import matplotlib.gridspec as gridspec


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


# Convert QECA rule number to ECA rule number
# All those magic numbers! ;)
# 204 is the identity rule; The ^ is bitwise exclusive or,
# so 204^R changes from the interpretation of bit from
# "final state" to "needs a bit flip operation".
# The sum and values in the array use local invertability
# symmetries to recount rules.
def Crule(Qrule):
    a = [5, 5, 20, 20]
    return 204 ^ sum(a[i] * (Qrule & (1 << i)) for i in range(4))


# ECA transition funcion
def ecaf(R, Ni):
    """ R: classical ECA rule number
        Ni: 3-site neighborhood of site i
        returns: the next state of site i
    """
    # neighborhood state as a decimal number
    k = sum(j << i for i, j in enumerate(Ni[::-1]))
    # 1<<k = 2^k is the neighborhood state as a power of two.
    # If that power of two is in the binary expansion
    # of the rule number the next center state is 1.
    # Equivalently the bitwise-and (&) of the rule number
    # with the neighborhood's power of 2 is non zero:
    if R & (1 << k):  # not equal to zero
        return 1
    # otherwise the next center state is 0
    else:  # equal to zero
        return 0


#def ecaf(R, Ni):
#    k = sum(j << i for i, j in enumerate(Ni[::-1]))
#    if R & (1 << k):
#        return 1
#    else:
#        return 0
#
#C = np.zeros(T, L)
#C[0, L//2] = 1
#for t in range(1, T):
#    for j in range(0, L):
#        C[t, j] = ecaf(R, C[t-1, j-1 : j+2])


# ECA time evolution
def iterate(L, T, R, IC, BC):
    """ L: Number of sites
        T: Number of time steps
        R: Classical rule number
        IC: Initial condition data (1d array of length L)
        BC: boundary conditions: "0" for periodic or "1-00"
            for boundaries fixed to 0.
            (also valid BC: "1-01", "1-10", "1-11")
        returns: C, the space time evolution of the automata
        Assumes boundaries fixed to 0.

    """
    BC_type, *BC_conf = BC.split("-")
    if BC_type == "1":
        BC_conf = BC_conf[0]
        L += 2
        C = np.zeros((T, L), dtype=np.int32)
        C[0, 1:-1] = IC
        C[:, 0] = int(BC_conf[0])
        C[:, -1] = int(BC_conf[1])

        def oldN(C, j, t, L):
            return C[t - 1, j - 1 : j + 2]

    elif BC_type == "0":
        C = np.zeros((T, L), dtype=np.int32)
        C[0, :] = IC

        def oldN(C, j, t, L):
            return [C[t - 1, (j - 1) % L], C[t - 1, j], C[t - 1, (j + 1) % L]]

    for t in range(1, T):
        for j in range(0, L):
            C[t, j] = ecaf(R, oldN(C, j, t, L))
    return C[:, 1:-1]


# save multipage pdfs
def multipage(fname, figs=None, clf=True, dpi=300, clip=True, extra_artist=False):
    pp = PdfPages(fname)
    if figs is None:
        figs = [plt.figure(fignum) for fignum in plt.get_fignums()]
    for fig in figs:
        if clip is True:
            fig.savefig(
                pp, format="pdf", bbox_inches="tight", bbox_extra_artist=extra_artist
            )
        else:
            fig.savefig(pp, format="pdf", bbox_extra_artist=extra_artist)
        if clf == True:
            fig.clf()

    pp.close()


# copy/paste of network definitions
# reproduced here for convenience
def network_density(mat):
    l = len(mat)
    lsq = l * (l - 1)
    return sum(sum(mat)) / lsq


def network_clustering(mat):
    l = len(mat)
    matsq = matrix_power(mat, 2)
    matcube = matrix_power(mat, 3)
    for i in range(len(matsq)):
        matsq[i][i] = 0
    denominator = sum(sum(matsq))
    numerator = np.trace(matcube)
    if numerator == 0.0:
        return 0.0
    return numerator / denominator


def network_disparity(mat, eps=1e-17j):
    numerator = np.sum(mat ** 2, axis=1)
    denominator = (np.sum(mat, axis=1)) ** 2
    return (1 / len(mat) * sum(numerator / (denominator + eps))).real


def compare(L=100, T=500):
    for R in range(16):
        fig, axs = plt.subplots(1, 2)
        Rc = Crule(R)
        IC = np.zeros(L)
        IC[L // 2] = 1
        C = iterate(L, T, Rc, IC, BC="1-00")

        # calculate classical MI
        M = np.zeros((L, L))
        for i in range(L):
            for j in range(L):
                M[i, i] = 0.0
                M[i, j] = np.sum(C[:, i] * C[:, j]) / T

        # plot the evolution and the MI
        axs[0].imshow(C, origin="lower", interpolation="none", cmap="Greys")
        axs[1].imshow(M, interpolation="none")
        fig.suptitle("$R_{\mathrm{classical}} = %d$; $R = %d$" % (Rc, R))

        # Round network measures for better display
        ND = network_density(M)
        CC = network_clustering(M)
        Y = network_disparity(M)
        if ND is np.inf:
            ND = "inf"
        if CC is np.inf:
            CC = "inf"
        if Y is np.inf:
            Y = "inf"

        # report network measure values
        axs[1].text(
            1.1, 0.5, "$\mathcal{D} = %s$" % round(ND, 3), transform=axs[1].transAxes
        )
        axs[1].text(
            1.1, 0.4, "$\mathcal{C} = %s$" % round(CC, 3), transform=axs[1].transAxes
        )
        axs[1].text(
            1.1, 0.3, "$\mathcal{Y} = %s$" % round(Y, 3), transform=axs[1].transAxes
        )

    plt.subplots_adjust(right=0.9)
    multipage("figures/classical_MI.pdf", clip=True)


def main():
    fig, axs = plt.subplots(64, 4, figsize=(4, 64))
    L = 39
    T = L
    for R in range(256, T=2 * 35):
        # fig, ax = plt.subplots(1, 1, figsize=(1, 1))
        print(R)
        r, c = int(R // 4), R % 4
        ax = axs[r, c]
        IC = np.zeros((T, L))
        IC[L // 2] = 1
        C = iterate(L, T, R, IC, BC="0")
        ax.imshow(C, interpolation="none", rasterized=False, cmap="Greys")
        ax.set_axis_off()
        ax.set_title(f"R={R}")
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    multipage("figures/classical/classical_ECA.pdf", clip=False, dpi=512)


def plot(Rs=[30, 90, 110], L=65, T=65 // 2):

    fig = plt.figure(figsize=(3.375, 2.5))
    gs0 = gridspec.GridSpec(2, 3)
    gs1 = gridspec.GridSpec(2, 3)
    gs0.update(wspace=0.1, top=1, bottom=0.18, left=0.2, right=0.98)
    gs1.update(left=0.07, right=1.1, bottom=0.1, top=1.01)

    ax0 = fig.add_subplot(gs0[0, 0])
    ax1 = fig.add_subplot(gs0[0, 1])
    ax2 = fig.add_subplot(gs0[0, 2])
    ax3 = fig.add_subplot(gs1[1, :])
    axs = [ax0, ax1, ax2]

    IC = np.zeros(L)
    IC[L // 2] = 1
    letts = letters[:3]
    for i, (R, lett, ax) in enumerate(zip(Rs, letts, axs)):
        C = iterate(L, T, R, IC, BC="1-00")
        ax.imshow(C, cmap="Greys", origin="lower", interpolation="None")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("$C_{%s}$" % R)
        ax.text(0.01, 0.15, lett, transform=ax.transAxes, weight="bold")
        xticks = [0, L // 2, L - 1]
        ax.set_xticks(xticks)
        ax.set_xticklabels([])
        if i == 0:
            yticks = [0, 16, 32]
            ax.set_yticks(yticks)
            ax.set_xticklabels(xticks)
            ax.set_yticklabels(yticks)
            ax.set_xlabel("Site $j$")
            ax.set_ylabel("Time $t$")

    legend_elements = [
        Patch(facecolor="w", edgecolor="k", label="0,"),
        Patch(facecolor="k", edgecolor="k", label="1"),
    ]

    ax.legend(
        handles=legend_elements,
        loc="upper right",
        handlelength=0.8,
        handletextpad=0.5,
        frameon=False,
        bbox_to_anchor=[1.2, 0.1],
        ncol=2,
        columnspacing=0.5,
    )

    im = Image.open("figures/classical/classical_rule_expansion.png")

    ax3.imshow(np.asarray(im))
    ax3.axis("off")
    ax3.text(0.01, 0.87, letters[3], transform=ax3.transAxes)
    ax3.text(0.03, 0.59, "$C_{30}$", transform=ax3.transAxes)
    ax3.text(0.03, 0.37, "$C_{90}$", transform=ax3.transAxes)
    ax3.text(0.03, 0.15, "$C_{110}$", transform=ax3.transAxes)

    plt.savefig("figures/classical/C30_90_110_V2.pdf",
                dpi=700, bbox_inches="tight")




def plot_mm21(Rs=[90, 30, 110], L=513, T=1+513 // 2):

    fig, axs = plt.subplots(1, 3, figsize=(6, 1.5))

    IC = np.zeros(L)
    IC[L // 2] = 1
    letts = letters[:3]
    for i, (R, lett, ax) in enumerate(zip(Rs, letts, axs)):
        C = iterate(L, T, R, IC, BC="1-00")
        ax.imshow(C, cmap="Greys", origin="lower", interpolation="None")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("$C_{%s}$" % R)
        xticks = [0, L // 2, L - 1]
        yticks = [0, L//4, L//2]
        ax.set_xticks(xticks)
        ax.set_xticklabels([])
        ax.set_yticks(yticks)
        ax.set_yticklabels([])
        if i == 0:
            ax.set_xticklabels(xticks)
            ax.set_yticklabels(yticks)
            ax.set_xlabel("Site $j$")
            ax.set_ylabel("Time $t$")

    plt.savefig("../march_meeting_2021/C30_90_110.png",
                dpi=1800, bbox_inches="tight")


# default behavior of this script
if __name__ == "__main__":

    plot_mm21()

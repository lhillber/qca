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

# plotting defaults
mpl.rcParams["text.latex.preamble"] = ["\\usepackage{amsmath}"]
font = {"size": 9, "weight": "normal"}
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["font.family"] = "STIXGeneral"
mpl.rcParams["pdf.fonttype"] = 42
mpl.rc(*("font",), **font)

# lookup table for classical to quantum rule number
# lut[quantum] = classical
lut = {
    0: 204,
    1: 201,
    2: 198,
    3: 195,
    4: 156,
    5: 153,
    6: 150,
    7: 147,
    8: 108,
    9: 105,
    10: 102,
    11: 99,
    12: 60,
    13: 57,
    14: 54,
    15: 51,
}

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
            return C[t-1, j-1:j+2]

    elif BC_type == "0":
        C = np.zeros((T, L), dtype=np.int32)
        C[0, :] = IC

        def oldN(C, j, t, L):
            return [C[t-1, (j-1) % L], C[t-1, j], C[t-1, (j+1) % L]]

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
        Rc = lut[R]
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


# default behavior of this script
if __name__ == "__main__":
    fig, axs = plt.subplots(64, 4, figsize=(4, 64))
    L = 39
    T = L
    for R in range(256):
        #fig, ax = plt.subplots(1, 1, figsize=(1, 1))
        print(R)
        r, c = int(R//4), R % 4
        ax = axs[r, c]
        IC = np.zeros(L)
        IC[L // 2] = 1
        C = iterate(L, T, R, IC, BC="0")
        ax.imshow(C, interpolation="none", rasterized=False, cmap="Greys")
        ax.set_axis_off()
        ax.set_title(f"R={R}")
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    multipage("figures/classical_ECA.pdf", clip=False, dpi=512)

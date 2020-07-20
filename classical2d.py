# classical2d.py
#
# Two-dimensional Classical elementary cellular automata are simulated
#
# By Logan Hillberry


# import modules
import numpy as np
from numpy.linalg import matrix_power
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.animation as animation
from matplotlib.patches import Patch
from matplotlib import rc
from figures import letters
import matplotlib as mpl
from PIL import Image
import matplotlib.gridspec as gridspec


rc("text", usetex=True)
font = {"size": 12, "weight": "normal"}
mpl.rc(*("font",), **font)
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["text.latex.preamble"] = [
    r"\usepackage{amsmath}",
    r"\usepackage{sansmath}",  # sanserif math
    r"\sansmath",
]

def make_mask(j, k, Ly, Lx, r, BC_type):
    mask = [True]*2*r*2
    if Ly == 1:
        for i in range(2*r):
            mask[i] = False
    if Lx == 1:
        for i in range(2*r, 4*r):
            mask[i] = False

    if BC_type == "1":
        if j-r < 0:
            for i in range(r-j):
                mask[i] = False
        if j+r > Ly-1:
            for i in range(r, 2*r+j-Ly+1):
                mask[i] = False

        if k-r < 0:
            for i in range(2*r, 3*r-k):
                mask[i] = False

        if k+r > Lx-1:
            for i in range(3*r, 4*r+k-Lx+1):
                mask[i] = False
    return mask


def central(j, k, Ly, Lx):
    return np.ravel_multi_index([j, k], (Ly, Lx))


def index_arr(j, k, r):
    # U, D, L, R
    return np.vstack((
        np.r_[np.arange(j-r, j), np.arange(j+1, j+r+1), np.ones(2*r)*j],
        np.r_[np.ones(2*r)*k, np.arange(k-r, k), np.arange(k+1, k+r+1)]
        )).astype(int)


def neighbors(j, k, Ly, Lx, r, BC_type):
    mask = make_mask(j, k, Ly, Lx, r, BC_type)
    index = index_arr(j, k, r)
    Njk = np.ravel_multi_index(index, (Ly, Lx), mode="wrap")
    return Njk[mask]


def neighborhood(j, k, Ly, Lx, r, BC_type):
    return np.r_[central(j, k, Ly, Lx), neighbors(j, k, Ly, Lx, r, BC_type)]


# ECA transition funcion
def ecaf(R, Ni, c):
    s = np.sum(Ni)
    if R & (1 << s):  # not equal to zero, then flip
        return (c+1)%2
    # otherwise the next center state is 0
    else:  # equal to zero, then remain
        return c


# ECA time evolution
def iterate(Ly, Lx, T, R, IC, BC):
    """ Ly: Number of rows
        Lx: Nuber of columns
        T: Number of time steps
        R: Totalistic rule
        IC: Initial condition data (1d array of length L)
        BC: boundary conditions: "0" for periodic or "1-0000"
            for boundaries fixed to 0.
            (also valid BC: "1-1111", "1-1010", etc.)
        returns: C, the space time evolution of the automata
    """
    BC_type, *BC_conf = BC.split("-")
    if BC_type == "1":
        shift = 1
        BC_conf = BC_conf[0]
        Ly += 2
        Lx += 2
        C = np.zeros((T, Ly, Lx), dtype=np.int32)
        C[0, 1:-1, 1:-1] = IC
        C[:, 0, :] = int(BC_conf[0])
        C[:, -1, :] = int(BC_conf[1])
        C[:, :, 0] = int(BC_conf[2])
        C[:, :, -1] = int(BC_conf[3])
    elif BC_type == "0":
        shift = 0
        C = np.zeros((T, Ly, Lx), dtype=np.int32)
        C[0, :] = IC

    for t in range(1, T):
        for s in range(1+1):
            for j in range(shift, Ly-shift):
                for k in range(shift, Lx-shift):
                    if (j+k) % (1+1) == s:
                        rs, cs = index_arr(j, k, 1)
                        if s == 0:
                            C[t, j, k] = ecaf(R, C[t-1, rs%Ly, cs%Lx], C[t-1, j, k])
                        else:
                            C[t, j, k] = ecaf(R, C[t, rs%Ly, cs%Lx], C[t-1, j, k])


    if BC_type == "1":
        return C[:, 1:-1, 1:-1]

    elif BC_type == "0":
        return C


# default behavior of this script
if __name__ == "__main__":

    Ly = 75
    Lx = 75
    T = 30
    Rs = [2, 6, 10, 18]
    IC = np.zeros((Ly, Lx))
    IC[Ly//2, Lx//2] = 1
    BC = "1-0000"

    fig, axs = plt.subplots(2, 2)

    Cgrids = np.zeros((len(Rs), T, Ly, Lx))

    ims = []
    for ll, R in enumerate(Rs):
        lj, lk = np.unravel_index(ll, (2, 2))
        Cgrids[ll] = iterate(Ly, Lx, T, R, IC, BC)
        im = axs[lj, lk].imshow(Cgrids[ll, 0], vmin=0, vmax=1, cmap="inferno_r")
        axs[lj, lk].axis("off")
        axs[lj, lk].set_title(f"R={R}")
        ims.append(im)

    def updatefig(i):
        fig.suptitle(f"t={i}")
        for ll, R in enumerate(Rs):
            z = Cgrids[ll, i]
            ims[ll].set_array(z)
        return ims

    ani = animation.FuncAnimation(fig, updatefig, frames=range(T),
                                  interval=200, blit=False)
    #plt.show()
    ani.save('figures/animation/2D_R2-6-10-18_small.mp4')


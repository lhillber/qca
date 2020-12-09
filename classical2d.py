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

tri_dict = {
        1: [0, 2],
        2: [1, 2],
        3: [1, 3],
        4: [0, 3]}

clip_dict = {
        1: [1, 3],
        2: [0, 3],
        3: [0, 2],
        4: [1, 2]}


def make_mask(j, k, Ly, Lx, r, BC_type, tri=0):
    mask = np.array([True]*2*r*2)
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

    if tri in [1, 2, 3, 4]:
        mask[tri_dict[tri]] = False
    return mask


def central(j, k, Ly, Lx):
    return np.ravel_multi_index([j, k], (Ly, Lx))


def index_arr(j, k, r):
    # U, D, L, R
    return np.vstack((
        np.r_[np.arange(j-r, j), np.arange(j+1, j+r+1), np.ones(2*r)*j],
        np.r_[np.ones(2*r)*k, np.arange(k-r, k), np.arange(k+1, k+r+1)]
        )).astype(int)


# ECA transition funcion
def ecaf(R, Ni, c):
    s = np.sum(Ni)
    if R & (1 << s):  # not equal to zero, then flip
        return (c+1)%2
    # otherwise the next center state is unchanged
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

    tri_map = {(0, 0): [1, 3],
               (0, 1): [4, 2],
               (1, 0): [2, 4],
               (1, 1): [3, 1]}

    for t in range(1, T-1, 2):
        for Pi, P in enumerate([1, -1]):
            #blocks = [0, 3, 1, 2, 3, 1, 2, 0]
            blocks = [0, 3, 1, 2]*2

            bjs, bks = np.unravel_index(blocks, (2, 2))

            C[t+Pi, :] = C[t+Pi-1, :]
            bjs = [(j+Pi) % 2 for j in bjs]
            bks = [(k+Pi) % 2 for k in bks]
            Ny = int(np.ceil((Ly+Pi) / 2))
            Nx = int(np.ceil((Lx+Pi) / 2))
            for J in range(Ny):
                for K in range(Nx):
                    for jj, kk in zip(bjs, bks):
                        j = 2*J + P * jj
                        k = 2*K + P * kk
                        if shift <= j < Ly-shift and shift <= k < Lx-shift:
                            tri = tri_map[jj, kk][Pi]
                            mask = make_mask(j, k, Ly, Lx, 1, BC_type, tri=tri)
                            rs, cs = index_arr(j, k, 1)
                            rs = np.array(rs[mask])
                            cs = np.array(cs[mask])
                            # add oposite corner to neighborhood
                            #rs = np.r_[rs, rs.sum()-j]
                            #cs = np.r_[cs, cs.sum() - k]
                            C[t+Pi, j, k] = ecaf(R, C[t+Pi, rs%Ly, cs%Lx], C[t+Pi, j, k])

    if BC_type == "1":
        return C[:, 1:-1, 1:-1]

    elif BC_type == "0":
        return C


# default behavior of this script
if __name__ == "__main__":

    Ly = 51
    Lx = 51
    T = 500
    Rs = [1, 2, 3,  6]
    IC = np.zeros((Ly, Lx))
    IC[0,0] = 1
    IC[0,Lx-1] = 1
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
    ani.save('figures/animation/2D_R1-2-3-6_block-xx.mp4')


from figures import names
from qca import QCA
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib import rc

from figures import letters

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


L = 17
dt = 0.1
fig = plt.figure(figsize=(3.375, 5.2))

titles = [r"$T_1$", "Ryd.", r"$T_6$", "Ryd.", r"$F_4$", "Ryd."]
# titles = [r"$\tilde{T}_1$", "Ryd.", r"$\tilde{T}_6$", "Ryd.", r"$F_4$", "Ryd."]
idx = 0
for row, rule in enumerate(["T_1", "T_6", "F_4"]):
    cbar_mode = "single"
    grid = ImageGrid(
        fig,
        int("31" + str(1 + row)),
        nrows_ncols=(1, 2),
        direction="column",
        axes_pad=0.1,
        label_mode="L",
        cbar_mode=cbar_mode,
        cbar_location="right",
        cbar_size="15%",
        cbar_pad=0.05,
    )

    for subcol, type in enumerate(["QCA", "Rydberg"]):
        ax = grid[subcol]
        lett = letters[idx]
        title = titles[idx]
        idx += 1
        ax.text(0.07, 0.84, lett, transform=ax.transAxes, weight="bold", color="w")
        ax.set_title(title)
        raw = np.loadtxt(f"data/rydberg/{type}_{rule}.dat")
        expZ = raw[:, 2]
        expZ = expZ.reshape(expZ.size // L, L)[:: int(1 / dt)]
        cmap = "inferno"
        # if type == "QCA":
        #    if rule[0] == "T":
        #        Q = QCA(dict(L=L, T=1000, dt=1, R=int(rule[-1]), r=1, V="H", IC="c3_f1", BC="1-00", E=0, N=1,
        #            totalistic=False, hamiltonian=False, trotter=True, symmetric=False))
        #        expZ = Q.exp("Z")[:21, :]
        #        cmap = "inferno_r"
        im = ax.imshow(
            expZ, origin="lower", interpolation="none", cmap=cmap, rasterized=True
        )

        ax.set_xticks([0, 8, 16])
        ax.set_yticks([0, 10, 20])
        ax.set_xticklabels([])
        if subcol == 0:
            ax.set_ylabel(r"Time, $t$")
        else:
            cb = plt.colorbar(im, cax=ax.cax)
            cb.set_ticks([-1, 1])
            ax.cax.set_yticklabels([r"$\uparrow$", r"$\downarrow$"])
            ax.cax.text(
                1.3,
                0.5,
                names["exp_Z"],
                rotation=0,
                transform=ax.transAxes,
                ha="left",
                va="center",
            )
            cb.solids.set_edgecolor("face")
        if row == 2 and subcol == 0:
            ax.set_xticklabels([0, 8, 16])
            ax.set_xlabel(r"Site, $j$")
        if row in (0, 1, 2) and subcol == 0:
            ax.set_yticklabels([0, 10, 20])


fig.subplots_adjust(bottom=0.1, left=0.04, top=0.94, hspace=0.4, right=0.97)
plt.savefig("figures/figure5/figure5_V3.pdf", dpi=600)

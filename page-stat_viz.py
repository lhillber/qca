import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.special import gamma
from figures import letters
from matplotlib import rc

fontstyle = {'pdf.fonttype': 42,
'text.usetex': True,
'text.latex.preamble': '\\usepackage{amsfonts}',
'font.family': 'serif',
'axes.labelsize': 9,
'font.size': 9,
'legend.fontsize': 8,
'xtick.labelsize': 9,
'ytick.labelsize': 9}
plt.rcParams.update(fontstyle)
rc("mathtext", default='regular')

def poisson_func(r):
    return 1 / (1+r)**2


def stats_func(r, beta):
    if beta == None:
        return poisson_func(r)
    elif beta == 1:
        Z = 8/27
    elif beta == 2:
        Z = 4*np.pi/(81*np.sqrt(3))
    return 1/Z * (r+r*r)**beta / (1+r+r*r)**(1+3*beta/2)


def page_func(ell, a, logK, L):
    return (ell + 1) * np.log2(a) - np.log2(1 + a ** (-L + 2 * (ell + 1))) + logK


fig, axs = plt.subplots(1, 2, figsize=(3.375, 1.6))

rs = np.linspace(0, 4, 100)
betas = [None, 1, 2]
labels = [r"Poisson", r"GOE", r"GUE"]
colors = ["crimson", "k", "olivedrab"]
for eta, label, c in zip(betas, labels, colors):
    axs[1].plot(rs, stats_func(rs, eta), label=label, c=c)

L = 19
ells = np.linspace(0, L - 2, 100)
aa = [2.0, 1.5, 1.1]
labels = [r"$a=%s$" % a for a in aa]
colors = ["olivedrab", "C4", "crimson"]
for a, c, label in zip(aa, colors, labels):
    p = page_func(ells, a, logK=0, L=L)
    axs[0].plot(ells, p - p[0], label=label, c=c)
    axs[0].plot(ells[:50], np.log2(a) * ells[:50], c=c, ls="--")

# axs[0].legend(loc="upper left", bbox_to_anchor=(0.95, 0.94),
#              handlelength=0.8, handletextpad=0.5, frameon=False)
axs[1].set_ylabel(r"$\mathcal{D}(r)$")
axs[1].set_xlabel(r"Gap ratio $r$")
axs[1].set_xticks([0, 2, 4])
axs[1].set_yticks([0.0, 0.5, 1.0])
axs[1].text(0.8, 0.85, letters[1], transform=axs[1].transAxes)

# axs[1].legend(loc="upper left", bbox_to_anchor=(0.95, 0.94),
#              handlelength=0.8, handletextpad=0.5, frameon=False)
axs[0].set_ylabel(r"$s^{\rm bond}_{\ell}$")
axs[0].set_xlabel(r"Cut $\ell$")
fig.subplots_adjust(left=0.14, right=0.98, top=0.95, bottom=0.3, wspace=0.64)
axs[0].set_xticks([0, 8, 16])
axs[0].set_yticks([0, 3, 6, 9])
axs[0].text(0.8, 0.85, letters[0], transform=axs[0].transAxes)
plt.savefig("figures/page-stats_V1.pdf")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.special import gamma
from figures import letters
from matplotlib import rc

rc("text", usetex=True)
font = {"size": 12, "weight": "normal"}
mpl.rc(*("font",), **font)
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["text.latex.preamble"] = [
    r"\usepackage{amsmath}",
    r"\usepackage{sansmath}",  # sanserif math
    r"\sansmath",
]


def brody_func(x, eta):
    b = (gamma((eta + 2) / (eta + 1))) ** (eta + 1.0)
    return b * (eta + 1.0) * x ** eta * np.exp(-b * x ** (eta + 1.0))


def page_func(ell, a, logK, L):
    return (ell + 1) * np.log2(a) - np.log2(1 + a ** (-L + 2 * (ell + 1))) + logK


fig, axs = plt.subplots(1, 2, figsize=(3.375, 1.6))

ds = np.linspace(0, 4, 100)
etas = [0, 0.5, 1]
labels = [r"$\eta=0$", r"$\eta=1/2$", r"$\eta=1$"]
for eta, label in zip(etas, labels):
    axs[1].plot(ds, brody_func(ds, eta), label=label)

L = 19
ells = np.linspace(0, L - 2, 100)
aa = [2.0, 1.5, 1.1]
labels = [r"$a=%s$" % a for a in aa]
colors = ["C3", "C4", "C5"]
for a, c, label in zip(aa, colors, labels):
    p = page_func(ells, a, logK=0, L=L)
    axs[0].plot(ells, p - p[0], label=label, c=c)
    axs[0].plot(ells[:50], np.log2(a) * ells[:50], c=c, ls="--")

# axs[0].legend(loc="upper left", bbox_to_anchor=(0.95, 0.94),
#              handlelength=0.8, handletextpad=0.5, frameon=False)
axs[1].set_ylabel(r"$\mathcal{D}(d)$")
axs[1].set_xlabel(r"Gap, $d$")
axs[1].set_xticks([0, 2, 4])
axs[1].set_yticks([0.0, 0.5, 1.0])
axs[1].text(0.83, 0.85, letters[1], transform=axs[1].transAxes)

# axs[1].legend(loc="upper left", bbox_to_anchor=(0.95, 0.94),
#              handlelength=0.8, handletextpad=0.5, frameon=False)
axs[0].set_ylabel(r"$s^{\rm bond}_{\ell}$")
axs[0].set_xlabel(r"Cut, $\ell$")
fig.subplots_adjust(left=0.14, right=0.98, top=0.95, bottom=0.3, wspace=0.64)
axs[0].set_xticks([0, 8, 16])
axs[0].set_yticks([0, 3, 6, 9])
axs[0].text(0.83, 0.85, letters[0], transform=axs[0].transAxes)
plt.savefig("figures/brody-page_V3.pdf")

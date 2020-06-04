import matplotlib.pyplot as plt
import numpy as np
from core import evolve
import matplotlib as mpl
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

def convergance(L, T, R, r, V, IC, BC, totalistic):
    """Run and plot convergence study"""
    dts = np.array([1 / 2**n for n in (0, 1, 2, 3, 4, 5, 6, 7)])
    e1s = []
    e2s = []
    for dt in dts:
        gen0 = evolve(L, T, dt, R, r, V, IC, BC, E=0,
                      totalistic=totalistic, hamiltonian=True,
                      symmetric=False, trotter=False)
        gen1 = evolve(L, T, dt, R, r, V, IC, BC, E=0,
                      totalistic=totalistic, hamiltonian=True,
                      symmetric=False, trotter=True)
        gen2 = evolve(L, T, dt, R, r, V, IC, BC, E=0,
                      totalistic=totalistic, hamiltonian=True,
                      symmetric=True, trotter=True)
        for ti, (s0, s1, s2) in enumerate(zip(gen0, gen1, gen2)):
            e1 = 1 - np.abs(np.sum(np.conj(s1) * s0))**2
            e2 = 1 - np.abs(np.sum(np.conj(s2) * s0))**2
        e1s.append(e1)
        e2s.append(e2)
        print(f"dt:{dt}, asymmetric err:{e1}, symmetric err:{e2}")

    e1s = np.array(e1s)
    e2s = np.array(e2s)
    fit1 = np.polyfit(np.log10(dts), np.log10(e1s), deg=1)
    fit2 = np.polyfit(np.log10(dts), np.log10(e2s), deg=1)

    fig, ax = plt.subplots(1, 1, figsize=(3.375, 3))
    def efunc(x, m, b):
        return 10 ** b * x**m

    ax.loglog(dts, efunc(dts, *fit1), c="k", ls="--", lw=2)
    ax.loglog(dts, efunc(dts, *fit2), c="k", ls="--", lw=2)
    ax.loglog(dts, e1s,
        marker="s", ms=6, mec="r", mfc="none", ls="none")
    ax.loglog(dts, e2s,
        marker="o", ms=6, mec="r", mfc="none", ls="none")

    ax.set_xlabel("Time step, dt")
    ax.set_ylabel(r"Error")
    ax.text(0.1, 0.8,f"slope: {round(fit1[0], 2)}",
            transform=ax.transAxes)
    ax.text(0.4, 0.25,f"slope: {round(fit2[0], 2)}",
            transform=ax.transAxes)
    ax.set_xticks([1e-2, 1e-1, 1])
    ax.minorticks_off()
    print(f"slopes:{fit1[0]}, {fit2[0]}")
    fig.subplots_adjust(left=0.2, bottom=0.2, right=0.95, top=0.95)
    plt.savefig("figures/convergence_R6.pdf")




def convergance2(L, T, R, r, V, IC, BC, totalistic):
    """Run and plot convergence study"""
    dts = np.array([1 / 2**n for n in (0,  2,  4, 6)])
    fig, ax = plt.subplots(1, 1, figsize=(2.25,2))
    for dt in dts:
        IC += str(int(1/dt))
        gen0 = evolve(L, T, dt, R, r, V, IC, BC, E=0,
                      totalistic=totalistic, hamiltonian=True,
                      symmetric=True, trotter=True)

        gen1 = evolve(L, T, dt, R, r, V, IC, BC, E=0,
                      totalistic=totalistic, hamiltonian=True,
                      symmetric=True, trotter=True)
        es = []
        ts = []
        for ti, (s0, s1) in enumerate(zip(gen0, gen1)):
            e1 = 1 - np.abs(np.sum(np.conj(s0) * s1))**2
            es.append(e1)
            ts += [dt*ti]
        ax.loglog(ts, es,
            marker="o", ms=4, ls="none", label=dt)
    ax.legend()
    es = np.array(es)
    #fit1 = np.polyfit(np.log10(dts), np.log10(e1s), deg=1)
    def efunc(x, m, b):
        return 10 ** b * x**m
    ax.set_xlabel("Time step, dt")
    ax.set_ylabel(r"Error")
    #ax.text(0.1, 0.86,f"slope: {round(fit1[0], 2)}",
    #        transform=ax.transAxes)
    #print(f"slopes:{fit1[0]}, {fit2[0]}")
    fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)
    plt.savefig("figures/convergence_R6.pdf")


if __name__ == "__main__":
    convergance(L=11, T=1, R=6, r=1,
                V="X", IC="R123", BC="1-00",
                totalistic=False)


import matplotlib.pyplot as plt
import numpy as np
from qca import QCA
from figures import firstdiff, colors, names

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


def measure_Lscaling(
    Skey,
    Lkey=[6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    BC="1-00",
    ICkey=["c3_f1", "R123"],
    V="H",
    T=1000,
    axs=None,
    measures=["C", "Y"],
    line_kwargs=dict(),
    scatter_kwargs=dict()
):
    if axs is None:
        fig, axs = plt.subplots(2, 1, figsize=(1.3, 1.0), sharex=True)

    lta = np.zeros((len(Skey), len(ICkey), len(Lkey), len(measures)))
    dlta = np.zeros((len(Skey), len(ICkey), len(Lkey), len(measures)))
    for i, S in enumerate(Skey):
        for j, IC in enumerate(ICkey):
            for k, L in enumerate(Lkey):
                Q = QCA(
                    dict(
                        L=L,
                        T=T,
                        dt=1,
                        R=S,
                        r=1,
                        V=V,
                        IC=IC,
                        BC=BC,
                        E=0,
                        N=1,
                        totalistic=False,
                        hamiltonian=False,
                        trotter=True,
                        symmetric=False,
                    )
                )
                # Q.check_repo(test=False)
                for m, meas in enumerate(measures):
                    if meas[0] == "D":
                        d = Q.get_measure(meas[1:], save=True)
                        d = np.abs(firstdiff(d, acc=2, dx=1))

                    else:
                        d = Q.get_measure(meas, save=True)
                    d = d[500:]
                    lta[i, j, k, m] = np.mean(d)
                    dlta[i, j, k, m] = np.std(d)
                Q.close()
    for m, measure in enumerate(measures):
        ax = axs[m]
        for i, S in enumerate(Skey):
            c = colors[S]
            for j, IC in enumerate(ICkey):
                y = lta[i, j, :, m]
                dy = dlta[i, j, :, m]
                x = Lkey
                #ax.fill_between(x, y + dy, y - dy, facecolor=c, alpha=0.2)
                if BC[0] == "1":
                    bc = "fixed"
                    edgecolors = c
                    facecolors = c
                elif BC[0] == "0":
                    bc = "periodic"
                    edgecolors = c
                    facecolors = "none"
                ax.scatter(x, y, marker="o", s=15,
                           edgecolor=edgecolors, facecolors=facecolors, label="$T_{%s}$, %s BC" % (S, bc))
                mm, bb = np.polyfit(x, y, 1)
                xs = np.linspace(x[0], x[-1], 100)
                ys = bb + mm * xs
                #label=f"$\lambda = {round(mm, 2)}$"
                ax.plot(xs, ys, c=c, **line_kwargs)
                print(f"{measure} slope ** V={V}, S={S}, BC={BC}: {mm}")
        ax.plot(np.array(x), np.array(x)//2, c="k")
        #ax.scatter(x, r, marker="o", s=8, c="k")
        #ax.fill_between(x, r + dr, r - dr, facecolor="k", alpha=0.2)

        ax.legend(bbox_to_anchor=(1, 1))

        ax.set_xticks([6, 10, 14, 18])
        ax.set_xticklabels([6, 10, 14, 18])
        ax.set_ylabel(names[measure + "avg"])
        ax.set_xlabel(r"System size, $L$")


fig, axs = plt.subplots(1, 1, figsize=(3.375, 4))
measure_Lscaling(
    Skey=[1, 6, 13, 14],
    Lkey=[6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    BC="1-00",
    ICkey=["c3_f1"],
    V="H",
    T=1000,
    axs=[axs],
    measures=["sbisect_2"],
)

measure_Lscaling(
    Skey=[1, 6, 13, 14],
    Lkey=[6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    BC="0",
    ICkey=["c3_f1"],
    V="H",
    T=1000,
    axs=[axs],
    measures=["sbisect_2"],
    scatter_kwargs={"facecolor": "none"},
    line_kwargs={"ls": "--"}
)

plt.savefig("figures/Lscaling_BC-conditions.pdf", bbox_inches="tight")

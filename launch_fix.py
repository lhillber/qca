from qca import QCA, main
import numpy as np

Ls = [6, 7, 8]

ICs = [f"r{r}" for r in np.arange(7, 100, 4)]
BCs = ["0", "1-00", "1-11"]
Ns = [1000]
Vs = ["H"]

#ICs = ["c3_f1"]
#Vs = [f"HP_{ph}" for ph in np.arange(0, 180+6, 6)][::10]
#BCs = ["1-00"]
#Ns = [1]

Rs = [1, 6, 13, 14]

params = dict(
    tasks=["rhoj", "rhojk", "bisect"],
    Ls=Ls,
    Ts=[500.0],
    dts=[1.0],
    Rs=Rs,
    rs=[1],
    Vs=Vs,
    ICs=ICs,
    BCs=BCs,
    Es=[0.0],
    Ns=Ns,
    totalistic=False,
    hamiltonian=False,
    trotter=True,
    symmetric=False,
    nprocs=4
)

main(**params)

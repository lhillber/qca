from qca import main
import numpy as np

N = 500
Ls = [7, 9, 13, 15]

Es  = np.array([0.0033, 0.0039, 0.0047, 0.0057, 0.0068, 0.0081, 0.0097, 0.0116,
                0.0139, 0.0166, 0.0199, 0.0238, 0.0285, 0.0341, 0.0408, 0.0488,
                0.0584, 0.0699, 0.0836, 0.1])

params = dict(
    tasks=["rhoj", "rhojk", "bitstring"],
    Ls=Ls,
    Ts=[100.0],
    Rs=[6],
    rs=[1],
    Vs=["H"],
    ICs=["c1_f0"],
    BCs=["1-00"],
    Es=Es,
    Ns=[N],
    totalistic=False,
    hamiltonian=False,
    trotter=True,
    symmetric=False
)

main(**params)

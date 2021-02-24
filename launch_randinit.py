from qca import main
import numpy as np

L = 13
# N = 100
# Rs = [1, 6, 9, 13, 14]
# exprob = 100*np.round(np.linspace(1/L, 1-1/L, 50), 2)
# exprob = exprob.astype(int)

exprob = np.arange(7, 100, 2)
N = 500
Rs = [1, 6, 13, 14]

params = dict(
    tasks=["rhoj", "rhojk"],
    Ls=[L],
    Ts=[100.0],
    Rs=Rs,
    rs=[1],
    Vs=["H"],
    ICs=[f"r{p}" for p in exprob],
    BCs=["0"],
    Es=[0.0],
    Ns=[N],
    totalistic=False,
    hamiltonian=False,
    trotter=True,
    symmetric=False
)

main(**params)

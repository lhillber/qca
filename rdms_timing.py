from states import make_state
from matrix import rdms, rdms2, rdms3
from timeit import timeit

u = make_state(15, "R")

rdms(u, [4])

for rdm in [rdms, rdms2, rdms3]:
    print(rdm)
    print(timeit("rdm(u, [4, 5])", globals=globals(), number=1000))

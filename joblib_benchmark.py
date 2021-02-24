from qca import main
from time import time
import matplotlib.pyplot as plt

times = []
nprocs_list = [2,4,6,8,10,12,14,16,20]
for nprocs in nprocs_list:
    t0 = time()
    main(
        Rs=[1,2,3,4,5,6,7,8,9,10],
        ICs=["R"],
        BCs=["0", "1-00"],
        nprocs=nprocs,
        recalc=True)
    t1 = time()
    times.append(t1-t0)

plt.plot(nprocs_list, times)
plt.xlabel("n parallel jobs")
plt.xlabel("execution time (s)")
plt.show()

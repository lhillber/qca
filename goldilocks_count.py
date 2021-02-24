from matrix import dec_to_bin
from itertools import permutations
import numpy as np

r = 2

Rs = np.arange(0, 2**(1+(2**r)))
for R in Rs:
    R2 = dec_to_bin(R, 1+(2*r))[::-1]
    hoodsum = 0
    for elnum, Rel in enumerate(R2):
        K = elnum*[1] + (2*r-elnum) * [0]
        hoods = list(set([p for p in permutations(K, 2*r)]))
        hoods = map(list, hoods)
        if Rel == 1:
            for hood in hoods:
                hoodsum += 1
    if hoodsum/(2**(2*r)) == 0.5:
        print(R)


#Rs = np.arange(0, 2**(2**(2**r)))
# print(Rs)
# for R in Rs:
#    R2 = dec_to_bin(R, 2**(2*r))
#    if sum(R2)/len(R2) == 0.5:
#        print(R)

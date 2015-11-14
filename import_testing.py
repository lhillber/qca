#!/usr/bin/python3

'''
Here is an example of how to load in and extract QECA simulation results.
The entire set of python files is available at github.com/lhillber/eca.

'''

import fio as io
import numpy as np
import states as ss
import plotting as pt
import matplotlib.pyplot as plt

# import results from the file path/name
results = io.read_results( 
        fname
        ='/home/lhillber/documents/qca/output/block_eca/102_150/data/QX_R102_ICs12_L13_tmax70.res' )
       
# reform single site density matricies with the real and imaginary parts
sd_list = results['sdr'] + 1j*results['sdi'] 

# grab the time series of mutual information networks
mi = results['mi']

# grab the time series of entropy of all bi-partite cuts
ec = results['ec']

# grab the time series of inverse participation ratio 
ipr = results['ipr']

dat = [[np.trace(rj.dot(ss.ops['1'])).real for rj in rj_list] for rj_list in sd_list]

#pt.plot_spacetime_grid(dat,'nz')
pt.plot_spacetime_grid(ec,'entropy of cut')
#plt.show()

print(mi[2])

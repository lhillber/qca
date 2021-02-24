# Merge QCA from different experimental runs but with otherwise identical
# parameters.

from qca import QCA, QCA_from_file
from core1d import save_dict_hdf5
from h5py import File
from copy import copy
import os
import numpy as np

# =============================================================================
# number of experiments to average
Nexperiments = 2
# experiment subdirectory names
experiment_subders = [f"experiment{i}" for i in range(Nexperiments)]
# setting root of all data (this file's directory)
root = os.path.dirname(os.path.abspath(__file__))
# name of base directory containing experiment directories,
# e.g. `data_15-sites`
basename = "test_data"
tasks = ["rhoj", "rhojk", "bisect"]
# =============================================================================


# compose root and basname
baseder = os.path.join(root, basename)
# compose output directory
outder = os.path.join(root, basename + "_merged")
# tasks to average

def find_files(der):
    """Return the full path to all files in directory der."""
    files = []
    # r=root, d=directories, f=files
    for r, ds, fs in os.walk(der):
        for f in fs:
            files.append(os.path.join(r, f))
    return files


# use one of the experiment directories to set up the accumulated simulations
inder = os.path.join(baseder, experiment_subders[0])
fnames_in = find_files(inder)
Qs = np.zeros(len(fnames_in), dtype="object")
for i, fname in enumerate(fnames_in):
    Qin = QCA_from_file(fname, der=inder)
    params_in = Qin.params
    params_out = copy(params_in)
    params_out["N"] = Nexperiments * Qin.N
    # Create and save an empty QCA object
    # to be populated with the accumulated data
    Qout = QCA(params_out, der=outder)
    Qs[i] = Qout

# Accumulate into a result dictionary
for i, fname in enumerate(fnames_in):
    result = {k:0 for k in tasks}
    for experiment_subder in experiment_subders:
        inder = os.path.join(baseder, experiment_subder)
        Qin = QCA_from_file(fname, der=inder)
        for task in tasks:
            result[task] += getattr(Qin, task) / Nexperiments

    # save result to hdf5 file associated with accumalted QCA fname
    with File(Qs[i].fname, "a") as h5file:
        save_dict_hdf5(result, h5file)

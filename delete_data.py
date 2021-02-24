import os
from h5py import File
from qca import QCA_from_file


def find_files(der):
    """Return the full path to all files in directory der."""
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(der):
        for file in f:
            files.append(os.path.join(r, file))
    return files


der = "/home/lhillber/documents/research/cellular_automata/qeca/qca/data/"


fs = find_files(der)
print(len(fs))
c = 0
tmpfname = "data/tmp.hdf5"
for f in fs:
    if f.split(".")[-1] == "hdf5":
        try:
            Q = QCA_from_file(f)
        except (KeyError, OSError):
            continue
        avail = Q.available_tasks
        if Q.V[:2] == "HP":
            if len(Q.V.split("_")[1].split(".")) == 2:
                print(Q.V)
                #os.remove(Q.fname)

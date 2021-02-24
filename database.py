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
        if Q.L not in (17, 18, 19, 20):
            c += 1
            print("enter")
            print(Q.params)
            print(avail)
            L = Q.L
            Qfname = Q.fname
            with File(tmpfname, "w") as tmpfile:
                with File(Qfname, "r") as infile:
                    for k in infile.keys():
                        if "ebipart" not in infile.keys() and "ebipartdata" not in infile.keys():
                            if k == "espec":
                                tmpfile["ebisectdata"] = infile[k][:].real
                            elif k == "ebisect":
                                tmpfile["ebisectdata"] = infile[k][:].real
                        elif k == "ebipart":
                            if "ebipartdata" not in avail:
                                for l in range(L-1):
                                    tmpfile["ebipartdata"+f"/l{l}"] = infile[k][f"l{l}"][:].real
                        elif k == "ebipartdata":
                            for l in range(L-1):
                                tmpfile[k + f"/l{l}"] = infile[k][f"l{l}"][:].real

                        if k == "ebisectdata":
                            tmpfile[k] = infile[k][:].real

                        if k == "bipart":
                            for l in range(L-1):
                                tmpfile[k+f"/l{l}"] = infile[k][f"l{l}"][:]

                        if k not in ("espec", "ebisect", "ebipart", "ebipartdata", "ebisectdata", "bipart"):
                            if k[0] == "s":
                                tmpfile[k] = infile[k][:].real
                            else:
                                tmpfile[k] = infile[k][:].real
                tmpfile.flush()
                print(f"del {Qfname}")
                os.remove(Qfname)
            print(f"rename tmp to {Qfname}")
            os.rename(tmpfname, Qfname)
            with File(Q.fname, "r") as infile:
                print("old", avail)
                print("new", [k for k in infile.keys()])
            #with File(tmpfname, "r") as tmpfile:
            #    print("new", [k for k in tmpfile.keys()])

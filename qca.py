#! /usr/bin/env python
# coding: utf-8
#
# qca.py
#
# by Logan Hillberry
#
#
#  Description:
#  ===========
#  Provides two key functionalities:
#
#  1)
#  Object-Oriented class for interacting with density matrix data saved
#  by simulations. Enables calculation of entropies, expectation values,
#  mutual information, and network measures.
#
#  2)
#  Command line interface and parallelism for quantum cellular automata
#  simulations. The method main() will parse command line arguments, in
#  which single values or lists of values may be given. Handeling of lists
#  of parameters is set by the thread_as variable which may take the values
#  "product", "cycle", "repeat", for generating the a list of complete
#  parameter sets by making the list products, list cycles, or repeating
#  the final values of shorter lists. For example suppose there are three
#  parameters "a", "b", "c", and "d" with respective lists of values
#  [1, 2, 3], [10, 20], [100]. The following parameter sets may be
#  generated:
#
#   thread_as = "product":
#        [{"a":1, "b": 10, "c": 100},
#         {"a":1, "b": 20, "c": 100},
#         {"a":2, "b": 10, "c": 100},
#         {"a":2, "b": 20, "c": 100},
#         {"a":3, "b": 10, "c": 100},
#         {"a":3, "b": 20, "c": 100}
#        ]
#
#   thread_as = "cycle":
#        [{"a":1, "b": 10, "c": 100},
#         {"a":2, "b": 20, "c": 100},
#         {"a":3, "b": 10, "c": 100}
#        ]
#
#   thread_as = "repeat":
#        [{"a":1, "b": 10, "c": 100},
#         {"a":2, "b": 20, "c": 100},
#         {"a":3, "b": 20, "c": 100},
#        ]
#
# Since simulations are independent they can be parallelized. Here this
# is done with the MPI4py library and each process works on a subset of
# simulations in round-robbin ordering. Consider N processors (numbered by
# their "rank" from 0 to N-1) running M simulations (numbered 0 to M-1), then
# rank r will run simulations r, N+r, 2N+r, ... M - r. Before distributing
# work to the processors, the simulation parameter sets are randomly shuffled
# to improve the balance of the work load.
#
# While running, the program estimates the time remaining to completion
# by maintaining an average time taken to complete a simulation of a given
# system size L and computing the average number of simulations of each
# L remaining. In parallel, the time estimate is reduced by a factor of N
# and each rank maintains a different estimate. This estimate is very crude,
# and some times nonsensical
#
#
# Usage:
# =====
#
# From the command line run:
#
#   python3 qca.py -argname [val [val...]]
#
# where -argname is the parameter name you wish to supply and [val [val ...]]
# denotes either a single value or a space separated list of values for that
# parameter. Flag parameters (trotter, symmetric, totalistic, Hamiltonian,
# and recalc) don't accept lists. If a flag is passed, then logical not
# of the default is used. If a parameter is not supplied in the command line,
# a default value is used which is set in the dictionary `defaults`
# defined below. Further, see
#
#   python3 qca.py --help
#
# for more information on parameters.
#
# To run in parallel use:
#
#   mpiexec --oversubscribe -np `N` python3 qca.py -argname [val [val...]]
#
# where N is the number of processors.
#
# To launch simulations from within another script:
#
# import this modules main function
#
#   from qca import main
#
# then provide key word arguments to main() corresponding to the
# desired values of parameters. Missing arguments are filled with the
# defaults.


import os
import argparse
from random import Random
from sys import stdout
from time import time
from datetime import datetime
from copy import copy

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from h5py import File

from matrix import ops as OPS
from matrix import listkron
import measures as ms
from core1d import record, hash_state, save_dict_hdf5, params_list_map
from figures import exp2_fit, names

non_params = ["recalc", "tasks", "nprocs", "thread_as"]

defaults = {
    # parameters
    "L": 15,  # system size, int/list
    "Lx": 1,  # number of columns in system (2d if >1), int/list
    "T": 100.0,  # simulation time, float/list
    "dt": 1.0,  # time step, float/list
    "R": 6,  # rule numbering, int/list
    "r": 1,  # neighborhood radius, int/list
    "V": "H",  # Local update unitary, str/list
    "IC": "c3_f1",  # initial condition, str/list
    "BC": "1-00",  # boundary condition, str/list
    "E": 0.0,  # depolarization error rate, float/list
    "N": 1,  # number of trials to average, int/list
    "trotter": True,  # trotter time ordering?, bool
    "tris": "0",  # triangle evolution sequence
    "blocks": "0",  # block partition update sequence
    "rods": "0",  # rod partition update sequence
    "symmetric": False,  # symmetrized trotter?, bool
    "totalistic": False,  # totalistic rule numbering?, bool
    "hamiltonian": False,  # continuous time evolution?, bool
    # non parameters:
    "recalc": False,  # recalculate tasks?, bool
    "tasks": ["rhoj", "rhojk"],  # save density matricies, list of str
    "nprocs": 1,  # number of processers
    "thread_as": "product",  # Combining lists of above parameters, str
}


parser = argparse.ArgumentParser()


def get_defaults():
    return vars(parser.parse_args([]))


parser.add_argument(
    "-L",
    default=defaults["L"],
    nargs="*",
    type=int,
    help="Number of qubits used in simulation",
)

parser.add_argument(
    "-Lx",
    default=defaults["Lx"],
    nargs="*",
    type=int,
    help="Number of columns (width), 2D if Lx > 1",
)

parser.add_argument(
    "tris",
    default=defaults["tris"],
    nargs="*",
    type=str,
    help="Triangle orientation sequence evolution (for 2d)"
         + "0:+ (default 2d is crosses), or 1: ⌜, 2: ⌞, 3: ⌟, 4: ⌝",
)

parser.add_argument(
    "blocks",
    default=defaults["blocks"],
    nargs="*",
    type=str,
    help="Block partition update squence (2x2 Margolus neighborhood)"
         + "0: No block partition, or 1: upper left, 2: upper right, 3: lower left, 4: lower right",
)

parser.add_argument(
    "rods",
    default=defaults["rods"],
    nargs="*",
    type=str,
    help="Rod orientation sequence evolution (for 2d)"
         + "0:+ (default 2d is crosses), or 1: up, 2: down, 3: left, 4: right",
)

parser.add_argument(
    "-T", default=defaults["T"], nargs="*", type=float, help="Total simulation time"
)

parser.add_argument(
    "-dt", default=defaults["dt"], nargs="*", type=float, help="Simulation time step"
)

parser.add_argument(
    "-R", default=defaults["R"], nargs="*", type=int, help="QCA rule number"
)

parser.add_argument(
    "-r",
    default=defaults["r"],
    nargs="*",
    type=int,
    help="Number of qubits in a neighborhood",
)

parser.add_argument(
    "-V", default=defaults["V"], nargs="*", type=str, help="Activation operator"
)

parser.add_argument(
    "-IC",
    default=defaults["IC"],
    nargs="*",
    type=str,
    help="Initial condition state specification (see states.py)",
)

parser.add_argument(
    "-BC",
    default=defaults["BC"],
    nargs="*",
    type=str,
    help="Boundary treatment: '0' for periodic, 1-<config> for fixed where"
    + " <config> is a string of 1's and 0's of length 2*r specifying the"
    + " fixed boundary state.",
)

parser.add_argument(
    "-E",
    type=float,
    default=defaults["E"],
    nargs="*",
    help="Probibility of an error per qubit upon update of a neighborhood."
    + " enter as a probability/r (between 0 and 1/r)",
)

parser.add_argument(
    "-N",
    default=defaults["N"],
    nargs="*",
    type=int,
    help="Number of versions to simulate"
    + " For averaging over random qunatities, if applicable",
)

parser.add_argument(
    "-thread_as",
    "--thread_as",
    default=defaults["thread_as"],
    type=str,
    choices=["product", "cycle", "repeat"],
    help="Handeling of lists of parameters."
    + "'product' for list products \n"
    + "'cycle' for zipping with cycling shorter lists \n"
    + "'repeat' for zipping with repeating final value in shorter lists",
)

parser.add_argument(
    "-trotter",
    "--trotter",
    action=f"store_{str(not defaults['trotter']).lower()}",
    help="Trotter time ordering? (True if no flag given)",
)

parser.add_argument(
    "-symmetric",
    "--symmetric",
    action=f"store_{str(not defaults['symmetric']).lower()}",
    help="Symmetric Trotter applicationl? (False if no flag given)",
)

parser.add_argument(
    "-totalistic",
    "--totalistic",
    action=f"store_{str(not defaults['totalistic']).lower()}",
    help="Totalistic rule numbering? (False if no flag given)",
)

parser.add_argument(
    "-hamiltonian",
    "--hamiltonian",
    action=f"store_{str(not defaults['hamiltonian']).lower()}",
    help="Hamiltonian (continuous) rule numbering? (False if no flag given)",
)

parser.add_argument(
    "-recalc",
    "--recalc",
    action=f"store_{str(not defaults['recalc']).lower()}",
    help="Recalculate tasks even if available? (False if no flag given)",
)

parser.add_argument(
    "-tasks",
    "--tasks",
    default=defaults["tasks"],
    nargs="*",
    type=str,
    help="Density matrix calculations to be performed."
    + " rhoj -- single site,"
    + " rhojk -- two site,"
    + " bipart -- all bipartitions,"
    + " bisect -- central bipartition"
    + " ebipart -- all bipartitions, entanglement spectrum,"
    + " ebisect -- central bipartition, entanglement spectrum"
)

parser.add_argument(
    "-nprocs",
    "--nprocs",
    default=defaults["nprocs"],
    type=int,
    help="Number of parallel workers running requested simulations."
         +"set to -1 to use all avalable slots,"
         +"set to -2 to use all but one available slots."
)



def QCA_from_file(fname=None, name=None, der=None):
    """ QCA factory function to load a file name into a class.
    fname: full file location
    name: uniqie QCA hash file name
    der: directory from which to load `name`"""
    if fname is None:
        if name is not None:
            if der is None:
                der = os.path.join(os.path.dirname(
                    os.path.abspath(__file__)), "data")
            fname = os.path.join(der, name)

    with File(fname, "r") as h5file:
        params = eval(h5file["params"][0].decode("UTF-8"))
    der = os.path.dirname(fname)
    return QCA(params=params, der=der)


class QCA:
    """Object-Oriented interface"""

    def __init__(self, params=None, der=None):
        if params is None:
            params = {k:v for k, v in defaults.items() if k not in non_params}
        else:
            default_params = {k:v for k, v in defaults.items() if k not in non_params}
            default_params.update(params)
            params = copy(default_params)
        if der is None:
            der = os.path.join(os.path.dirname(
                os.path.abspath(__file__)), "data")
        os.makedirs(der, exist_ok=True)
        params["T"] = float(params["T"])
        params["dt"] = float(params["dt"])
        params["E"] = float(params["E"])
        params["Ly"] = int(params["L"] / params["Lx"])
        if type(params["tris"]) == str:
            tris = [[tri for tri in map(int, tris)] for tris in params["tris"].split("_")]
            if len(tris) != params["r"] + 1:
                tris = [tris[0]] * (params["r"] + 1)
            params["tris"] = tris

        if type(params["rods"]) == str:
            rods = [[rod for rod in map(int, rods)] for rods in params["rods"].split("_")]
            if len(rods) != params["r"] + 1:
                rods = [rods[0]] * (params["r"] + 1)
            params["rods"] = rods

        if type(params["blocks"]) == str:
            blocks = [b for b in map(int, params["blocks"])]
            params["blocks"] = blocks

        reject_keys = copy(non_params)
        if params["Lx"] == 1:
            reject_keys.append("Lx")
            reject_keys.append("Ly")

        if params["tris"] == [[0], [0]]:
            reject_keys.append("tris")

        if params["rods"] == [[0], [0]]:
            reject_keys.append("rods")

        if params["blocks"] == [0]:
            reject_keys.append("blocks")

        self.params = params
        self.reject_keys = reject_keys
        self.der = der
        self.uid = hash_state(self.params, reject_keys=self.reject_keys)
        self.fname = os.path.join(self.der, self.uid) + ".hdf5"

        if "params" not in self.available_tasks:
            with File(self.fname, "a") as h5file:
                h5file.create_dataset("params",
                    data=np.array([str(self.params)], dtype='S'))


    def __getattr__(self, attr):
        """Acess hdf5 data as class atribute"""
        try:
            return self.params[attr]
        except KeyError:
            with File(self.fname, "r") as h5file:
                if attr in ("bipart", "ebipartdata"):
                    return [h5file[attr][f"l{l}"][:] for l in range(self.L - 1)]
                else:
                    return h5file[attr][:]
        # TODO: find a way to give a better error message if data is not available

    def close(self, force=False):
        """Ensure hdf5 file is closed. Delete it if it has no data """
        if "h5file" in self.__dict__:
            self.h5file.close()
        if self.available_tasks == ["params"]:
            print(f"Deleting empty {self.uid}")
            os.remove(self.fname)

    @property
    def ts(self):
        """"Simulation times."""
        return np.arange(0, self.T + self.dt, self.dt)

    @property
    def available_tasks(self):
        """Show available data."""
        try:
            with File(self.fname, "r") as h5file:
                return [k for k in h5file.keys()]
        except OSError:
            return []

    @property
    def file_size(self):
        size = os.path.getsize(self.fname) / 1000000.0
        print(f"file size (MB): {round(size, 2)}")
        return size

    def to2d(self, flat_data, subshape=None):
        """Reshape measures for 2D grids."""
        shape = [len(self.ts), self.Ly, self.Lx]
        if subshape is not None:
            shape += [d for d in subshape]
        return flat_data.reshape(shape)

    def diff(self, x, dt=None, acc=8):
        """First derivative of x at accuracy 8."""
        assert acc in [2, 4, 6, 8]
        if dt is None:
            dt = self.dt
        coeffs = [
            [1.0 / 2],
            [2.0 / 3, -1.0 / 12],
            [3.0 / 4, -3.0 / 20, 1.0 / 60],
            [4.0 / 5, -1.0 / 5, 4.0 / 105, -1.0 / 280],
        ]
        dx = np.sum(
            np.array(
                [
                    (
                        coeffs[acc // 2 - 1][k - 1] * x[k * 2:]
                        - coeffs[acc // 2 - 1][k - 1] * x[: -k * 2]
                    )[acc // 2 - k: len(x) - (acc // 2 + k)]
                    / dt
                    for k in range(1, acc // 2 + 1)
                ]
            ),
            axis=0,
        )
        return dx


    def rolling(self, func, x, winsize=None):
        """Func applied to rolling window of winsize points (default is L/dt)"""
        if winsize is None:
            winsize = int(self.L / self.dt)
        nrows = x.size - winsize + 1
        n = x.strides[0]
        xwin = np.lib.stride_tricks.as_strided(x, shape=(nrows,winsize), strides=(n,n))
        return func(xwin, axis=1)

    def _check_repo(self, test=True):
        """Data transfer method for my old code base."""
        der = "/home/lhillber/documents/research/cellular_automata"
        der = os.path.join(der, "qeca/qops/qca_output/master/data")
        keys = ["L", "T", "V", "r", "R", "IC", "BC"]
        L, T, V, r, R, IC, BC = [self.params[k] for k in keys]
        name = f"L{L}_T{int(T)}_V{V}_r{r}_S{R}_M{2}_IC{IC}_BC{BC}"
        fname = os.path.join(der, name) + ".hdf5"
        key_map = [
            ("one_site", "rhoj"),
            ("two_site", "rhojk"),
            ("cut_twos", "bipart"),
            ("cut_half", "bisect"),
            ("sbond", "sbipart"),
            ("scenter", "sbisect"),
            ("sbond-2", "sbipart_2"),
            ("scenter-2", "sbisect_2"),
            ("sbond-1", "sbipart_1"),
            ("scenter-1", "sbisect_1"),
        ]
        old_h5file = File(fname, "r")
        available_keys = [k for k in old_h5file.keys()]
        for kold, knew in key_map:
            if kold in available_keys:
                print(f"{kold} loaded into {knew}")
                if not test:
                    with File(self.fname, "a") as h5file:
                        try:
                            h5file[knew] = old_h5file[kold][:]
                        except OSError:
                            h5file[knew][:] = old_h5file[kold][:]
                if knew == "bipart":
                    for l in range(self.L-1):
                        if not test:
                            with File(self.fname, "a") as h5file:

                                try:
                                    h5file[knew][f"l{l}"] = old_h5file[kold][f"c{l}"][
                                        :
                                    ]
                                except OSError:
                                    h5file[knew][f"l{l}"][:] = old_h5file[kold][
                                        f"c{l}"
                                    ][:]
        old_h5file.close()


    def check_der(self, der, fname=None, test=True):
        """Data transfer method from der to self.der."""
        if fname is None:
            der_fname = os.path.join(der, self.uid) + ".hdf5"
        try:
            check_h5file = File(der_fname, "r")
        except OSError:
            return

        available_keys = [k for k in check_h5file.keys()]
        for k in available_keys:
            if not test:
                with File(self.fname, "a") as h5file:
                    try:
                        h5file[k] = check_h5file[k][:]
                    except (RuntimeError, OSError):
                        h5file[k][:] = check_h5file[k][:]
                if k == "bipart":
                    for l in range(self.L-1):
                        with File(self.fname, "a") as h5file:
                            try:
                                h5file[k][f"l{l}"] = check_h5file[k][f"l{l}"][:]
                            except (RuntimeError, OSError):
                                h5file[k][f"l{l}"][:] = check_h5file[k][f"l{l}"][:]
        check_h5file.close()


    def run(self, tasks, recalc=False, verbose=True, add_string=""):
        """Run tasks and save data to hdf5 file."""
        t0 = time()
        needed_tasks = [k for k in tasks if k not in self.available_tasks]
        if "ebisectdata" in self.available_tasks and "ebisect" in needed_tasks:
            del needed_tasks[needed_tasks.index("ebisect")]
        if "ebipartdata" in self.available_tasks and "ebipart" in needed_tasks:
            del needed_tasks[needed_tasks.index("ebipart")]
        print_params = {k:v for k,v in self.params.items() if k not in self.reject_keys}
        added_tasks = []
        if recalc:
            if verbose:
                print("Rerunning:")
                print("   ", print_params)
            rec = record(self.params, tasks)
            added_tasks = tasks
        else:
            if len(needed_tasks) > 0:
                if verbose:
                    print("Running:")
                    print("   ", print_params)
                rec = record(self.params, needed_tasks)
                added_tasks = needed_tasks
        if len(added_tasks) > 0:
            with File(self.fname, "a") as h5file:
                save_dict_hdf5(rec, h5file)
                h5file.flush()
            del rec
        t1 = time()
        elapsed = t1 - t0
        if verbose:
            p_string = "\n" + "=" * 80 + "\n"
            p_string += f"{datetime.now().strftime('%d %B %Y, %H:%M:%S')}\n"
            p_string += add_string
            #p_string += "Rank: {}\n".format(self.rank)
            if len(added_tasks) == 0:
                p_string += f"Nothing to add to {self.uid}\n"
            else:
                p_string += f"Updated: {self.uid}\n"
                p_string += f"    with {added_tasks}\n"
            p_string += f"Parameters: {print_params}\n"
            p_string += f"Available data: {self.available_tasks}\n"
            p_string += "Total file size: {:.2f} MB\n".format(
                os.path.getsize(self.fname) / 1000000.0
            )
            p_string += "Took: {:.2f} s\n".format(elapsed)
            p_string += "Data at:\n"
            p_string += self.fname + "\n"
            p_string += "=" * 80 + "\n"
            print(p_string)

    def get_measure(self, meas, save=False, Dmode="std"):
        """Parse string `meas` to compute entropy, expectation values,
           and network measures. `Underscore integer` selects the entropy
           order.  A prepended `D` computes fluctuations based on absolute
           value of first derivative (Dmode=`diff`) or standard deviation
           (Dmode=`std`, default)."""
        func, *args = meas.split("_")

        if func == "exp":
            return getattr(self, func)(*args, save=save)

        elif func in ("C", "D", "Y", "P") or func[0]=="s":
            args = list(map(int, args))
            return getattr(self, func)(*args, save=save)

        elif func[0] == "D":
            if Dmode == "std":
                args = list(map(int, args))
                m = getattr(self, func[1:])(*args, save=save)
                return self.rolling(np.std, m)
            elif Dmode == "diff":
                args = list(map(int, args))
                m = getattr(self, func[1:])(*args, save=save)
                return np.abs(self.diff(m, np.std))

    def get_measures(self, measures, save=False, Dmode="std"):
        """Collect a list of measures"""
        return [self.get_measure(meas, save=save, Dmode=Dmode) for meas in measures]

    def plot(self, meas, tmin=0, tmax=None, stride=1, ax=None, figsize=None, **args):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig = plt.gcf()
        m = self.get_measure(meas)
        ts = self.ts
        if tmax is None:
            tmax = ts[-1]
        mask = np.logical_and(ts>=tmin, ts<=tmax)
        if len(m.shape) == 1:
            ax.plot(ts[mask][::stride], m[mask][::stride], **args)
            try:
                ax.set_ylabel(names[meas])
            except KeyError:
                pass
            ax.set_xlabel(names["time"])
        elif len(m.shape) == 2:
            im = ax.imshow(m[mask,:][::stride],
                origin="lower",
                extent=[-0.5,
                        m.shape[1]-0.5,
                        tmin-self.dt*stride/2,
                        tmax+self.dt*stride/2],
                **args)
            cbar = fig.colorbar(im)
            try:
                cbar.set_label(names[meas])
            except:
                pass
            if m.shape[1] == self.L - 1:
                ax.set_xlabel(names["cut"])
            else:
                ax.set_xlabel(names["site"])
            ax.set_ylabel(names["time"])
        return ax

    def delete_measure(self, key):
        """ Remove a key from the hdf5 file."""
        with File(self.fname, "a") as h5file:
            try:
                del h5file[key]
            except KeyError:
                pass

    def save_measure(self, key, data):
        """Save data to an hdf5 dataset named `key`"""
        avail = self.available_tasks
        with File(self.fname, "a") as h5file:
            if key not in avail:
                h5file[key] = data
            else:
                h5file[key][:] = data
            h5file.flush()


    def s(self, order=1, save=False):
        """Local Renyi entropy"""
        key = f"s_{order}"
        try:
            s = getattr(self, key)
        except KeyError:
            s = np.array([ms.get_entropy(rj, order=order) for rj in self.rhoj])
            setattr(self, key, s)
        if save:
            self.save_measure(key, s)
        return s


    def s2(self, order=1, save=False):
        """Two-site Renyi entropy"""
        key = f"s2_{order}"
        try:
            s2 = getattr(self, key)
        except KeyError:
            s2 = np.array([ms.get_entropy2(rjk, order=order)
                          for rjk in self.rhojk])
            setattr(self, key, s2)
        if save:
            self.save_measure(key, s2)
        return s2


    def exp(self, op, name=None, save=False):
        """Expectation value of local op"""
        if type(op) == str:
            name = op
            op = OPS[op]
        else:
            if name is None:
                raise TypeError("please name your op with the `name` kwarg")
        key = f"exp_{name}"
        try:
            exp = getattr(self, key)
        except KeyError:
            exp = np.array([ms.get_expectation(rj, op) for rj in self.rhoj])
            setattr(self, key, exp)
        if save:
            self.save_measure(key, exp)
        return exp


    def exp2(self, ops, name=None, save=False):
        """Expectation value of two-site op"""
        if type(ops) == str:
            name = ops
            ops = [OPS[op] for op in ops]
        else:
            if name is None:
                raise TypeError("please name your op with the `name` kwarg")
        key = f"exp2_{name}"
        try:
            exp2 = getattr(self, key)
        except KeyError:
            exp2 = np.array([
                ms.get_expectation2(rjk, *ops) for rjk in self.rhojk])
            setattr(self, key, exp2)
        if save:
            self.save_measure(key, exp2)
        return exp2


    def sbipart(self, order=2, save=False):
        """Bipartition Renyi entropy"""
        key = f"sbipart_{order}"
        try:
            sbipart = getattr(self, key)
        except KeyError:
            avail = self.available_tasks
            if "bipart" in avail:
                sbipart = np.transpose(
                    np.array([
                        ms.get_entropy(rhos, order=order)
                            for rhos in self.bipart])
                )
            elif "ebipartdata" in avail:
                sbipart = np.transpose(
                    np.array([
                        ms.get_entropy_from_spectrum(es, order=order)
                            for es in self.ebipartdata])
                )
            else:
                raise KeyError(
                    f"Can not compute sbipart with tasks f{avail}")
            setattr(self, key, sbipart)
        if save:
            self.save_measure(key, sbipart)
        return sbipart


    def sbisect(self, order=2, save=False):
        """Bisection Renyi entropy"""
        key = f"sbisect_{order}"
        try:
            sbisect = getattr(self, key)
        except KeyError:
            avail = self.available_tasks
            if "ebisectdata" in avail:
                sbisect = ms.get_entropy_from_spectrum(
                    self.ebisectdata, order=order)
            elif "ebipartdata" in avail:
                sbisect = ms.get_entropy_from_spectrum(
                    self.ebipartdata[int((self.L - 1) // 2)], order=order)
            elif "bisect" in avail:
                sbisect = ms.get_entropy(self.bisect, order=order)
            elif "bipart" in avail:
                sbisect = ms.get_entropy(
                    self.bipart[int((self.L - 1) // 2)], order=order)
            else:
                raise KeyError(
                    f"Can not compute sbisect with tasks f{avail}")
            setattr(self, key, sbisect)

        if save:
            self.save_measure(key, sbisect)
        return sbisect


    def ebipart(self, save=False):
        """Bipartition entanglement spectrum"""
        key = "ebipartdata"
        try:
            ebipart = getattr(self, key)
        except KeyError:
            ebipart =[ms.get_spectrum(r) for r in self.bipart]
            setattr(self, key, ebisect)

        if save:
            self.save_measure(key, ebipart)
        return ebispart

    def ebisect(self, save=False):
        """Bisection entanglement spectrum"""
        key = "ebisectdata"
        try:
            ebisect = getattr(self, key)
        except KeyError:
            avail = self.available_tasks
            if "bisect" in avail:
                ebisect = ms.get_spectrum(self.bisect)
            elif "bipart" in avail:
                ebisect = ms.get_spectrum(
                    self.bipart[int((self.L - 1) // 2)])
            elif "ebipartdata" in avail:
                ebisect = self.ebipartdata[int((self.L-1)/2)]
            else:
                raise KeyError(
                    f"Can not compute ebisect with tasks f{avail}")
            setattr(self, key, ebisect)

        if save:
            self.save_measure(key, ebisect)
        return ebisect


    def MI(self, order=2, save=False):
        """Mutual information adjacency matrix"""
        key = f"MI_{order}"
        try:
            MI = getattr(self, key)
        except KeyError:
            s = self.s(order)
            s2 = self.s2(order)
            MI = np.array([
                ms.get_MI(sone, stwo) for sone, stwo in zip(s, s2)])
            setattr(self, key, MI)
        if save:
            self.save_measure(key, MI)
        return MI


    def g2(self, ops, name=None, save=False):
        """g2 correlator of ops"""
        if type(ops) == str:
            name = ops
        else:
            if name is None:
                raise TypeError("Please name your ops with the `name` kwarg")
        key = f"g2_{name}"
        try:
            g2 = getattr(self, key)
        except KeyError:
            e1 = self.exp(ops[0])
            e2 = self.exp(ops[1])
            e12 = self.exp2(ops)
            g2 = np.array([
                ms.get_g2(eAB, eA, eB ) for eAB, eA, eB in zip(e12, e1, e2)])
            setattr(self, key,g2)
        if save:
            self.save_measure(key, g2)
        return g2


    def C(self, order=2, save=False):
        """Mutual information clustering coefficient"""
        key = f"C_{order}"
        try:
            C = getattr(self, key)
        except KeyError:
            MI = self.MI(order)
            C = np.array([ms.network_clustering(mi) for mi in MI])
            setattr(self, key, C)
        if save:
            self.save_measure(key, C)
        return C


    def D(self, order=2, save=False):
        """Mutual information density"""
        key = f"D_{order}"
        try:
            D = getattr(self, key)
        except KeyError:
            MI = self.MI(order)
            D = np.array([ms.network_density(mi) for mi in MI])
            setattr(self, key, D)
        if save:
            self.save_measure(key, D)
        return D


    def Y(self, order=2, save=False):
        """Mutual information disparity"""
        key = f"Y_{order}"
        try:
            Y = getattr(self, key)
        except KeyError:
            MI = self.MI(order)
            Y = np.array([ms.network_disparity(mi) for mi in MI])
            setattr(self, key, Y)
        if save:
            self.save_measure(key, Y)
        return Y


    def P(self, order=2, save=False):
        """Mutual information path length"""
        key = f"P_{order}"
        try:
            P = getattr(self, key)
            raise KeyError
        except KeyError:
            MI = self.MI(order)
            P = np.array([ms.network_pathlength(mi) for mi in MI])
            setattr(self, key, P)
        if save:
            self.save_measure(key, P)
        return P


    def H(self, save=False):
        """Bitstring (square-amplitude of state) entropy"""
        key = "Hdata"
        try:
            H = getattr(self, key)
        except KeyError:
            H = np.array([ms.get_bitstring_entropy(ps)
                         for ps in self.bitstring])
            setattr(self, key, H)
        if save:
            self.save_measure(key, H)
        return H


    def F(self, save=False):
        """Bitstring entropy Fidelity"""
        key = "Fdata"
        try:
            F = getattr(self, key)
        except KeyError:
            p1 = copy(self.params)
            p1["N"] = 1
            p1["E"] = 0.0
            Q1 = QCA(p1)
            try:
                qs = Q1.bitstring
            except KeyError:
                raise KeyError("Need corresponding ideal simulation")
            ps = self.bitstring
            F = np.array([ms.get_bitstring_fidelity(p, q) for p, q in zip(ps, qs)])
            setattr(self, key, F)
        if save:
            self.save_measure(key, F)
        return F


    def CF(self, order=1, save=False):
        """
            Clustering scaled by ideal simulation and incoherent state:
            CF = (Cmeasured - Cincoherent ) / (Cexpected - Cincoherent)
            """
        key = f"CF_{order}"
        try:
            CF = getattr(self, key)
        except KeyError:
            p1 = copy(self.params)
            p1["N"] = 1
            p1["E"] = 0.0
            Q1 = QCA(p1)
            try:
                C1s = Q1.C(order)
            except KeyError:
                raise KeyError("Need corresponding ideal simulation")
            C2s = self.C(order)
            CF = np.array([ms.get_clustering_fidelity(C2, C1, self.L) for C1, C2 in zip(C1s, C2s)])
            setattr(self, key, CF)
        if save:
            self.save_measure(key, CF)
        return CF


def to_list(val):
    if type(val) == list:
        return val
    return [val]


def main(
    thread_as=None,
    tasks=None,
    recalc=None,
    Ls=None,
    Lxs=None,
    Ts=None,
    dts=None,
    Rs=None,
    rs=None,
    Vs=None,
    ICs=None,
    BCs=None,
    Es=None,
    Ns=None,
    totalistic=None,
    hamiltonian=None,
    trotter=None,
    symmetric=None,
    nprocs=None,
    der=None
):


    args = parser.parse_args()
    if thread_as is None:
        thread_as = args.thread_as
    if tasks is None:
        tasks = to_list(args.tasks)
    if recalc is None:
        recalc = args.recalc
    if nprocs is None:
        nprocs = args.nprocs
    if Ls is None:
        Ls = to_list(args.L)
    if Lxs is None:
        Lxs = to_list(args.Lx)
    if Ts is None:
        Ts = to_list(args.T)
    if dts is None:
        dts = to_list(args.dt)
    if Rs is None:
        Rs = to_list(args.R)
    if rs is None:
        rs = to_list(args.r)
    if Vs is None:
        Vs = to_list(args.V)
    if ICs is None:
        ICs = to_list(args.IC)
    if BCs is None:
        BCs = to_list(args.BC)
    if Es is None:
        Es = to_list(args.E)
    if Ns is None:
        Ns = to_list(args.N)
    if totalistic is None:
        totalistic = args.totalistic
    if hamiltonian is None:
        hamiltonian = args.hamiltonian
    if trotter is None:
        trotter = args.trotter
    if symmetric is None:
        symmetric = args.symmetric

    # initialize parmeters template
    params = dict(
        totalistic=totalistic,
        hamiltonian=hamiltonian,
        trotter=trotter,
        symmetric=symmetric,
    )

    # make list of params based on list of values
    params_list = params_list_map[thread_as](
        params, Ls, Lxs, Ts, dts, Rs, rs, Vs, ICs, BCs, Es, Ns
    )

    main_from_params_list(params_list, tasks, recalc, der, nprocs)


def remianing_time_projection(Ls, elapsed):
    projection  = np.array(copy(elapsed))
    known_mask = np.array(elapsed) > 0.0
    if sum(known_mask) == 0:
        return projection
    elif sum(known_mask) == 1:
        return [projection[known_mask][0] * (2**i) for i in range(len(projection))]
    unknown_mask = np.logical_not(known_mask)
    func, m, b = exp2_fit(np.array(Ls)[known_mask],
                          np.array(elapsed)[known_mask])
    projection[unknown_mask] = func(np.array(Ls)[unknown_mask])
    return projection


def main_from_params_list2(params_list, tasks, recalc, der=None):
    # sort by small L first
    allL = [p["L"] for p in params_list]
    params_list = [p for _, p in sorted(zip(allL, params_list),
                   key=lambda pair: pair[0])]

    # initialize job meta data
    numsims = len(params_list)  # number of sims in job
    simnum = 1  # current job number

    # initalize parallel communication
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    # what system sizes will each core be running
    allmyLs = [p["L"] for simnum, p in enumerate(params_list)
               if (simnum - 1) % nprocs==rank]
    myLs = list(set(allmyLs))
    # number of sims of the same size
    numperLdict = {L: 0 for L in myLs}
    for L in allmyLs:
        numperLdict[L] += 1
    numperL = [numperLdict[L] for L in myLs]
    visits = [0 for L in myLs]  # number of visits to L
    elapsed = [0 for L in myLs]  # number of visits to L
    estimate = 0  # remaining time

    # run all requested simulations
    for simnum, params in enumerate(params_list):
        if (simnum - 1) % nprocs == rank:
            params.update({"rank": rank, "nprocs": nprocs})
            stdout.write(
                f"Rank {rank}/{nprocs} running simulation {simnum}/{numsims}."
                + f"     (~{round(estimate, 1)} s remaining)     \r"
            )
            stdout.flush()
            ta = time()
            Q = QCA(params, der=der)
            Q.run(tasks, recalc=recalc)
            Q.close()
            tb = time()
            e = tb - ta
            ind = myLs.index(Q.L)
            if e > 0.1:  # only include new sims in average
                visits[ind] += 1
                elapsed[ind] = ((visits[ind] - 1)
                                * elapsed[ind] + e) / visits[ind]
            numremain = np.array(numperL) - np.array(visits)
            projection = remianing_time_projection(myLs, elapsed)
            estimate = numremain.dot(projection)


def work_load(params, tasks, der, recalc, add_string):
    Q = QCA(params, der=der)
    Q.run(tasks, recalc=recalc, add_string=add_string)
    Q.close()


def main_from_params_list(params_list, tasks, recalc, der=None, nprocs=-2):
    allL = [p["L"] for p in params_list]
    params_list = [p for _, p in sorted(zip(allL, params_list),
                   key=lambda pair: pair[0])]
    add_string = f"Running {len(params_list)} job(s) with {nprocs} worker(s)\n"
    Parallel(n_jobs=nprocs)(delayed(work_load)(params, tasks, der, recalc, add_string)
        for params in params_list)


if __name__ == "__main__":
    main()

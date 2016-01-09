#qca

By Logan Hillberry

The project enables the simulation of quantum elementary cellular automata.
Dependencies include: python3, numpy, scipy, matplotlib, mpi4py, and h5py.
Simulations are done in 3

The file run.py contains a dictionary of parameters which specify the simulation.
The keys include:

'output_dir' : name of a directory that will be created where automatically
named data and plot files will be placed (into sub directories called data and
plots respectively). Note, paths are found by assuming a 'documents' folder in
the user's home directory (note the lowercase d).

'fname' :  if this key is provided, 'output_dir' is not required (and thus
ignored) and files are saved to a **full path** provided by fname. This feature isn't currently well tested.

'mode' : specify with a string 'sweep' or 'block'

'V' : a list of strings representing 1-qubit operators which will be multiplied
together to make the local update operation. Note ['X'] corresponds to classical
ECA. Other examples include ['H'] and ['H', 'X', 'T']. The string must be a key
in the ops dictionary provided in the file states.py.

'R' : an integer representing a unitary ECA rule code indexed from 0 to 255
(only 16 of which are unitary)

'S' : (use instead of 'R', never both) an integer enumerating the 16 unitary
ECA's. See time_evolve.py for the correspondence between R and S.

'IC': a string specifying the initial condition. The sting must be
interpretable by the functions in states.py (see that file for examples). Also
accepts a list of tuples to specify global super positions. The tuple contains
an IC string and a weighting coefficient for the superposition. 
The sum of the squares of the weightings must be one. For example 'l0' is a single spin-down
(|1>) at site index zero in an otherwise spin-up lattice, while [(1.0/sqrt(2),
'l0'), (-1.0/sqrt(2), 'l1')] is a singlet between sites 0 and 1 in an otherwise
spin-up lattice.

'L' : an integer specifying the length of the lattice. Must be larger than 3,
and probably smaller than 20 for now. Simulation time scales exponentially with
system size (with a base of about 2.1). Note that the site index j runs from 0
to L-1, 0 being the leftmost qubit.

'T' : an integer specifying the number of iterations through which the eca will
be simulated. Timing scales roughly linearly.

Inside of run.py, the user my specify lists of all of the above parameters which
will then be nested to make a list of simulation dictionaries.
One could also construct a list of simulation dictionaries by zipping together the lists of
each individual parameter; this would require all the lists to be the same
length. Then, each independent simulation is run in parallel up to the number of
cores available on the machine.

To run a simulation, use the following Linux terminal command while in the
directory containing the scripts.

```
exec -np <n> python3 run.py
```
where n is the number of cores available on the machine (8 for my 4 core
hyper-threaded i7).





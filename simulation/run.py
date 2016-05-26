#!/usr/bin/python3

# =============================================================================
# This is an example script for running qca simulations.
#
# params: dictionary specifying a single simulation (make a list of params dics
#       to run sim ulations in parallel)
#   + output_dir: sub directory into which data/plots from this run will save
#   + mode : str, select update patterns for run ('sweep', 'alt', or 'block')
#   + L    : int, lattice sizes for run
#   + T    : int, number of iterations for run
#   + S    : int \in [0,15], S rule numbers for run
#   + V    : str or nparray, center qubit operator; gets conditioned by neighbors according to S
#   + IC   : str or tuple, Initial conditions (ICs) for run (see states.py for more info)
#   + BC   : str Boundary conditions (BCs) for run ('0' for ring '1' for 0...0 box)
#   - optional/alternate params:
#       + R          : int \in [0, 255] R rule number (equal to S for the 16 unitary R)
#       + V_name     : str, label V in the file name, use if V given as a npyarray
#       + init_state : nparray, full quantum state
#       + IC_name    : str, label init_state in the file name
#
# sim_tasks: what to do to the full quantum state
#   +'one_site'  : singe site reduced density matricies (rdm)
#   +'two_site'  : two site pair rdms
#   +'IPT'       : inverse partition ratio (and Fourier transform)
#   +'bi_partite': bi-partition rdms (expensive calculation! only for L<19)
#
# measure_tasks: what to do with the rdms
#   +'s'    : local von Neumann entropy
#   +'sc'   : entropy of bi-partitions (requires bi_partite)
#   +'mom'  : spin moments <AjBk>(t) for A,B \in {x, y, z}^2 (<Aj> stored on
#             diagonal i=j when A==B, which is typical)
#   +'g'    : correlator g2(Aj,Bk;t) = <AjBk> - <Aj><Bk>
#   +'m'    : mutual information adjaceny matrix
#   +'nm'   : network measures
#   +'stats': statistics on time and space averages of g and mom grids
#
# coord_tasks : coordinates selecting A and B in mom and g
#
# nm_tasks: network measures on m:
#   +'ND' : network density
#   +'CC' : clustering coefficient
#   +'Y'  : disparity
#
# corrj: the index at w.r.t whcih the g2 grid is plotted (correlations of the
#        rest of the lattice with site jcorr). use the string jcorr = 'L/2' for the
#        center of the lattic
#
# NOTE: order matters, in measure_tasks. Keep roughly this order (A -> B means B
#       to the right of A in the list):
# s -> m
# m -> nm
# mom -> g
# mom,g -> stats
#
#
# By Logan Hillberry
# =============================================================================

from cmath  import sqrt, sin, cos, pi
import simulation.launch as launch

# lists of parameters to simulate
# -------------------------------

output_dir = 'testing/scenter'

mode_list = ['alt']

L_list = [21]

T_list = [3]

S_list = [1]

V_list = ['HP_0']

IC_list = ['c3_f1']

BC_list = ['1_00']


# flag to make a few default plots 
# (requires 's', 'mom', 'g', 'm', 'nm' and 'stats' to be in measure_tasks)
make_plots = True

# list of tasks to complete
# -------------------------
# tasks on the full quantum states
sim_tasks = ['one_site', 'two_site','sbond', 'scenter']
rewrite_states = True

# tasks on the reduced density matricies ('sc' requires bi_bartite in sim_tasks).
measure_tasks = ['s', 'mom', 'g', 'm', 'nm', 'stats']
rewrite_measures = True

# sub tasks for spin projections ('mom') and g2 ('g')
coord_tasks = ['xx', 'yy', 'zz']

# network measures on mutual info 'm'
nm_tasks = ['ND', 'CC', 'Y']

# site w.r.t. which g2 is plotted (any integer less than L or use string 'L/2')
corrj = 'L/2'

# nest parameter lists
params_list = [
           {
            'output_dir' : output_dir,
            'mode' : mode,
            'L' : L,
            'T' : T,
            'S' : S,
            'V' : V,
            'IC': IC,
            'BC': BC
             }
        for mode in mode_list  \
        for L    in L_list     \
        for T    in T_list     \
        for S    in S_list     \
        for V    in V_list     \
        for IC   in IC_list    \
        for BC   in BC_list    ]


# launch the simulations in parallel
# ----------------------------------
if __name__ == '__main__':
    launch.launch_parallel( params_list,
                            sim_tasks=sim_tasks,
                            rewrite_states = rewrite_states,
                            measure_tasks=measure_tasks,
                            coord_tasks=coord_tasks,
                            nm_tasks=nm_tasks,
                            corrj=corrj,
                            rewrite_measures = rewrite_measures,
                            make_plots = make_plots
                            )

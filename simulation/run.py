#!/usr/bin/python3
# =============================================================================
# order matters, in measure_tasks. Keep roughly this order (A before B = A -> B)
# s -> m
# m -> nm
# mom -> g
# mom,g -> stats
# ==============r==============================================================

from cmath  import sqrt, sin, cos, pi
import simulation.launch as launch


# lists of parameters to simulate
# -------------------------------
output_dir = 'testing'

mode_list = ['sweep']

L_list = [16]

T_list = [60]

S_list = [6]

V_list = ['H']

IC_list = ['f0','f0_t90-p0','f0_t90-p90']

BC_list = ['1']

# list of tasks to complete
#--------------------------
# tasks on the full quantum states
sim_tasks = ['one_site', 'two_site', 'IPR']
rewrite_states = False

# tasks on the reduced density matricies (sc requires bi_bartite in sim_tasks).
measure_tasks = ['s', 'sc', 'mom', 'g', 'm', 'nm', 'stats']
rewrite_measures = False

# sub tasks for spin prjections ('mom') and g2 ('g')
coord_tasks = ['xx', 'yy', 'zz']

# network measures on 'm'
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
if __name__ == '__main__':
    launch.launch_parallel( params_list,
                            sim_tasks=sim_tasks,
                            rewrite_states = rewrite_states,
                            measure_tasks=measure_tasks,
                            coord_tasks=coord_tasks, 
                            nm_tasks=nm_tasks, 
                            corrj=corrj ,
                            rewrite_measures = rewrite_measures
                            )

from django.core.management.base import BaseCommand, CommandError
from qcaadmin.models import *

from mpi4py import MPI
from simulation import time_evolve, measures
import simulation.fio as io


class Command(BaseCommand):
    help = 'Runs a simulation with specified arguments.'

    def add_arguments(self, parser):
        parser.add_argument('R', nargs='+', type=int)
        parser.add_argument('V', nargs='+')
        parser.add_argument('mode', nargs='+') #either 'sweep' or 'block'
        parser.add_argument('IC', nargs='+', type=int)

    def handle(self, *args, **options):
        time = 30

        ic = InitialCondition.objects.get(pk=options["IC"][0])
        result = SimResult(V=options["V"][0],R=options["R"][0],IC=ic,T=time,location="None",completed=False)
        if (options["mode"][0] == "sweep"): result.mode = True
        else: result.mode = False

        result.save()
        pk = result.pk


        params_list = [{
                'output_dir': "automated",
                'mode': options["mode"][0],
                'V': options["V"][0],
                'S': options["R"][0],
                'IC': 'G',
                'L': ic.length,
                'T': time
                }]


        # initialize communication
        comm = MPI.COMM_WORLD
        # get the rank of the processor
        rank = comm.Get_rank()
        # get the number of processsors
        nprocs = comm.Get_size()

        # use rank 0 to give each simulation a file name
        if rank == 0:
            for params in params_list:
                # iterate version numbers for random throw IC's
                if params['IC'][0] == 'r':
                    fname = io.make_file_name(params, iterate = True)
                # don't iterate file names with a unique IC name
                else:
                    fname = io.make_file_name(params, iterate = False)
                # set the file name for each simulation
                params['fname'] = fname


        names = []
        # boradcast updated params list to each core
        params_list = comm.bcast(params_list, root=0)
        for i, params in enumerate(params_list):
            # each core selects params to simulate without the need for a master
            if i % nprocs == rank:
                fname = time_evolve.run_sim(params, force_rewrite=False)
                measures.measure(params, fname, force_rewrite=True)
                names.append(fname)


        result = SimResult.objects.get(pk=pk)
        result.completed=True
        result.location=names[0]
        result.save()


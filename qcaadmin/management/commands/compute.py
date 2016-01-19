from django.core.management.base import BaseCommand, CommandError
from qcaadmin.models import *

from mpi4py import MPI
from simulation import time_evolve, measures
import simulation.fio as io

import json
import numpy as np

class Command(BaseCommand):
    help = 'Runs a simulation with specified arguments.'

    def add_arguments(self, parser):
        parser.add_argument('R', nargs='+', type=int)
        parser.add_argument('V', nargs='+')
        parser.add_argument('mode', nargs='+') #either 'sweep' or 'block'
        parser.add_argument('IC', nargs='+', type=int)
        parser.add_argument('pk', nargs='+', type=int)
        parser.add_argument('time', nargs='+', type=int)

    def handle(self, *args, **options):

        ic = InitialCondition.objects.get(pk=options["IC"][0])

        lookup = {
                204:0,
                201:1,
                198:2,
                195:3,
                156:4,
                153:5,
                150:6,
                147:7,
                108:8,
                105:9,
                102:10,
                99:11,
                60:12,
                157:13,
                57:14,
                51:15
                }

        icparse = json.loads(ic.data)
        icdata = np.zeros(2**ic.length,dtype=complex)
        index = 0
        for obj in icparse:
            icdata[index] = obj["re"] + 1j*obj["im"]
            index += 1

        params = {
                'output_dir': "automated",
                'mode': options["mode"][0],
                'V': options["V"][0],
                'S': lookup[options["R"][0]],
                'L': ic.length,
                'T': options["time"][0],
                'init_state': icdata,
                'IC_name': ic.title,
                'BC':'1'
                }


        sim_tasks = ['one_site','two_site']
        if (ic.length <= 17): sim_tasks.append("bi_partite")
        if (ic.length <= 17): sim_tasks.append("IPR")

        params["fname"] = io.make_file_name(params, iterate = True)

        state_res = time_evolve.run_sim(params, force_rewrite=True, sim_tasks=sim_tasks)
        res = measures.measure(params, state_res, force_rewrite=True)

        result = SimResult.objects.get(pk=options["pk"][0])
        result.completed=True
        result.location=params['fname']
        result.save()


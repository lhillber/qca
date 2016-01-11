from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from qcaadmin.models import *
import json

from django.db.models import Q
import subprocess

import h5py
import numpy as np
import math

import datetime

import simulation.fio as io
import simulation.time_evolve as time_evolve
import simulation.measures as measures

from django.contrib.auth import authenticate

def home(request):
    return render(request,"qcaadmin/home.html")


def iclist(request):
    ics = InitialCondition.objects.all()
    for word in request.GET["filter"].split():
        ics = ics.filter(Q(title__icontains=word) | Q(length__icontains=word))

    result = []
    for ic in ics:
        sims = SimResult.objects.filter(IC=ic,completed=False)
        result.append({
            "pk": ic.pk,
            "length": ic.length,
            "title": ic.title,
            "simrunning": (len(sims) > 0)
            })
    return HttpResponse(json.dumps(result))


def datalist(request):
    sims = SimResult.objects.filter(IC=request.GET["ic"])
    result = []
    for sim in sims:
        result.append({
            "pk": sim.pk,
            "V": sim.V,
            "R": sim.R,
            "isSweep": sim.mode,
            "completed": sim.completed,
            })
    return HttpResponse(json.dumps(result))

def startSimulation(request):
    body = request.body.decode("utf-8")
    data = json.loads(request.body.decode("utf-8"))


    user = authenticate(username="simuser",password=data["Password"])
    if user is None: return HttpResponse("Wrong password.")


    ip = request.META['REMOTE_ADDR']
    if  not (ip == "127.0.0.1" or "131.215." in ip): return HttpResponse("Must be on Caltech or Mines campus.")

    if (len(SimResult.objects.filter(V=data["V"],R=data["R"],IC=data["IC"],mode=(data["isSweep"]))) > 0): return HttpResponse("Simulation already evaluated.")

    if (len(SimResult.objects.filter(completed=False)) >= 4): return HttpResponse("Too many simulations running.")

    time = 50

    ic = InitialCondition.objects.get(pk=data["IC"])
    result = SimResult(V=data["V"],R=data["R"],IC=ic,T=time,location="None",completed=False)
    result.mode = data["isSweep"]

    result.save()
    mode = "block"
    if (data["isSweep"]): mode = "sweep"


    cmd = ["nohup","python"]
    cmd.append("/home/prall/development/qca/manage.py")
    cmd.append("compute")
    cmd.append(str(data["R"]))
    cmd.append(str(data["V"]))
    cmd.append(mode)
    cmd.append(str(data["IC"]))
    cmd.append(str(result.pk))
    cmd.append(str(time))


    #return HttpResponse(json.dumps(cmd))
    subprocess.Popen(cmd)
    return HttpResponse("Launched")



def simStatus(request):
    running = []
    sims = SimResult.objects.filter(completed=False)
    for sim in sims:
        running.append(sim.pk)
    return HttpResponse(json.dumps(running))

def listify(obj):
    if isinstance(obj, h5py.Dataset):
        return listify(list(obj))
    if isinstance(obj, h5py.Group):
        return listify(list(obj))
    if isinstance(obj, np.ndarray):
        return listify(list(obj))
    elif isinstance(obj,list):
        return [listify(x) for x in obj]
    elif isinstance(obj,float) and math.isnan(obj):
        return None
    elif isinstance(obj,complex):
        return {
            "re":obj.real,
            "im":obj.imag
                }
    else:
        return obj

def simData(request):
    simresult = SimResult.objects.get(pk=request.GET["pk"])
    f = h5py.File(simresult.location)

    result = {
            "meta":json.dumps({
                "pk":simresult.pk,
                "length":simresult.IC.length,
                "ic":simresult.IC.pk,
                "ictitle":str(simresult.IC),
                "V":simresult.V,
                "R":simresult.R,
                "T":simresult.T,
                "isSweep":simresult.mode,
                }),
            }




    for key in f.keys():
        if (key == "init_state"): continue
        if ("cut" in key): continue
        data = f[key]


        result[key] = json.dumps(listify(f[key]))

    return HttpResponse(json.dumps(result))



def getICData(request):
    body = request.body.decode("utf-8")
    data = json.loads(request.body.decode("utf-8"))

    if ("pk" in data):
        icobj = InitialCondition.objects.get(pk=data['pk'])
        iclist = json.loads(icobj.data)
        ic = np.zeros(2**icobj.length,dtype=complex)

        index = 0
        for obj in iclist:
            ic[index] = obj["re"] + 1j*obj["im"]
            index += 1

        length = icobj.length
    else:
        length = len(data["compList"][0]["values"])
        ic= np.zeros(2**length,dtype=complex)

        norm = 0
        for component in data["compList"]:
            norm += component["magnitude"]*component["magnitude"]


        for component in data["compList"]:
            factor = np.exp(1j*component["phase"]/16)*component["magnitude"]/np.sqrt(norm)
            index = 0
            for vector in component["values"]: index = index*2 + vector
            ic[index] += factor


    params = {
        'output_dir' : 'IC',
        'mode' : 'alt',
        'L' : length,
        'T' : 1,
        'S' : 0,
        'V' : 'H',
        'init_state': ic,
        'IC_name': 'temp'+str(datetime.datetime.now()),
        'BC': '1'
            }

    params["fname"] = io.make_file_name(params, iterate = True)
    state_res = time_evolve.run_sim(params, force_rewrite=True)
    measures.measure(params, state_res, force_rewrite=True)

    f = h5py.File(params["fname"])
    result = {}
    if ("pk" in data): result["title"] = icobj.title

    for key in f.keys():
        if (key == "init_state"): continue
        if ("cut" in key): continue
        data = f[key]

        result[key] = json.dumps(listify(f[key]))

    return HttpResponse(json.dumps(listify(result)))


def saveIC(request):
    body = request.body.decode("utf-8")
    data = json.loads(request.body.decode("utf-8"))

    length = len(data["compList"][0]["values"])
    ic= np.zeros(2**length,dtype=complex)


    norm = 0
    for component in data["compList"]:
        norm += component["magnitude"]*component["magnitude"]


    for component in data["compList"]:
        factor = np.exp(1j*component["phase"]/16)*component["magnitude"]/np.sqrt(norm)
        index = 0
        for vector in component["values"]: index = index*2 + vector
        ic[index] += factor

    icobj = InitialCondition(title=data["title"],length=length,data=json.dumps(listify(ic)))
    icobj.save()

    return HttpResponse(str(icobj.pk))



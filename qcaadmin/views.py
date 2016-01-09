from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from qcaadmin.models import *
import json

from django.db.models import Q
import subprocess

import h5py
import numpy as np
import math

def home(request):
    #if not request.user.is_authenticated():
    #    return HttpResponseRedirect(reverse("qcaadmin.views.login"))
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

    if (data["Password"] != "muffins"): return HttpResponse("Wrong password.")
    ip = request.META['REMOTE_ADDR']
    if  not (ip == "127.0.0.1" or "131.215." in ip): return HttpResponse("Must be on Caltech or Mines campus.")

    if (len(SimResult.objects.filter(V=data["V"],R=data["R"],IC=data["IC"],mode=(data["isSweep"]))) > 0): return HttpResponse("Simulation already evaluated.")

    if (len(SimResult.objects.filter(completed=False)) >= 4): return HttpResponse("Too many simulations running.")


    mode = "block"
    if (data["isSweep"]): mode = "sweep"

    cmd = ["nohup","python"]
    cmd.append("/home/prall/development/qca/manage.py")
    cmd.append("compute")
    cmd.append(str(data["R"]))
    cmd.append(str(data["V"]))
    cmd.append(mode)
    cmd.append(str(data["IC"]))


    #return HttpResponse(json.dumps(cmd))
    subprocess.Popen(cmd)
    return HttpResponse("Launched")



def simStatus(request):
    running = []
    sims = SimResult.objects.filter(completed=False)
    for sim in sims:
        running.append(sim.pk)
    return HttpResponse(json.dumps(running))



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

    for key in f.keys():
        data = f[key]


        result[key] = json.dumps(listify(f[key]))

    return HttpResponse(json.dumps(result))




def getICData(request):
    return HttpResponseRedirect("/")

def setICData(request):
    return HttpResponseRedirect("/")



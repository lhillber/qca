from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect

def home(request):
    #if not request.user.is_authenticated():
    #    return HttpResponseRedirect(reverse("qcaadmin.views.login"))
    return render(request,"qcaadmin/home.html")




def login(request):
    return HttpResponseRedirect("/")

def logout(request):
    return HttpResponseRedirect("/")

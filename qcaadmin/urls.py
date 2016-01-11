from django.conf.urls import patterns, include, url
import qcaadmin.views

urlpatterns = [
    url(r'^$', qcaadmin.views.home),
    url(r'^iclist/$', qcaadmin.views.iclist),
    url(r'^datalist/$', qcaadmin.views.datalist),
    url(r'^startSimulation/$', qcaadmin.views.startSimulation),
    url(r'^simStatus/$', qcaadmin.views.simStatus),
    url(r'^simData/$', qcaadmin.views.simData),
    url(r'^getICData/$', qcaadmin.views.getICData),
    url(r'^saveIC/$', qcaadmin.views.saveIC),
]

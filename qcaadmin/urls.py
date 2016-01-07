from django.conf.urls import patterns, include, url
import qcaadmin.views

urlpatterns = [
    url(r'^$', qcaadmin.views.home),
    url(r'^login/$', qcaadmin.views.login),
    url(r'^logout/$', qcaadmin.views.logout),
]

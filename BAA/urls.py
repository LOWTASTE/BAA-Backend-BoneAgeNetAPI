from django.conf.urls import url
from rest_framework.urlpatterns import format_suffix_patterns

from BAA import views

urlpatterns = [
    # url(r'^snippets/$', views.snippet_list),
    # url(r'^snippets/(?P<pk>[0-9]+)/$', views.snippet_detail),
    url(r'^BAAModel/$', views.BAAModelAPI.as_view()),
    url(r'^BAAModel_Data/$', views.BAAModel_DataAPI.as_view()),
    url(r'^PredictAPI/$', views.PredictAPI.as_view()),
    url(r'^TrainAPI/$', views.TrainAPI.as_view()),
    url(r'^AlgorithmAPI/$', views.AlgorithmAPI.as_view()),
    url(r'^Testtt/$', views.TestAPI.as_view()),
    # url(r'^BAAModelInfoAPI/$', views.BAAModelInfoAPI.as_view()),
    # url(r'^users/$', views.UserList.as_view()),
    # url(r'^users/(?P<pk>[0-9]+)/$', views.UserDetail.as_view()),
]

urlpatterns = format_suffix_patterns(urlpatterns)

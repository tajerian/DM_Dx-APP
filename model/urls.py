from django.urls import path
from . import views

app_name = 'model'

urlpatterns = [
    path('', views.home, name= 'model'),
]

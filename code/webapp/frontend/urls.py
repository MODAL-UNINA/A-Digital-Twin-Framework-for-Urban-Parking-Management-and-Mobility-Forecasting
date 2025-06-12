from django.contrib.auth.views import LogoutView
from django.urls import path

from . import views

urlpatterns = [
    path("logout/", LogoutView.as_view(), name="logout"),
    path("", views.main, name="main"),
]

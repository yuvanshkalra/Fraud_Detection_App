from django.urls import re_path

from . import views

# re_path with an optional trailing slash: Retrofit posts to `/v1/detect`
# (no slash), but curl users habitually add one.
urlpatterns = [
    re_path(r"^$", views.index),
    re_path(r"^healthz/?$", views.healthz),
    re_path(r"^v1/detect/?$", views.DetectView.as_view()),
    re_path(r"^v1/transcribe/?$", views.TranscribeView.as_view()),
]

from django.urls import path
from . import views

app_name = 'dashboard'

urlpatterns = [
    path('', views.upload_cohort, name='upload'),
    path('results/<int:cohort_id>/', views.results_view, name='results'),
    path('download/<int:cohort_id>/', views.download_report, name='download'),
]

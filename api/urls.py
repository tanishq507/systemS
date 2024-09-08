from django.urls import path
from api import views

urlpatterns = [
    path('',views.index_page),
    path('predict', views.predict_diabetictype),
    path('helloesp', views.hello),
    path('data', views.data),
    path('RUL', views.RUL),
]
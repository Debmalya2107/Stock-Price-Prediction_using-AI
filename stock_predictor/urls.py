"""
URL Configuration for Stock Predictor App
"""

from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('predict/', views.predict_stock_price, name='predict_stock_price'),
    path('model-info/', views.get_model_info, name='model_info'),
]

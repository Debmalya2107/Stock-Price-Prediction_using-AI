from django.contrib import admin
from .models import StockPrediction, ModelMetrics

@admin.register(StockPrediction)
class StockPredictionAdmin(admin.ModelAdmin):
    list_display = ['ticker_symbol', 'predicted_price', 'confidence_score', 'created_at']
    list_filter = ['ticker_symbol', 'created_at']
    search_fields = ['ticker_symbol']
    ordering = ['-created_at']

@admin.register(ModelMetrics)
class ModelMetricsAdmin(admin.ModelAdmin):
    list_display = ['model_name', 'r2_score', 'mae', 'mse', 'created_at']
    list_filter = ['model_name', 'created_at']
    ordering = ['-created_at']

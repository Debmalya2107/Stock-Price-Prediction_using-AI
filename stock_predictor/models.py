from django.db import models

class StockPrediction(models.Model):
    """Model to store stock prediction results"""
    ticker_symbol = models.CharField(max_length=10)
    predicted_price = models.DecimalField(max_digits=10, decimal_places=2)
    confidence_score = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.ticker_symbol}: ${self.predicted_price}"

class ModelMetrics(models.Model):
    """Model to store ML model evaluation metrics"""
    model_name = models.CharField(max_length=50, default="LSTM")
    mse = models.FloatField()
    mae = models.FloatField()
    r2_score = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.model_name} - R²: {self.r2_score:.4f}"

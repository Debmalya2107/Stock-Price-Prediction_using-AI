from django.apps import AppConfig

class StockPredictorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'stock_predictor'

    def ready(self):
        """Load ML model when Django starts"""
        try:
            from .views import load_ml_model
            load_ml_model()
            print("🚀 ML model pre-loaded successfully!")
        except Exception as e:
            print(f"⚠️ Could not pre-load ML model: {e}")

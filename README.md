# 📈 Stock-Price-Prediction_using-AI

A **full-stack Django web application** that predicts next-day stock prices using **LSTM neural networks** with real-time data integration and interactive visualizations.  
It provides an **end-to-end machine learning pipeline**, making complex financial forecasting accessible through a clean and user-friendly web interface.

---

## 💡 Usage

### Web Interface
1. Enter a stock ticker (e.g., `AAPL`, `TSLA`)  
2. Click **Predict Price**  
3. View:
   - Predicted next-day price  
   - Interactive historical chart  
   - Model accuracy metrics  

### API Endpoints
- `GET /` → Main interface  
- `POST /predict/` → Returns JSON prediction  

Example JSON output:

{
  "ticker": "AAPL",
  "predicted_price": 152.45,
  "current_price": 150.30,
  "prediction_date": "2024-08-25",
  "confidence_metrics": {
    "mse": 2.15,
    "mae": 1.23,
    "r2_score": 0.87
  }
}
#  Project Structure
stock_prediction_webapp/
├── manage.py
├── requirements.txt
├── README.md
├── ml_pipeline.py
│
├── stock_prediction/         # Django project config
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
│
├── stock_predictor/          # Main app
│   ├── models.py
│   ├── views.py
│   ├── urls.py
│   ├── templates/
│   │   └── index.html
│   └── migrations/
│
├── ml_models/                # Trained ML models
│   ├── lstm_model.h5
│   └── scaler.pkl
│
├── data/
│   └── sample_stock_data.csv
└── static/

#  Deployment
# Clone repo
git clone https://github.com/Debmalya2107/Stock-Price-Prediction_using-AI


# Install dependencies
pip install -r requirements.txt

# Migrate & collect static
python manage.py migrate
python manage.py collectstatic

# Start server
python manage.py runserver

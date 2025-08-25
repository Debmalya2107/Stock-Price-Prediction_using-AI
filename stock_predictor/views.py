"""
Views for Stock Prediction Django App
Phase 4: Django Web Application Integration
"""

import json
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import tensorflow as tf
import joblib
import yfinance as yf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Global variables to store loaded model and scaler
_model = None
_scaler = None

def load_ml_model():
    """Load the trained LSTM model and scaler once when server starts"""
    global _model, _scaler

    if _model is None or _scaler is None:
        try:
            # Load the trained LSTM model
            model_path = os.path.join(settings.ML_MODELS_PATH, 'lstm_model.h5')
            _model = tf.keras.models.load_model(model_path)
            print(f"✅ LSTM model loaded from: {model_path}")

            # Load the fitted MinMaxScaler
            scaler_path = os.path.join(settings.ML_MODELS_PATH, 'scaler.pkl')
            _scaler = joblib.load(scaler_path)

            print(f"✅ MinMaxScaler loaded from: {scaler_path}")

        except Exception as e:
            print(f"❌ Error loading ML model/scaler: {str(e)}")
            raise e

    return _model, _scaler

def index(request):
    """Render the main page with stock prediction interface"""
    return render(request, 'index.html')

@csrf_exempt
@require_http_methods(["POST"])
def predict_stock_price(request):
    """
    Handle stock price prediction requests

    Phase 4: Django Web Application Integration
    - Load pre-trained LSTM model and scaler
    - Handle POST requests from form
    - Fetch latest historical stock data
    - Preprocess data using loaded scaler
    - Make prediction using model
    - Return JsonResponse with results
    """
    try:
        # Parse request data
        data = json.loads(request.body)
        ticker_symbol = data.get('ticker', '').upper().strip()

        if not ticker_symbol:
            return JsonResponse({
                'success': False,
                'error': 'Please provide a valid stock ticker symbol'
            })

        # Load model and scaler
        model, scaler = load_ml_model()

        # Fetch latest historical stock data
        print(f"📊 Fetching data for {ticker_symbol}...")

        # Try to fetch real data, fallback to sample data if fails
        try:
            stock = yf.Ticker(ticker_symbol)
            hist_data = stock.history(period="1y")  # Get 1 year of data

            if hist_data.empty:
                raise ValueError(f"No data found for ticker {ticker_symbol}")

            hist_data.reset_index(inplace=True)

        except Exception as yf_error:
            print(f"⚠️ yfinance error: {yf_error}")
            print("📁 Using sample data instead...")

            # Use sample data as fallback
            sample_path = os.path.join(settings.DATA_PATH, 'sample_stock_data.csv')
            hist_data = pd.read_csv(sample_path)
            hist_data['Date'] = pd.to_datetime(hist_data['Date'])
            ticker_symbol = "SAMPLE"  # Update ticker to indicate sample data

        # Preprocess data exactly as training data was processed
        print("🔧 Preprocessing data...")

        # Use closing prices for prediction
        prices = hist_data['Close'].values.reshape(-1, 1)

        # Scale the data using the loaded scaler
        scaled_prices = scaler.transform(prices)

        # Create sequence for prediction (last 60 days)
        sequence_length = 60

        if len(scaled_prices) < sequence_length:
            return JsonResponse({
                'success': False,
                'error': f'Insufficient data. Need at least {sequence_length} days of historical data.'
            })

        # Get the last 60 days for prediction
        last_sequence = scaled_prices[-sequence_length:]
        last_sequence = np.reshape(last_sequence, (1, sequence_length, 1))

        # Make prediction
        print("🔮 Making prediction...")
        scaled_prediction = model.predict(last_sequence)

        # Inverse transform to get actual price
        predicted_price = scaler.inverse_transform(scaled_prediction)[0][0]

        # Calculate confidence metrics based on recent price volatility
        recent_prices = hist_data['Close'].tail(30).values
        volatility = np.std(recent_prices)
        confidence = max(0.5, min(0.95, 1 - (volatility / np.mean(recent_prices))))

        # Prepare historical data for chart (last 30 days)
        chart_data = hist_data.tail(30).copy()
        chart_data['Date'] = chart_data['Date'].dt.strftime('%Y-%m-%d')

        # Calculate model evaluation metrics using recent data
        metrics = calculate_model_metrics(hist_data, model, scaler)

        # Prepare response
        response_data = {
            'success': True,
            'ticker': ticker_symbol,
            'predicted_price': round(float(predicted_price), 2),
            'confidence': round(confidence, 4),
            'current_price': round(float(hist_data['Close'].iloc[-1]), 2),
            'price_change': round(float(predicted_price - hist_data['Close'].iloc[-1]), 2),
            'historical_data': {
                'dates': chart_data['Date'].tolist(),
                'prices': [round(price, 2) for price in chart_data['Close'].tolist()],
                'volumes': chart_data['Volume'].tolist()
            },
            'metrics': metrics,
            'prediction_date': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        }

        print(f"✅ Prediction completed: ${predicted_price:.2f}")
        return JsonResponse(response_data)

    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        })

def calculate_model_metrics(data, model, scaler):
    """Calculate model evaluation metrics for display"""
    try:
        # Use last portion of data for evaluation
        prices = data['Close'].values.reshape(-1, 1)
        scaled_prices = scaler.transform(prices)

        # Create sequences
        sequence_length = 60
        X, y = [], []

        for i in range(sequence_length, len(scaled_prices)):
            X.append(scaled_prices[i-sequence_length:i, 0])
            y.append(scaled_prices[i, 0])

        if len(X) < 10:  # Need minimum data for evaluation
            return {'mse': 0, 'mae': 0, 'r2_score': 0}

        X = np.array(X)
        y = np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        # Use last 20% for evaluation
        eval_size = max(1, len(X) // 5)
        X_eval = X[-eval_size:]
        y_eval = y[-eval_size:]

        # Make predictions
        y_pred = model.predict(X_eval)

        # Transform back to original scale
        y_eval_orig = scaler.inverse_transform(y_eval.reshape(-1, 1))
        y_pred_orig = scaler.inverse_transform(y_pred)

        # Calculate metrics
        mse = mean_squared_error(y_eval_orig, y_pred_orig)
        mae = mean_absolute_error(y_eval_orig, y_pred_orig)
        r2 = r2_score(y_eval_orig, y_pred_orig)

        return {
            'mse': round(float(mse), 4),
            'mae': round(float(mae), 4),
            'r2_score': round(float(r2), 4)
        }

    except Exception as e:
        print(f"⚠️ Metrics calculation error: {e}")
        return {'mse': 0, 'mae': 0, 'r2_score': 0}

@require_http_methods(["GET"])
def get_model_info(request):
    """Return information about the loaded ML model"""
    try:
        model, scaler = load_ml_model()

        return JsonResponse({
            'success': True,
            'model_info': {
                'type': 'LSTM Neural Network',
                'framework': 'TensorFlow/Keras',
                'input_shape': str(model.input_shape),
                'output_shape': str(model.output_shape),
                'parameters': int(model.count_params()),
                'scaler_range': list(scaler.feature_range)
            }
        })

    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        })

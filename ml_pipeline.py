"""
Complete Machine Learning Pipeline for Stock Price Prediction
Following the 4-Phase ML Structure as specified in the project requirements.

Phase 1: Data Preprocessing/Transform
Phase 2: Data Modeling  
Phase 3: Evaluation Phase
Phase 4: Django Web Application Integration (implemented separately)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import yfinance as yf
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class StockPredictionPipeline:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.sequence_length = 60
        self.train_data = None
        self.test_data = None

    def phase1_data_preprocessing(self, ticker_symbol="AAPL", period="2y", use_sample_data=False):
        """
        Phase 1: Data Preprocessing/Transform
        - Data Collection from yfinance or sample CSV
        - Import Data using pandas
        - Clean Data and handle missing values
        - Feature Engineering
        - Data Visualization
        - Variable Definition (X and Y)
        - Data Split (80/20)
        """
        print("=== PHASE 1: DATA PREPROCESSING/TRANSFORM ===")

        # 1. Data Collection
        if use_sample_data:
            print("📊 Loading sample data from CSV...")
            self.data = pd.read_csv('data/sample_stock_data.csv')
            self.data['Date'] = pd.to_datetime(self.data['Date'])
        else:
            print(f"📊 Fetching historical data for {ticker_symbol}...")
            stock = yf.Ticker(ticker_symbol)
            self.data = stock.history(period=period)
            self.data.reset_index(inplace=True)

        print(f"✅ Data collected: {len(self.data)} records")

        # 2. Import Data (already done with pandas)
        print("✅ Data imported using pandas")

        # 3. Clean Data
        print("🧹 Cleaning data and handling missing values...")
        initial_count = len(self.data)
        self.data = self.data.dropna()
        final_count = len(self.data)
        print(f"✅ Removed {initial_count - final_count} rows with missing values")

        # 4. Feature Engineering
        print("🔧 Engineering new features...")
        self.data['SMA_20'] = self.data['Close'].rolling(window=20).mean()
        self.data['SMA_50'] = self.data['Close'].rolling(window=50).mean()
        self.data['Daily_Return'] = self.data['Close'].pct_change()
        self.data['Volatility'] = self.data['Daily_Return'].rolling(window=20).std()

        # Remove NaN values created by rolling calculations
        self.data = self.data.dropna()
        print("✅ Features engineered: SMA_20, SMA_50, Daily_Return, Volatility")

        # 5. Data Visualization
        print("📈 Generating data visualizations...")
        self._create_visualizations()

        # 6. Variable Definition
        print("🎯 Variable Definition:")
        print("   Independent Variables (X): Historical time-series data (60-day sequences of closing prices)")
        print("   Dependent Variable (Y): Next day closing price")

        # Prepare data for LSTM (using Close prices)
        prices = self.data['Close'].values.reshape(-1, 1)

        # Scale the data
        scaled_prices = self.scaler.fit_transform(prices)

        # Create sequences for LSTM
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_prices)):
            X.append(scaled_prices[i-self.sequence_length:i, 0])
            y.append(scaled_prices[i, 0])

        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        # 7. Data Split (80/20)
        print("✂️ Splitting data into training (80%) and testing (20%) sets...")
        split_index = int(len(X) * 0.8)

        self.X_train, self.X_test = X[:split_index], X[split_index:]
        self.y_train, self.y_test = y[:split_index], y[split_index:]

        print(f"✅ Training set: {len(self.X_train)} samples")
        print(f"✅ Testing set: {len(self.X_test)} samples")

        return self.X_train, self.X_test, self.y_train, self.y_test

    def _create_visualizations(self):
        """Create comprehensive data visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Stock Data Analysis and Visualization', fontsize=16, fontweight='bold')

        # Price trends
        axes[0, 0].plot(self.data['Date'], self.data['Close'], label='Close Price', linewidth=2)
        axes[0, 0].plot(self.data['Date'], self.data['SMA_20'], label='SMA 20', alpha=0.7)
        axes[0, 0].plot(self.data['Date'], self.data['SMA_50'], label='SMA 50', alpha=0.7)
        axes[0, 0].set_title('Price Trends with Moving Averages')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Price ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Volume analysis
        axes[0, 1].bar(self.data['Date'], self.data['Volume'], alpha=0.6, width=1)
        axes[0, 1].set_title('Trading Volume Over Time')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Volume')
        axes[0, 1].grid(True, alpha=0.3)

        # Daily returns distribution
        axes[1, 0].hist(self.data['Daily_Return'].dropna(), bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Daily Returns Distribution')
        axes[1, 0].set_xlabel('Daily Return')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)

        # Price vs Volume correlation
        axes[1, 1].scatter(self.data['Volume'], self.data['Close'], alpha=0.5)
        axes[1, 1].set_title('Price vs Volume Correlation')
        axes[1, 1].set_xlabel('Volume')
        axes[1, 1].set_ylabel('Close Price ($)')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('stock_data_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("✅ Visualizations created and saved")

    def phase2_data_modeling(self):
        """
        Phase 2: Data Modeling
        - Model Selection (LSTM for time-series regression)
        - Model Training using Training Data
        - Prediction on Test Data
        """
        print("\n=== PHASE 2: DATA MODELING ===")

        # 1. Model Selection
        print("🤖 Model Selection: LSTM Neural Network")
        print("   Problem Type: Regression (continuous output)")
        print("   Algorithm: Long Short-Term Memory (LSTM)")
        print("   Framework: TensorFlow/Keras")

        # Build LSTM model
        self.model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(self.sequence_length, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])

        self.model.compile(optimizer='adam', loss='mean_squared_error')

        print("✅ LSTM model architecture created")
        print(f"   Input shape: {self.model.input_shape}")
        print(f"   Output shape: {self.model.output_shape}")
        print(f"   Total parameters: {self.model.count_params()}")

        # 2. Model Training
        print("🏋️ Training LSTM model on Training Data...")
        history = self.model.fit(
            self.X_train, self.y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.1,
            verbose=1,
            shuffle=False  # Important for time series
        )

        print("✅ Model training completed")

        # Plot training history
        self._plot_training_history(history)

        # 3. Prediction on Test Data
        print("🔮 Making predictions on Test Data...")
        self.y_pred = self.model.predict(self.X_test)

        print("✅ Predictions generated")

        return self.model, self.y_pred

    def _plot_training_history(self, history):
        """Plot model training history"""
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

    def phase3_evaluation(self):
        """
        Phase 3: Evaluation Phase
        - Evaluate Model Result on Test Data
        - Generate Prediction Score Checking Techniques:
          * Mean Squared Error (MSE)
          * Mean Absolute Error (MAE)  
          * R2 Score
        """
        print("\n=== PHASE 3: EVALUATION PHASE ===")

        # Transform predictions back to original scale
        y_test_original = self.scaler.inverse_transform(self.y_test.reshape(-1, 1))
        y_pred_original = self.scaler.inverse_transform(self.y_pred)

        # Calculate evaluation metrics
        mse = mean_squared_error(y_test_original, y_pred_original)
        mae = mean_absolute_error(y_test_original, y_pred_original)
        r2 = r2_score(y_test_original, y_pred_original)
        rmse = np.sqrt(mse)

        # Print evaluation results
        print("📊 Model Performance Evaluation on Test Data:")
        print("=" * 50)
        print(f"Mean Squared Error (MSE):     ${mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): ${rmse:.4f}")
        print(f"Mean Absolute Error (MAE):    ${mae:.4f}")
        print(f"R² Score:                     {r2:.4f}")
        print("=" * 50)

        # Interpretation
        print("\n📈 Model Performance Interpretation:")
        if r2 > 0.8:
            print("🟢 Excellent model performance (R² > 0.8)")
        elif r2 > 0.6:
            print("🟡 Good model performance (R² > 0.6)")
        elif r2 > 0.4:
            print("🟠 Fair model performance (R² > 0.4)")
        else:
            print("🔴 Poor model performance (R² < 0.4)")

        print(f"   - On average, predictions are off by ${mae:.2f}")
        print(f"   - Model explains {r2*100:.1f}% of price variance")

        # Create prediction visualization
        self._plot_predictions(y_test_original, y_pred_original)

        # Save evaluation metrics
        metrics = {
            'MSE': float(mse),
            'RMSE': float(rmse),
            'MAE': float(mae),  
            'R2_Score': float(r2)
        }

        return metrics

    def _plot_predictions(self, y_true, y_pred):
        """Plot actual vs predicted prices"""
        plt.figure(figsize=(15, 8))

        # Plot actual vs predicted
        plt.subplot(2, 1, 1)
        plt.plot(y_true, label='Actual Prices', linewidth=2, alpha=0.8)
        plt.plot(y_pred, label='Predicted Prices', linewidth=2, alpha=0.8)
        plt.title('Actual vs Predicted Stock Prices (Test Data)', fontsize=14, fontweight='bold')
        plt.xlabel('Time Steps')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot prediction errors
        plt.subplot(2, 1, 2)
        errors = y_true.flatten() - y_pred.flatten()
        plt.plot(errors, color='red', alpha=0.7, linewidth=1)
        plt.title('Prediction Errors (Actual - Predicted)', fontsize=14, fontweight='bold')
        plt.xlabel('Time Steps')
        plt.ylabel('Error ($)')
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('prediction_results.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("✅ Prediction visualizations created and saved")

    def save_model_and_scaler(self, model_path='ml_models/lstm_model.h5', scaler_path='ml_models/scaler.pkl'):
        """Save trained model and scaler for Django integration"""
        # Save model
        self.model.save(model_path)
        print(f"✅ Model saved to: {model_path}")

        # Save scaler
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"✅ Scaler saved to: {scaler_path}")

    def run_complete_pipeline(self, ticker_symbol="AAPL", use_sample_data=True):
        """Run the complete 3-phase ML pipeline"""
        print("🚀 Starting Complete Stock Prediction ML Pipeline")
        print("=" * 60)

        try:
            # Phase 1: Data Preprocessing
            self.phase1_data_preprocessing(ticker_symbol, use_sample_data=use_sample_data)

            # Phase 2: Data Modeling
            self.phase2_data_modeling()

            # Phase 3: Evaluation
            metrics = self.phase3_evaluation()

            # Save for Django integration
            self.save_model_and_scaler()

            print("\n🎉 Pipeline completed successfully!")
            print("=" * 60)

            return metrics

        except Exception as e:
            print(f"❌ Pipeline failed: {str(e)}")
            return None

# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = StockPredictionPipeline()

    # Run complete pipeline with sample data
    print("Running ML Pipeline with Sample Data...")
    metrics = pipeline.run_complete_pipeline(use_sample_data=True)

    if metrics:
        print(f"\nFinal Results:")
        for metric, value in metrics.items():
            print(f"{metric}: {value}")
